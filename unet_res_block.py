# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm, split_args
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.dynunet_block import get_conv_layer

from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer

from monai.utils import ensure_tuple_rep, look_up_option, optional_import
rearrange, _ = optional_import("einops", name="rearrange")
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")



class AttentionBlock(nn.Module):

    def __init__(
            self, 
            spatial_dims: int, 
            f_int: int, 
            f_g: int, 
            f_l: int, 
            dropout=0.0,
            act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            ):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )
        self.lrelu = get_act_layer(name=act_name)


    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.lrelu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class SABlock(nn.Module):
    """
    A self-attention block for 3D data.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
            save_attn (bool, optional): to make accessible the attention matrix. Defaults to False.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.save_attn = save_attn
        self.att_mat = torch.Tensor()


    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1) 

        # Linear transformation to obtain queries, keys, and values
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # Split qkv into q, k, v along the last dimension

        # Reshape q, k, v for self-attention
        q = q.view(b, d * h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(b, d * h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, d * h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        att_mat = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        if self.save_attn:
            self.att_mat = att_mat.detach()  # Save attention matrix

        # Apply dropout and softmax along the sequence length dimension
        att_mat = self.drop_weights(torch.nn.functional.softmax(att_mat, dim=-1))

        # Weighted sum of values using attention scores
        x = torch.einsum("bhqk,bhkd->bhqd", att_mat, v)

        # Reshape output to original shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, d, h, w, c)

        # Linear projection and dropout
        x = self.out_proj(x)
        x = self.drop_output(x)

        x = x.reshape(b, c, d, h, w)
        return x

class SSE(nn.Module):
    """
    Re-implementation of the Squeeze-and-Excitation block based on:
    "Hu et al., Squeeze-and-Excitation Networks, https://arxiv.org/abs/1709.01507".
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        act_name: tuple[str, dict] | str = "sigmoid",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2, or 3.
            in_channels: number of input channels.
            act_name: activation type of the output squeeze layer. Defaults to "sigmoid".

        """
        super().__init__()
        self.conv = get_conv_layer(
            spatial_dims,
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            dropout=0.0,
            act=None,
            norm=None,
            conv_only=True,
        )
        self.sigmoid = get_act_layer(name=act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape (batch, in_channels, spatial_1[, spatial_2, ...]).
        """
        y = self.conv(x)
        y = self.sigmoid(y)
        result = x * y
        return result

class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        se_layer = False,
        scse_layer = False
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            act=None,
            norm=None,
            conv_only=False,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if se_layer:
            self.se_layer = SSE(
                spatial_dims=spatial_dims, in_channels=out_channels
            )
        elif scse_layer:
            self.scse_layer = ChannelSELayer(
                spatial_dims=spatial_dims, in_channels=out_channels
            )
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout=dropout,
                act=None,
                norm=None,
                conv_only=False,
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)


    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        if hasattr(self, "se_layer"): 
            out = self.se_layer(out)
        if hasattr(self, "scse_layer"): 
            out = self.scse_layer(out)
        out += residual
        out = self.lrelu(out)
        return out

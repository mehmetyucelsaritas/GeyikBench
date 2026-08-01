"""
/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2024, ITU/ISO/IEC
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Union, Tuple, Optional, Type, Iterable, List, Dict
import torch
from torch import nn
from torch.nn import functional as F

# ugly global to avoid putting the export model inside the model
model_for_export = None


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        is_separable: bool = False,
        hidden_separable_channels: Optional[int] = None,
        post_activation: Optional[Type] = nn.PReLU,
        index: int = 0,
        groups: int = 1,
        **kwargs,
    ):
        """
        Args:
            in_channels: the number of input channels
            out_channels: the number of output channels
            kernel_size: the convolution's kernel size
            stride: the convolution's stride(s)
            padding: the convolution's padding
            is_separable: whether to implement convolution separably
            hidden_separable_channels: If is_separable, the number of hidden channels between convolutions. If None, use out_channels
            post_activation: activation function to use after convolution. If None, no activation after convolution
            **kwargs: additional kwargs to pass to nn.Conv2d
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        if padding is not None:
            self.padding = (padding, padding) if isinstance(padding, int) else padding
        else:
            self.padding = tuple([k // 2 for k in self.kernel_size])
        self.is_separable = is_separable
        self.post_activation = post_activation

        if self.is_separable:
            self.hidden_separable_channels = hidden_separable_channels or out_channels
            if index % 2 == 0:
                modules = [
                    nn.Conv2d(
                        self.in_channels,
                        self.hidden_separable_channels,
                        (self.kernel_size[0], 1),
                        (self.stride[0], 1),
                        (self.padding[0], 0),
                        groups=groups,
                        **kwargs,
                    ),
                    nn.Conv2d(
                        self.hidden_separable_channels,
                        self.out_channels,
                        (1, self.kernel_size[1]),
                        (1, self.stride[1]),
                        (0, self.padding[1]),
                        **kwargs,
                    ),
                ]
            else:
                modules = [
                    nn.Conv2d(
                        self.in_channels,
                        self.hidden_separable_channels,
                        (1, self.kernel_size[1]),
                        (1, self.stride[1]),
                        (0, self.padding[1]),
                        groups=groups,
                        **kwargs,
                    ),
                    nn.Conv2d(
                        self.hidden_separable_channels,
                        self.out_channels,
                        (self.kernel_size[0], 1),
                        (self.stride[0], 1),
                        (self.padding[0], 0),
                        **kwargs,
                    ),
                ]
        else:
            modules = [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    **kwargs,
                )
            ]

        if self.post_activation is not None:
            modules.append(self.post_activation())

        super(Conv, self).__init__(*modules)


class MultiBranchModule(nn.Module):
    """A module representing multple, parallel branches. If the input is a list, each element in the list is fed into the corresponding branch,
    otherwise the input is fed into every branch. The outputs of each branch are then merged."""

    def __init__(self, *branch_modules, merge_dimension: int = -3):
        """
        Args:
            branch_modules: modules to run in parallel
            merge_dimension: the dimension to merge outputs from each branch
        """
        super().__init__()
        self.branches = nn.ModuleList(branch_modules)
        self.merge_dimension = merge_dimension

    def forward(self, args: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        inputs = args if isinstance(args, list) else len(self.branches) * [args]
        branch_outputs = [branch(input) for branch, input in zip(self.branches, inputs)]
        return torch.cat(branch_outputs, dim=self.merge_dimension)


class Attn(nn.Module):
    def __init__(self, to_train, c, n_h):
        super(Attn, self).__init__()
        self.n_h = n_h
        self.c = c
        self.to_train = to_train
        self.conv1_1 = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.conv1_3 = nn.Conv2d(c, c, kernel_size=1, bias=False)
        group = 8
        self.conv3_1 = nn.Conv2d(
            c, c, kernel_size=(3, 3), padding=(1, 1), groups=group, bias=False
        )
        self.conv3_2 = nn.Conv2d(
            c, c, kernel_size=(3, 3), padding=(1, 1), groups=group, bias=False
        )
        self.conv3_3 = nn.Conv2d(
            c, c, kernel_size=(3, 3), padding=(1, 1), groups=group, bias=False
        )
        self.conv2 = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x):
        q = self.conv3_1(self.conv1_1(x))
        k = self.conv3_2(self.conv1_2(x))
        v = self.conv3_3(self.conv1_3(x))
        s = q.shape
        if self.to_train:
            q = q.reshape(s[0], self.n_h, self.c // self.n_h, -1)
            k = k.reshape(s[0], self.n_h, self.c // self.n_h, -1)
            v = v.reshape(s[0], self.n_h, self.c // self.n_h, -1)
        else:
            q = q.reshape(1, self.n_h, self.c // self.n_h, -1)
            k = k.reshape(1, self.n_h, self.c // self.n_h, -1)
            v = v.reshape(1, self.n_h, self.c // self.n_h, -1)
        map = torch.matmul(q, k.transpose(-2, -1))
        p = torch.matmul(map, v)
        p = p.reshape(s)
        p = self.conv2(p)
        return p


class TFBlock(nn.Module):
    def __init__(self, to_train, c=64, n_h=2):
        super(TFBlock, self).__init__()

        self.attention = Attn(to_train, c, n_h)

    def forward(self, x):
        x = x + self.attention(x)
        return x


class ResidualBlock(nn.Sequential):
    def __init__(
        self,
        C: int = 64,
        C1: int = 160,
        C21: int = 32,
        C22: int = 32,
        C31: int = 64,
        index: int = 0,
    ):
        super(ResidualBlock, self).__init__(
            MultiBranchModule(
                Conv(C, C1, kernel_size=1),
                Conv(
                    C,
                    C22,
                    kernel_size=3,
                    is_separable=True,
                    hidden_separable_channels=C21,
                    index=index,
                    groups=2,
                ),
            ),
            Conv(C1 + C22, C, kernel_size=1, post_activation=None),
            Conv(
                C,
                C,
                kernel_size=3,
                post_activation=None,
                is_separable=True,
                hidden_separable_channels=C31,
                index=index,
                groups=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super(ResidualBlock, self).forward(x)


class ResidualBlock_TF(nn.Sequential):
    def __init__(
        self,
        to_train: bool,
        C: int = 64,
        C1: int = 160,
        C21: int = 32,
        C22: int = 32,
        C31: int = 64,
        index: int = 0,
    ):
        super(ResidualBlock_TF, self).__init__(
            MultiBranchModule(
                Conv(C, C1, kernel_size=1),
                Conv(
                    C,
                    C22,
                    kernel_size=3,
                    is_separable=True,
                    hidden_separable_channels=C21,
                    index=index,
                    groups=2,
                ),
            ),
            Conv(C1 + C22, C, kernel_size=1, post_activation=None),
            Conv(
                C,
                C,
                kernel_size=3,
                post_activation=None,
                is_separable=True,
                hidden_separable_channels=C31,
                index=index,
                groups=2,
            ),
            TFBlock(to_train, C, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super(ResidualBlock_TF, self).forward(x)


class SADLNet(nn.Sequential):
    """The network used during SADL inference"""

    def __init__(
        self,
        to_train: bool = True,
        input_channels: Iterable[int] = [3, 3, 3, 1, 1, 1],
        input_kernels: Iterable[int] = [3, 3, 3, 1, 1, 3],
        D1: int = 192,
        D2: int = 32,
        D3: int = 16,
        D4: int = 16,
        D5: int = 16,
        D6: int = 48,
        N: int = 20,
        C: int = 64,
        C1: int = 160,
        C21: int = 32,
        C22: int = 32,
        C31: int = 64,
        output_channels: int = 6,
    ):
        """
        Args:
            input_channels: the number of channels expected for each input
            input_kernels: the kernel size for each input convolution
            output_channels: the number of output channels
        """
        self.input_channels = input_channels
        self.input_kernels = input_kernels
        self.input_features = [D1, D2, D3, D4, D4, D5]
        super(SADLNet, self).__init__(
            MultiBranchModule(
                *[
                    Conv(c, d, kernel_size=k)
                    for c, d, k in zip(
                        self.input_channels, self.input_features, self.input_kernels
                    )
                ]
            ),
            Conv(sum(self.input_features), D6, kernel_size=1),
            Conv(D6, C, kernel_size=3, stride=2),
            *[ResidualBlock(C, C1, C21, C22, C31, index=i) for i in range(7)],
            ResidualBlock_TF(to_train, C, C1, C21, C22, C31, index=7),
            *[ResidualBlock(C, C1, C21, C22, C31, index=i) for i in range(8, 14)],
            ResidualBlock_TF(to_train, C, C1, C21, C22, C31, index=14),
            *[ResidualBlock(C, C1, C21, C22, C31, index=i) for i in range(15, N)],
            Conv(C, C, kernel_size=3),
            Conv(C, output_channels, kernel_size=3, post_activation=None),
        )
        # model for export: batch size=1 and no dynamic axis on reshape
        if to_train:
            global model_for_export
            model_for_export = SADLNet(
                False,
                input_channels,
                input_kernels,
                D1,
                D2,
                D3,
                D4,
                D5,
                D6,
                N,
                C,
                C1,
                C21,
                C22,
                C31,
                output_channels,
            )

    def get_example_inputs(
        self, patch_size: Union[int, Tuple[int, int]] = 144, batch_size: int = 1
    ):
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        global model_for_export
        return [
            torch.rand(
                batch_size, conv.in_channels, *patch_size, device=conv[0].weight.device
            )
            for conv in model_for_export[0].branches
        ]

    def to_onnx(
        self,
        filename: str,
        patch_size: int = 144,
        batch_size: int = 1,
        opset: int = 10,
        **kwargs,
    ) -> None:
        mode = self.training
        self.eval()
        global model_for_export
        model_for_export.load_state_dict(self.state_dict())
        model_for_export.eval()
        torch.onnx.export(
            model_for_export,
            self.get_example_inputs(patch_size, batch_size),
            filename,
            input_names=["in0", "in1", "in2", "in3", "in4", "in5"],
            dynamic_axes={
                "in0": {2: "h", 3: "w"},
                "in1": {2: "h", 3: "w"},
                "in2": {2: "h", 3: "w"},
                "in3": {2: "h", 3: "w"},
                "in4": {2: "h", 3: "w"},
                "in5": {2: "h", 3: "w"},
            },
            opset_version=opset,
            **kwargs,
        )
        self.train(mode)


class Net(nn.Module):
    """Wrapper for SADL model that implements input pre- and post-processing for training."""

    def __init__(
        self,
        input_channels: Iterable[Iterable[str]] = [
            ["rec_before_dbf_Y", "rec_before_dbf_U", "rec_before_dbf_V"],
            ["pred_Y", "pred_U", "pred_V"],
            ["bs_Y", "bs_U", "bs_V"],
            ["qp_base"],
            ["qp_slice"],
            ["ipb_Y"],
        ],
        input_kernels: Iterable[int] = [3, 3, 1, 1, 1, 1],
        D1: int = 192,
        D2: int = 32,
        D3: int = 16,
        D4: int = 16,
        D5: int = 16,
        D6: int = 48,
        N: int = 25,
        C: int = 64,
        C1: int = 192,
        C21: int = 32,
        C22: int = 64,
        C31: int = 48,
        path: str = None,
    ):
        super(Net, self).__init__()
        assert len(input_channels) == len(
            input_kernels
        ), "[ERROR] input size and kernels size not equal"
        self.input_channels = input_channels
        sizes = [len(a) for a in input_channels]
        self.SADL_model = SADLNet(
            True,
            sizes,
            input_kernels,
            D1,
            D2,
            D3,
            D4,
            D5,
            D6,
            N,
            C,
            C1,
            C21,
            C22,
            C31,
            6,
        )
        self.chroma_upsampler = nn.Upsample(scale_factor=2, mode="nearest")

    def preprocess_args(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return [
            torch.cat([batch[name] for name in input_], dim=1)
            for input_ in self.input_channels
        ]

    def postprocess_outputs(
        self, batch: Dict[str, torch.Tensor], out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Y_res, UV_res = out.split([4, 2], dim=1)
        return (
            batch["rec_before_dbf_Y"] + F.pixel_shuffle(Y_res, 2),
            torch.cat((batch["rec_before_dbf_U"], batch["rec_before_dbf_V"]), dim=1)[
                ..., ::2, ::2
            ]
            + UV_res,
        )

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        args = self.preprocess_args(batch)
        out = self.SADL_model(args)
        return self.postprocess_outputs(batch, out)

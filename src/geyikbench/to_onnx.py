"""
/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2022, ITU/ISO/IEC
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

import json
import argparse
import torch
import sys
import os
import importlib.util

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
print(sys.path)
from trainer import Trainer  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", "-c", type=str, help="config file corresponding to checkpoint"
)
parser.add_argument("--ckpt", type=str, help="path to training checkpoint")
parser.add_argument("--output", type=str, help="path to export onnx")
parser.add_argument(
    "--batch_size", type=int, default=1, help="batch size used in export's forward call"
)
parser.add_argument(
    "--patch_size",
    type=int,
    default=144,
    help="patch size used in export's forward call",
)
parser.add_argument(
    "--stage",
    type=int,
    choices=range(1, 4),
    help="if specified use a model from a particular stage [1-3]",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

# Model
if args.stage:
    if "model" in config[f"stage{args.stage}"]:
        json_model = config[f"stage{args.stage}"]["model"]
    else:
        quit("[WARNING] stage specified but no model available for this stage")
else:
    json_model = config["model"]

spec_model = importlib.util.spec_from_file_location("model", json_model["path"])
model = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(model)
modelobj = Trainer.instantiate_from_dict(model, json_model)

checkpoint = torch.load(args.ckpt, map_location=torch.device("cpu"))
modelobj.load_state_dict(checkpoint["model"])
modelobj.SADL_model.to_onnx(args.output, args.patch_size, args.batch_size)

#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle as pkl
import sys

import torch


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    print(obj.keys())
    obj = obj["teacher"]
    print(obj.keys())

    newmodel = {}
    print(len(obj.items()))
    skipped = 0
    for k, v in obj.items():
        if not k.startswith("module."):
            skipped += 1
            continue
        old_k = k
        k = k.replace("module.", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        k = k.replace("backbone.res", "backbone.bottom_up.res")
        k = k.replace("stem.backbone", "backbone.bottom_up.stem")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    print(skipped)

    res = {"model": newmodel, "__author__": "MOCO", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)

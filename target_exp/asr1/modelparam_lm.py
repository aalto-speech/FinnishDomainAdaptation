#!/usr/bin/env python

import sys
import numpy as np
import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.asr.asr_utils import torch_load

def main(args):
    """Run the main function"""

    model_loc = "exp/" + args[0] + "/rnnlm.model.best"
    print(model_loc)
    train_args = get_model_conf(model_loc)

    #model_module = "espnet.nets.pytorch_backend.lm.default:DefaultRNNLM"
    model_class = dynamic_import_lm(train_args.model_module,train_args.backend)
    if train_args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, train_args.train_dtype)
    else:
        dtype = torch.float32

    model = model_class(train_args.n_vocab,train_args).to(dtype=dtype)
    torch_load(model_loc, model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters: ", params)
    print("Length vocabulary: ", train_args.n_vocab)


if __name__ == '__main__':
    main(sys.argv[1:])
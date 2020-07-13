#!/usr/bin/env python

import os
import sys
import numpy as np
import torch

from espnet.asr.pytorch_backend.asr_init import load_trained_model

def main(args):
    """Run the main function"""

    model_name = args[0] +"_"+ args[3] +"_"+ os.path.splitext(os.path.basename(args[2]))[0] +"_"+ os.path.splitext(os.path.basename(args[1]))[0]
    model_loc = "exp/" + model_name + "/results/" + args[4]
    print(model_loc)
    model, _ = load_trained_model(model_loc)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("Trainable parameters: ", params)

if __name__ == '__main__':
    main(sys.argv[1:])
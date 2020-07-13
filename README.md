# FinnishDomainAdaptation
Domain Adaptation for Finnish data

This repository contains the configurations and scripts for training E2E models on Parliament and Yle data, using ESPnet toolkit.

## Requirements
You can either install ESPnet on triton or use the triton module 'espnet-2020'. Load all the required modules to your path.sh file.

## Directory Structure
It is an extension of the ESPnet repository. Refer to ESPnet directory structure here: https://espnet.github.io/espnet/tutorial.html#directory-structure. The directory should contain Kaldi style data, steps, utils, conf and local. Clone either WSJ or Tedlim2 recipe for this and then proceed with your experiments. In addition to the existing structure, create a folder for your configuration experiments. 

## Running scripts
'./run.sh' is the main script for running your experiments. You can modify the parameters directly within the script or pass them as command line arguments. 

## Performing Transfer Learning
The training configurations files need to be modified. Refer to the documentation - https://github.com/b-flo/espnet/blob/e972141b2110f4054f5046d73602089b39589a5c/doc/tutorial.md#how-to-use-finetuning

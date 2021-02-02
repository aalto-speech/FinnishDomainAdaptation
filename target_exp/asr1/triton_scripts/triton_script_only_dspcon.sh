#!/bin/bash
#SBATCH --job-name=train_lm_decoder_rnnlm
#SBATCH -N 1                 
#SBATCH --time=4-23:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -c 1
#SBATCH --mail-user=swati.choudhary@aalto.fi
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:2
srun ./run.sh --stage 4 --stop-stage 4 --backend pytorch --ngpu 2 --train_config studentconf/train_decoder_lm.yaml > wer_logs/train_lm_decoder_kieli_new.log
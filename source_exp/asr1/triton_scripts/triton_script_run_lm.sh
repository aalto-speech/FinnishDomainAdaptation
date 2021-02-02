#!/bin/bash
#SBATCH --job-name=test_train_transformer_lm_kieli       
#SBATCH --time=4-23:00:00
##SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mail-user=swati.choudhary@aalto.fi
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:2
srun ./run.sh --backend pytorch --stage 3 --stop-stage 3 --ngpu 2 > wer_logs/transformer_lm.log
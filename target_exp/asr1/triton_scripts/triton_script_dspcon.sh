#!/bin/bash
#SBATCH --job-name=kieli_10M_decode
#SBATCH -N 1                 
#SBATCH --time=4-23:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -c 1
#SBATCH --mail-user=swati.choudhary@aalto.fi
#SBATCH --mail-type=ALL
srun ./run_wo_adaptation.sh --stage 5 --stop-stage 5 --backend pytorch >> wer_logs/lm_decoder_kieli_10M.log
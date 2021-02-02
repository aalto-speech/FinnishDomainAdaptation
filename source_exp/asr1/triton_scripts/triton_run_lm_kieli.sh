#!/bin/bash
#SBATCH --job-name=decode_10M_lm_kieli       
#SBATCH --time=4-23:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mail-user=swati.choudhary@aalto.fi
#SBATCH --mail-type=ALL
##SBATCH --gres=gpu:v100:2
srun ./run_decode_large.sh --backend pytorch --stage 5 --stop-stage 5 >> wer_logs/lm_kieli_10M.log


# #!/bin/bash
# #SBATCH --job-name=clean_data         
# #SBATCH --time=2-23:00
# #SBATCH --mem=5G
# #SBATCH -N 1
# #SBATCH -c 1
# #SBATCH --mail-user=swati.choudhary@aalto.fi
# #SBATCH --mail-type=ALL
# #SBATCH --gres=gpu:v100:2
# srun ./run.sh --stage 4 --ngpu 2 --backend pytorch --train_config aaltoconf/train_transformer_32.yaml > wer_logs/training_model_transformer_encoder_32_2048units.log

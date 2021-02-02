#!/bin/bash
#SBATCH --time 2-12:0:0 --mem-per-cpu 16G --cpus-per-task 2
srun ./run_decode_large.sh --stage 5 --backend pytorch --decode_config /scratch/work/choudhs1/espnet/egs/swati_finnish_yle/asr1/aaltoconf/decode_beam_40_lm5_ctc3.yaml >> wer_logs/large_lm_2048.log


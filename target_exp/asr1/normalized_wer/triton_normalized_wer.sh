#!/bin/bash
#SBATCH --time 00:07:00 --mem 200
#sed "s/([^)]*-/(/g" -i dspcon_speed_perturbed_fulltransfer_charlm_dev.trn
#sed -e 's/^[ \t]*//' -i dspcon_speed_perturbed_fulltransfer_charlm_test.trn
#srun slf-results.sh hypothesis_data/dspcon_speed_perturbed_noadaptation_charlm_dev.trn data/dev >> wer_logs/wer_normalized_speed_perturbed_noadaptation_charlm_dev.log
srun slf-results.sh hypothesis_data/dspcon_no_adapt_test.trn data/test >> wer_logs/wer_normalized_noadaptation_charlm_test.log
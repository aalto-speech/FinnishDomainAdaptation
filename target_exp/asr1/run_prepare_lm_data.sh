#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# rnnlm related
use_wordlm=false     # false means to train/use a character LM
lm_resume=          # specify a snapshot file to resume LM training
lmtag="webcon_dspcon"              # tag for managing LMs

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dict=data/lang_1char/${lmtag}.txt
nlsyms=data/lang_1char/non_lang_syms_${lmtag}.txt

mkdir -p data/lang_1char/

## First remove end of sentence token from kielipankki data
#sed 's| <.*>||g' data/kielipankki/kielipankki.train > data/kielipankki/processed_data/${lmtag}

##Then merge webcon with dspcon transcripts
cat data/conv-fin-sanasto_train_org/text data/web.txt > data/${lmtag}

echo "make a non-linguistic symbol list"
cut -f 2- data/${lmtag} | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
cat ${nlsyms}

cut -f 2- data/conv-fin-sanasto_train/text > data/train.txt

echo "make a dictionary"
echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
text2token.py -s 1 -n 1 -l ${nlsyms} data/parl_kielifull | cut -f 2- -d" " | tr " " "\n" \
| sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
wc -l ${dict}

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi

lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

lmdatadir=data/local/lm_train
lmdict=${dict}
mkdir -p ${lmdatadir}
text2token.py -s 1 -n 1 -l ${nlsyms} data/parl_kielifull \
    | cut -f 2- -d" " > ${lmdatadir}/parl_kielifull.txt

echo "Done"
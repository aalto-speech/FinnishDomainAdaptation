#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/train_large_24l.yaml
lm_config=conf/lm_2048.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=false     # false means to train/use a character LM
lm_resume=          # specify a snapshot file to resume LM training
lmtag="charlm"              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
average_log=
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.

# feature configuration
do_delta=false


. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#ivec_extractor_dir=exp/ivec
train_set=all_cleaned_ivec
train_dev=yle-dev-new-ivec
train_test=yle-test-new-ivec
ivec_extractor_dir=exp/pretrained_vox_models

recog_set="yle-dev-new-ivec yle-test-new-ivec"

mkdir -p ${train_set}
mkdir -p ${train_dev}
mkdir -p ${train_test}

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    echo "NOTE: You should have the basic data directories from the normal run.sh already"

    utils/copy_data_dir.sh data/all_cleaned data/${train_set}
    utils/copy_data_dir.sh data/yle-dev-new data/${train_dev}
    utils/copy_data_dir.sh data/yle-test-new data/${train_test}

    cp data/all_cleaned/cmvn.ark data/all_cleaned_ivec/
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ]; then
    #train i-vector extractor: 
    echo "Stage 1: Traing i vec extractor"
    steps/nnet/ivector/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 data/${train_set} 512 ${ivec_extractor_dir}/diag_ubm_train
    steps/nnet/ivector/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 --ivector_dim 100 data/${train_set} ${ivec_extractor_dir}/diag_ubm_train ${ivec_extractor_dir}
fi

if [ ${stage} -le 2 ]; then
    local/extract_ivectors.sh --cmd "$train_cmd" --nj 30 \
      data/${train_set} ${ivec_extractor_dir} data/${train_set}/ivecs_pretrained
    for rtask in ${recog_set}; do
      local/extract_ivectors.sh --cmd "$train_cmd" --nj 10 \
        data/${rtask} ${ivec_extractor_dir} data/${rtask}/ivecs_pretrained
    done
    echo "vectors extracted"
fi

if [ $stage -le 3 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  echo "Compute mean"
  $train_cmd data/all_cleaned_ivec/ivecs/log/compute_mean.log \
    ivector-mean scp:data/all_cleaned_ivec/ivecs/ivectors_utt.scp \
    data/all_cleaned_ivec/ivecs/mean.vec || exit 1;
    echo "Finished mean vec computation"

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=100
  echo "Compute LDA"
  $train_cmd data/all_cleaned_ivec/ivecs/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:data/all_cleaned_ivec/ivecs/ivectors_utt.scp ark:- |" \
    ark:data/all_cleaned_ivec/utt2spk data/all_cleaned_ivec/ivecs/transform.mat || exit 1;
echo "Finished LDA computation"
fi

if [ ${stage} -le 4 ]; then

    # dump features for training
    echo "Dumping started for train set"
    local/dump_with_ivec.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/ivecs/mean.vec data/${train_set}/ivecs/transform.mat data/${train_set}/ivecs exp/dump_feats/train ${feat_tr_dir}
    echo "Dumping started for recog sets"
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/dump_with_ivec.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/ivecs/mean.vec data/${train_set}/ivecs/transform.mat data/${rtask}/ivecs exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
    echo "Dumping done for all"
fi


dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
          --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes about one day.  uses GPU (preferably use 2 gpu for faster training). If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5).

if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi

lmexpname=train_rnnlm_${backend}_2048
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 3: LM Preparation"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${train_test}/text > ${lmdatadir}/test.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
    fi

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
        echo "Training done"
fi



if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Decoding with character LM 2048 unit + decode beam size 40 + lm 0.5 + ctc 0.3"
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### experimenting with GPU, usually use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

    	score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi



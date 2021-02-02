#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0        # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/no_preprocess.yaml
train_config=conf/train_rnn.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ivec_extractor_dir=exp/ivec
train_set=train_trim_ivec
dev_set=dev_trim_ivec
recog_set="dev_ivec test_ivec"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    echo "NOTE: You should have the basic data directories from the normal run.sh already"
    utils/copy_data_dir.sh data/train_trim data/${train_set}
    utils/copy_data_dir.sh data/dev_trim data/${dev_set}
    utils/copy_data_dir.sh data/dev data/dev_ivec
    utils/copy_data_dir.sh data/test data/test_ivec
    cp data/train_trim/cmvn.ark data/train_trim_ivec/
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}/delta${do_delta}; mkdir -p ${feat_dt_dir}


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    #train i-vector extractor: 
    echo "Own ivector training"
    steps/nnet/ivector/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 data/${train_set} 512 ${ivec_extractor_dir}/diag_ubm_train
    steps/nnet/ivector/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 --ivector_dim 100 data/${train_set} ${ivec_extractor_dir}/diag_ubm_train ${ivec_extractor_dir}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2  ]; then
    echo "I vector extraction"
    local/extract_ivectors.sh --cmd "$train_cmd" --nj 30 \
      data/${train_set} ${ivec_extractor_dir} data/${train_set}/ivecs
    local/extract_ivectors.sh --cmd "$train_cmd" --nj 30 \
      data/${dev_set} ${ivec_extractor_dir} data/${dev_set}/ivecs
    for rtask in ${recog_set}; do
      local/extract_ivectors.sh --cmd "$train_cmd" --nj 30 \
        data/${rtask} ${ivec_extractor_dir} data/${rtask}/ivecs
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3  ]; then

    echo "Compute mean"
  $train_cmd data/${train_set}/ivecs/log/compute_mean.log \
    ivector-mean scp:data/${train_set}/ivecs/ivectors_utt.scp \
    data/${train_set}/ivecs/mean.vec || exit 1;
    echo "Finished mean vec computation"
fi

 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4  ]; then   
    echo " dump features for training" 
  local/dump_with_ivec_nolda.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/ivecs/mean.vec data/${train_set}/ivecs exp/dump_feats/${train_set} ${feat_tr_dir}
    local/dump_with_ivec_nolda.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${dev_set}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/ivecs/mean.vec data/${dev_set}/ivecs exp/dump_feats/${dev_set} ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        local/dump_with_ivec_nolda.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark data/${train_set}/ivecs/mean.vec data/${rtask}/ivecs exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
    echo "Done"
fi

dict=data/lang_char/${train_set}_units.txt
#nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 6: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
     | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
     wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config ${train_config} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json   ## training happens on trimmed data
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 5: Decoding"
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
        echo "${rtask}"
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        # ngpu=0
        # ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        #     asr_recog.py \
        #     --config ${decode_config} \
        #     --ngpu ${ngpu} \
        #     --backend ${backend} \
        #     --debugmode ${debugmode} \
        #     --verbose ${verbose} \
        #     --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
        #     --result-label ${expdir}/${decode_dir}/data.JOB.json \
        #     --model ${expdir}/results/${recog_model}  #\
        #     #--rnnlm ${lmexpdir}/rnnlm.model.best

        # score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

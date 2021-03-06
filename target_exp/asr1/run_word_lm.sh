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

# config files
#these are the default config paths, you can overwrite them while running
preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=studentconf/train.yaml
lm_config=studentconf/lm_word.yaml
decode_config=studentconf/decode.yaml

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000
lm_resume=          # specify a snapshot file to resume LM training
lmtag="word_lm"              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
average_log=
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.

# bpemode(unigram or bpe), default is unigram
nbpe=500
bpemode=bpe

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=conv-fin-sanasto_train_org
train_dev=dev
train_test=test
recog_set="dev test"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design test, training and dev sets by yourself. Data should be in kaldi-style
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Kaldi Feature Generation"
    fbankdir=fbank
   
   # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
   #80 features-  conf/fbank.conf
    for x in $train_set; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    echo "Features for training done"
    for x in $train_dev $train_test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 4 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done
    echo "Features for dev and test done"

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    
    # dump features for training
    echo "Dumping features started for train"
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    # dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
    #     data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    echo "Dumping features started for dev and test"    
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
    echo "feature dumps done"
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

# echo "dictionary: ${dict}"
# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     ### Task dependent. You have to check non-linguistic symbols used in the corpus.
#     echo "stage 2: Dictionary and Json Data Preparation"
#     mkdir -p data/lang_1char/

#     cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "\[" > ${nlsyms}
#     cat ${nlsyms}

#     echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#     text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
#     | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
#     wc -l ${dict}

#     data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
#          data/${train_set} ${dict} > ${feat_tr_dir}/data.json
#     # data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
#     #      data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
#     for rtask in ${recog_set}; do
#         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#         data2json.sh --feat ${feat_recog_dir}/feats.scp \
#             --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
#     done
#     echo "JSON files done"
# fi

# It takes about one day.  uses GPU (preferably use 2 gpu for faster training). If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5).
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi

lmexpname=train_rnnlm_${backend}_wordlm
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: LM Preparation"

#     if [ ${use_wordlm} = true ]; then
#         lmdatadir=data/local/wordlm_train
#         lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
#         mkdir -p ${lmdatadir}
#         cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train.txt
#         cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
#         cut -f 2- -d" " data/${train_test}/text > ${lmdatadir}/test.txt
#         text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
#     else
#         lmdatadir=data/local/lm_train
#         lmdict=${dict}
#         mkdir -p ${lmdatadir}
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
#             | cut -f 2- -d" " > ${lmdatadir}/train.txt
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
#             | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
#                 | cut -f 2- -d" " > ${lmdatadir}/test.txt
#     fi

#     echo "LM training begins"
#     ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#         lm_train.py \
#         --config ${lm_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --verbose 1 \
#         --outdir ${lmexpdir} \
#         --tensorboard-dir tensorboard/${lmexpname} \
#         --train-label ${lmdatadir}/train.txt \
#         --valid-label ${lmdatadir}/valid.txt \
#         --test-label ${lmdatadir}/test.txt \
#         --resume ${lm_resume} \
#         --dict ${lmdict}
# fi

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

# # Stage 4 is the Neural Network model training, uses GPU (preferably use 2 gpu for faster training)
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     echo "stage 4: Network Training"

#     ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
#         asr_train.py \
#         --config ${train_config} \
#         --preprocess-conf ${preprocess_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --outdir ${expdir}/results \
#         --tensorboard-dir tensorboard/${expname} \
#         --debugmode ${debugmode} \
#         --dict ${dict} \
#         --debugdir ${expdir} \
#         --minibatches ${N} \
#         --verbose ${verbose} \
#         --resume ${resume} \
#         --seed ${seed} \
#         --train-json ${feat_tr_dir}/data.json \
#         --valid-json ${feat_dt_dir}/data.json
#         echo "Training done"
# fi

# #Stage 5 is decoding - uses CPU 
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding using a word based, lm weight 0.5 ctc weight 0.3, TL for full model "
    nj=10
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        # average_checkpoints.py --backend ${backend} \
        #                         --snapshots ${expdir}/results/snapshot.ep.* \
        #                         --out ${expdir}/results/${recog_model} \
        #                         --num ${n_average}
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

        #### use CPU for decoding
        ngpu=0
        echo "Decode started"
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

# if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
#     echo "stage 8: Model parameters"
#     python modelparam.py \
# 	    ${train_set} ${preprocess_config} ${train_config} ${backend} ${recog_model} 
    
# fi

if [ "$stage" -le 9 ] && [ "$stop_stage" -ge 9 ]; then
    echo "stage 9: Language model parameters"
    python modelparam_lm.py \
	    ${lmexpname} 
fi

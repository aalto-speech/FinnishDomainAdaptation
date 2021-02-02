#from ../../../tools/env.sh:
module load cuda/10.0.130
module load cudnn/7.4.2-cuda
module load gcc/6.5.0
export GCC_VERSION=6.5.0
#module load cmake
NCCL_ROOT=/scratch/work/choudhs1/miniconda/envs/espnet_torch_11_cuda_10
export CPATH=$NCCL_ROOT/include:$CPATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH
export CFLAGS="-I$CUDA_ROOT/include $CFLAGS"
export CUDA_HOME=$CUDA_ROOT
export CUDA_PATH=$CUDA_ROOT

#additional modules:
# module load sph2pipe
# module load irstlm
module load sctk
module load sox

export IRSTLM=/scratch/elec/puhe/Modules/opt/kaldi-vanilla/kaldi-7637de7-6.4.0-2.28/tools/irstlm
export PATH=${PATH}:${IRSTLM}/bin
export SPH2PIPE=/scratch/elec/puhe/Modules/opt/kaldi-vanilla/kaldi-7637de7-6.4.0-2.28/tools/sph2pipe_v2.5
export PATH=${PATH}:${SPH2PIPE}

MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $MAIN_ROOT/tools/venv/bin/activate
fi
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

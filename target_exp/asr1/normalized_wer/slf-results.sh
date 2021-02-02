#!/bin/bash -e


if [ $# -ne 2 ]; then
   echo "usage: slf-results.sh [options] <hypotheses> <test-set>"
   echo "e.g.:  slf-results.sh lats-lms=10.trn data/devel"
   echo "main options (for others, see top of script file)"

   exit 0;
fi

module load sctk

hypotheses=$1
echo $hypotheses
test_set=$2

export TMPDIR=tmp

echo $hypotheses
echo $test_set

results () {
    sh score_sclite.sh ${test_set} ${hypotheses} 
}

results

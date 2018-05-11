#!/bin/bash
# set -x

ulimit -c unlimited

if [ $# -lt 4 ]; then
    echo "usage: $0 role num_servers num_workers bin [args..]"
    exit -1;
fi

# algorithm setting
export RANDOM_SEED=10
export DATA_DIR=./a9a-data
export NUM_FEATURE_DIM=123
export LEARNING_RATE=0.5
export TEST_INTERVAL=10
export SYNC_MODE=0
export NUM_ITERATION=200
export BATCH_SIZE=-1 # -1 means take all examples in each iteration

# worker/server/scheduler settings
export DMLC_ROLE=$1
shift
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

export DMLC_PS_ROOT_URI='128.104.222.74'
export DMLC_PS_ROOT_PORT=8001
if [[ $DMLC_ROLE = 'server' ]]
then
    export HEAPPROFILE=./S${i}
fi
if [[ $DMLC_ROLE = 'worker' ]]
then
    export HEAPPROFILE=./W${i}
fi

${bin} ${arg} &
wait


#!/bin/bash
# set -x

ulimit -c unlimited

if [ $# -lt 4 ]; then
    echo "usage: $0 role udf num_servers num_workers bin [args..]"
    exit -1;
fi

# algorithm setting
export RANDOM_SEED=10
export DATA_DIR=./digit
export NUM_FEATURE_DIM=784
export LEARNING_RATE=0.25
export TEST_INTERVAL=10
export SYNC_MODE=0
export NUM_ITERATION=100
export BATCH_SIZE=100 # -1 means take all examples in each iteration

# worker/server/scheduler settings
export DMLC_ROLE=$1
shift
export UDF=$1
shift
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

#export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_INTERFACE='br-flat-lan-1'
export DMLC_PS_ROOT_URI='10.11.10.1'
export DMLC_PS_ROOT_PORT=8001
if [[ $DMLC_ROLE = 'scheduler' ]]
then
    export PS_VERBOSE=1
fi
if [[ $DMLC_ROLE = 'server' ]]
then
    export HEAPPROFILE=./S${i}
fi
if [[ $DMLC_ROLE = 'worker' ]]
then
    export HEAPPROFILE=./W${i}
fi

sudo iftop -i br-flat-lan-1 -t > log &
${bin} ${arg}
ps aux | grep "iftop" |grep -v grep| cut -c 9-15 | xargs sudo kill -9
wait


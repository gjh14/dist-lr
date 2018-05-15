#!/bin/bash
# set -x

ulimit -c unlimited

if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

# algorithm setting
export RANDOM_SEED=10
export DATA_DIR=./digit
export NUM_FEATURE_DIM=784
export LEARNING_RATE=0.005
export TEST_INTERVAL=10
export SYNC_MODE=0
export NUM_ITERATION=200
export BATCH_SIZE=50 # -1 means take all examples in each iteration
export UDF=0.05

# worker/server/scheduler settings
export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8001
export DMLC_ROLE='scheduler'
${bin} ${arg} &


# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    ${bin} ${arg} &
done

# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    export HEAPPROFILE=./W${i}
    ${bin} ${arg} &
done

wait

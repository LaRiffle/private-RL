#!/bin/bash

export EXPPREFIX=$1
export AGENTID=$2 # reinforce, random, ac
export NUMRUNS=$3
export NUMEPS=$4
export ENVID='SecretBreakout-v0'

for i in `seq 1 ${NUMRUNS}`;
do
    python run.py \
        --seed ${i} \
        --env_id ${ENVID} \
        --agent_id ${AGENTID} \
        --exp_prefix ${EXPPREFIX} \
        --max_episodes ${NUMEPS} &
done
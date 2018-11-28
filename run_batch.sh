export EXPPREFIX='flight1'
export ENVID='SecretBreakout-v0'
export AGENTID='reinforce'

python run.py --seed 1 --env_id ${ENVID} --agent_id ${AGENTID} --exp_prefix ${EXPPREFIX} &
python run.py --seed 2 --env_id ${ENVID} --agent_id ${AGENTID} --exp_prefix ${EXPPREFIX} &
python run.py --seed 3 --env_id ${ENVID} --agent_id ${AGENTID} --exp_prefix ${EXPPREFIX} &
python run.py --seed 4 --env_id ${ENVID} --agent_id ${AGENTID} --exp_prefix ${EXPPREFIX}
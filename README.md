# Secret Breakout

This is a simplified version of breakout made specifically for secret agents learning in secret environments.

## Install

```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Play Breakout

Spacebar to start, left and right to play.
```sh
# example with visual interface
python atari_breakout.py
```

## Learn Breakout
```sh
python reinforce_breakout.py
```

## Random Baseline
```sh
python reinforce_breakout.py --random_action
```
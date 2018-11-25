[![CircleCI](https://circleci.com/gh/korymath/secret-breakout.svg?style=svg&circle-token=401570b69e540225deb1f315e4b83f04924d3582)](https://circleci.com/gh/korymath/secret-breakout)

# Secret Breakout

This is a simplified version of breakout made specifically for secret agents learning in secret environments.

## Install

Requires Python 3.6.

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

## Credits

This is based on the [simple Breakout game by Arthur198](https://gist.github.com/Arthur198/4a6ac71b8d646fb2fad6be347997ca77#file-atari_breakout-py).

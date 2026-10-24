# Observation and Actions

## Model contract

```
input   observation  (batch, 16, 8, 8)  float32
output  policy       (batch, 4096)      float32, raw logits
output  value        (batch,)           float32, in [-1, 1]
```

The batch axis is dynamic. `value` is from the perspective of the team to
move: `+1` wins, `-1` loses. Colony: One has no draws.

## Canonical view

The observation is always oriented to the team to move. For Magenta, board
coordinates are rotated 180 degrees. Planes 0-5 are always the moving team's
pieces and planes 6-11 are always the opponent's pieces. Plane 15 distinguishes
the actual side to move because Magenta wins rule tiebreaks.

## Planes

| Plane | Contents |
|---|---|
| 0-5 | own Ping, Daemon, Exploit, Trace, Ice, Kernel |
| 6-11 | enemy Ping, Daemon, Exploit, Trace, Ice, Kernel |
| 12 | firewall-locked squares |
| 13 | `actions_left / 2`: `1.0` then `0.5` within a turn |
| 14 | firewall urgency |
| 15 | `1.0` for Magenta to move, otherwise `0.0` |

Planes 0-12 are binary spatial planes. Planes 13-15 are scalars broadcast over
the board. Import constants instead of hardcoding dimensions:

```python
from colony_one import encoding

encoding.NUM_PLANES   # 16
encoding.NUM_ACTIONS  # 4096
```

## Actions

The policy index is:

```
index = from_square * 64 + to_square
square = row * 8 + col
```

Both squares use canonical coordinates. There is no move/shoot axis: legal
target geometry determines the action kind. Use the environment for legal
actions and masks; do not reimplement the rules.

```python
mask = env.legal_mask()       # bool, (4096,)
indices = env.legal_action_indices()
```

Only roughly 12-30 actions are usually legal. Mask illegal entries when
selecting actions and computing training targets:

```python
logits[~mask] = -np.inf
```

The referee reads and softmaxes only legal logits, but training without the
mask wastes capacity.

## Environment timing

`env.step(action)` applies one action. Each team takes two actions per turn;
the current team changes only after the second. Use `env.current_team`,
`env.actions_left`, and `env.round_number` after every step.

## Firewall urgency

Plane 14 is computed by `encoding.firewall_urgency(round_number)`. Do not
derive it separately. The firewall locks a ring after rounds 8, 16, 24, and 26.

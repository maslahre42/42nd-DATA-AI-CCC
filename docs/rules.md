# Rules

This is the tournament game specification. `colony_one/` is authoritative if
this document disagrees with the implementation.

## Board and timing

- Board: 8x8; coordinates are `(row, col)`, zero-indexed.
- Teams: Cyan and Magenta. Cyan moves first.
- The starting position comes from `colony_one/scene.json`. The Level 3 scene gives each team a
  Kernel, a Ping and an Exploit; the Level 4 scene gives each team one of all six pieces (Ping,
  Daemon, Exploit, Trace, Ice, Kernel).
- A turn is two actions by one team. A round is Cyan's turn then Magenta's.
- `env.step()` applies one action. Do not assume teams alternate after a step.
- No passing, no draws and no repetition rule. There is no separate move limit: the firewall's
  last lock after round 26 ends every game (see below).

## Pieces

| Piece | Movement | Capture |
|---|---|---|
| Ping | orthogonal, up to 2, empty squares only | diagonal, up to 2, moves onto target |
| Daemon | orthogonal slide, up to 3 | shoots adjacent enemy; distance 2-3 move-capture |
| Exploit | 2+1 L-jump; jumps over pieces and locked squares | moves onto target |
| Trace | one orthogonal square, empty only | shoots diagonally at range 2-3 |
| Ice | orthogonal or diagonal, up to 2 | moves onto target |
| Kernel | one square in any direction | moves onto target |

Sliding pieces stop at pieces and locked squares. A friendly piece always
blocks. The Exploit may jump over, but never land on, a locked square.

For Daemon movement, an adjacent enemy blocks; an adjacent enemy can only be
shot. For Trace shooting, an adjacent diagonal piece blocks the line and cannot
be shot; only the first piece at range 2-3 is a candidate. Ping and Trace never
move/capture in each other's directions.

## Kernel protection

An attack on a Kernel, by move or shot, sacrifices an adjacent friendly
non-Kernel bodyguard instead when one exists. The attacker remains at its origin
even if the action was a move-capture. Select the sacrificed guard in this
order:

```text
Ping, Exploit, Trace, Daemon, Ice
```

With no adjacent bodyguard, destroying a Kernel ends the game immediately.

## Firewall and endgame

At the end of rounds 8, 16, 24, and 26, lock the outermost open ring:

```text
8x8 -> 6x6 -> 4x4 -> 2x2 -> fully locked
```

Locked squares are permanently unavailable. Pieces on a newly locked square are
destroyed. If that destroys a Kernel, the other team wins; if it destroys both
Kernels at once, Magenta wins.

When only both Kernels remain, resolve at the end of the current turn:

1. If the two Kernels are within Chebyshev distance 2 of each other (any
   direction, whatever stands between them), the team that just finished its
   turn loses.
2. Otherwise, the Kernel on the smaller ring, closer to the board centre, wins.
3. Equal rings: Magenta wins.

Every game has a winner. Magenta also wins every tiebreak; this compensates for
Cyan moving first and is represented in observation plane 15.

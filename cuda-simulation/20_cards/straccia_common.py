"""
straccia_common.py

Funzioni di riferimento condivise (unranking + simulate), gia' validate
esaustivamente nei passi precedenti su mazzi ridotti da 8 e 10 carte.
Le riusiamo qui senza modifiche per generare i dati di test/riferimento
del mazzo da 20 carte.
"""

import itertools
from math import factorial
from collections import deque


def multinomial(counts):
    total = sum(counts)
    r = factorial(total)
    for v in counts:
        r //= factorial(v)
    return r


def build_table(counts):
    dims = [c + 1 for c in counts]
    size = dims[0] * dims[1] * dims[2] * dims[3]
    table = [0] * size
    for combo in itertools.product(*[range(d) for d in dims]):
        idx = ((combo[0] * dims[1] + combo[1]) * dims[2] + combo[2]) * dims[3] + combo[3]
        table[idx] = multinomial(list(combo))
    return table, dims


def tbl_lookup(table, dims, c0, c1, c2, c3):
    idx = ((c0 * dims[1] + c1) * dims[2] + c2) * dims[3] + c3
    return table[idx]


def unrank(table, dims, rank, counts):
    c = list(counts)
    n = sum(counts)
    out = []
    for _ in range(n):
        for sym in range(4):
            if c[sym] == 0:
                continue
            c[sym] -= 1
            perms = tbl_lookup(table, dims, c[0], c[1], c[2], c[3])
            if rank < perms:
                out.append(sym)
                break
            rank -= perms
            c[sym] += 1
    return out


def simulate(deal_a, deal_b, max_turns):
    """Regole confermate: divisione a blocchi, reinserimento FIFO,
    leader iniziale = giocatore della prima meta' del mazzo."""
    hands = [deque(deal_a), deque(deal_b)]
    pile = deque()
    leader = 0
    turn = 0

    while len(hands[0]) > 0 and len(hands[1]) > 0 and turn < max_turns:
        attacker, defender = leader, 1 - leader

        v = hands[attacker].popleft()
        pile.append(v)
        turn += 1

        if len(hands[defender]) == 0:
            break

        if v == 0:
            leader = defender
            continue

        pending = v
        while pending > 0:
            if len(hands[defender]) == 0:
                break
            rv = hands[defender].popleft()
            pile.append(rv)
            turn += 1
            pending -= 1
            if rv != 0:
                attacker, defender = defender, attacker
                pending = rv

        if len(hands[defender]) == 0:
            leader = attacker
            break

        while pile:
            hands[attacker].append(pile.popleft())
        leader = attacker

    return turn
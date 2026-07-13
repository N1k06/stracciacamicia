"""
confirm_cycle.py

A differenza di inspect_hits_40.py / validate_hits_20.py (che verificano solo
che GPU e Python siano D'ACCORDO dato lo stesso tetto di turni), questo
script verifica se un candidato e' un CICLO INFINITO VERO, togliendo il tetto
e cercando uno STATO RIPETUTO (README §9).

Idea: lo stato completo di una partita e' (mano A, mano B, mazzetto, chi ha
l'iniziativa). Il gioco e' deterministico: stesso stato -> stesso futuro. Se
lo stesso identico stato si ripresenta due volte, la partita e' GARANTITA
ciclare per sempre in modo identico da quel punto in poi -- non e' piu' un
sospetto, e' una dimostrazione.

Se invece la partita termina (un giocatore resta senza carte) prima che
qualunque stato si ripeta, allora NON era affatto un ciclo: era solo una
partita piu' lunga del tetto usato nello screening iniziale, e va scartata
come falso positivo.

Tre modalita' d'uso:

  1. Sequenza diretta (non serve la tabella multinomiale, la composizione si
     deduce dalla sequenza stessa):
       python3 confirm_cycle.py --sequence 00000030100100302020 [--max-turns N]

  2. Singolo rank (richiede --counts per ricostruire il mazzo):
       python3 confirm_cycle.py --counts 14,2,2,2 --rank 246414 [--max-turns N]

  3. Batch su un intero file di hit (richiede --counts):
       python3 confirm_cycle.py --counts 28,4,4,4 hits_40.bin [--max-turns N] [--no-header]

Formato hits_file (modalita' 3):
  - default (con header, come hits_20.bin/hits_20_bitpacked.bin):
        int32 max_turns_screening, uint32 n, poi n x (uint64 rank, uint32 turns)
  - con --no-header (come hits_40.bin, prodotto in append da straccia_search_40.cu):
        sequenza continua di (uint64 rank, uint32 turns), senza alcun header
"""

import argparse
import struct
import sys
from collections import deque
from straccia_common import build_table, unrank


def find_cycle(deal_a, deal_b, max_turns=2_000_000):
    """Rigioca la partita SENZA tetto artificiale (fino a max_turns come sola
    valvola di sicurezza per non girare all'infinito in caso di problemi),
    controllando ad ogni round se lo stato completo si e' gia' presentato.

    Ritorna una tupla:
      ('cycle', primo_turno_visto, turno_corrente, lunghezza_ciclo)  -- CICLO CONFERMATO
      ('terminated', turno)                                          -- non era un ciclo, finisce
      ('inconclusive', max_turns)                                    -- ne' l'uno ne' l'altro entro il limite
    """
    hands = [deque(deal_a), deque(deal_b)]
    pile = deque()
    leader = 0
    turn = 0
    seen = {}

    while hands[0] and hands[1] and turn < max_turns:
        # Stato completo checkpoint: mano A, mano B, mazzetto (in ordine), leader.
        # Se si ripete, il gioco e' garantito ciclare per sempre da qui in poi.
        state = (tuple(hands[0]), tuple(hands[1]), tuple(pile), leader)
        if state in seen:
            first_turn = seen[state]
            return ("cycle", first_turn, turn, turn - first_turn)
        seen[state] = turn

        attacker, defender = leader, 1 - leader
        v = hands[attacker].popleft()
        pile.append(v)
        turn += 1

        if not hands[defender]:
            return ("terminated", turn)

        if v == 0:
            leader = defender
            continue

        pending = v
        while pending > 0:
            if not hands[defender]:
                break
            rv = hands[defender].popleft()
            pile.append(rv)
            turn += 1
            pending -= 1
            if rv != 0:
                attacker, defender = defender, attacker
                pending = rv

        if not hands[defender]:
            return ("terminated", turn)

        while pile:
            hands[attacker].append(pile.popleft())
        leader = attacker

    if not hands[0] or not hands[1]:
        return ("terminated", turn)
    return ("inconclusive", turn)


def read_hits(path, with_header):
    with open(path, "rb") as fh:
        data = fh.read()

    hits = []
    if with_header:
        max_turns, n = struct.unpack_from("<iI", data, 0)
        offset = 8
        for _ in range(n):
            rank, turns = struct.unpack_from("<QI", data, offset)
            offset += 12
            hits.append((rank, turns))
    else:
        record_size = 12
        n = len(data) // record_size
        for i in range(n):
            rank, turns = struct.unpack_from("<QI", data, i * record_size)
            hits.append((rank, turns))
    return hits


def print_result(label, result, max_turns):
    if result[0] == "cycle":
        _, first_turn, cur_turn, cycle_len = result
        print(f"{label} -> CICLO CONFERMATO: stato ripetuto dopo {cycle_len} turni "
              f"(primo visto al turno {first_turn}, rivisto al turno {cur_turn})")
    elif result[0] == "terminated":
        _, real_turns = result
        print(f"{label} -> NON è un ciclo: termina naturalmente al turno {real_turns}")
    else:
        print(f"{label} -> INCONCLUSIVO: nessuno stato ripetuto e nessuna fine entro "
              f"{max_turns} turni (aumenta --max-turns per un verdetto più certo)")


def run_single(deck, max_turns):
    half = len(deck) // 2
    counts_found = [deck.count(s) for s in range(4)]
    print(f"Sequenza: {''.join(map(str, deck))}")
    print(f"Carte totali: {len(deck)}  meta': {half}  composizione dedotta: {counts_found}")
    print(f"Limite di sicurezza: {max_turns} turni\n")

    result = find_cycle(deck[:half], deck[half:], max_turns)
    print_result("Risultato", result, max_turns)


def main():
    parser = argparse.ArgumentParser(
        description="Verifica se un candidato e' un ciclo infinito vero, o solo una partita lunga troncata.")
    parser.add_argument("hits_file", nargs="?",
                         help="file con una lista di hit da verificare in batch (richiede --counts)")
    parser.add_argument("--counts", type=str,
                         help='composizione del mazzo, es. "14,2,2,2" o "28,4,4,4" (richiesto per --rank o hits_file)')
    parser.add_argument("--sequence", type=str,
                         help="sequenza di carte da testare direttamente, es. 00000030100100302020 "
                              "(la composizione si deduce dalla sequenza, non serve --counts)")
    parser.add_argument("--rank", type=int,
                         help="singolo rank da testare (richiede --counts per ricostruire il mazzo)")
    parser.add_argument("--max-turns", type=int, default=2_000_000,
                         help="limite di sicurezza sui turni (default 2.000.000)")
    parser.add_argument("--no-header", action="store_true",
                         help="il file hits_file non ha header (formato hits_40.bin)")
    args = parser.parse_args()

    # --- Modalita' 1: sequenza diretta ---
    if args.sequence:
        deck = [int(c) for c in args.sequence.strip()]
        if any(s not in (0, 1, 2, 3) for s in deck):
            print("Errore: la sequenza deve contenere solo cifre 0,1,2,3")
            sys.exit(1)
        if len(deck) % 2 != 0:
            print("Errore: la sequenza deve avere un numero pari di carte (divisione a meta')")
            sys.exit(1)
        run_single(deck, args.max_turns)
        return

    # --- Modalita' 2: singolo rank ---
    if args.rank is not None:
        if not args.counts:
            print("Errore: --rank richiede anche --counts")
            sys.exit(1)
        counts = [int(x) for x in args.counts.split(",")]
        print("Costruzione tabella multinomiale...")
        table, dims = build_table(counts)
        deck = unrank(table, dims, args.rank, counts)
        print(f"Rank: {args.rank}")
        run_single(deck, args.max_turns)
        return

    # --- Modalita' 3: batch su file di hit ---
    if not args.hits_file:
        parser.print_help()
        sys.exit(1)
    if not args.counts:
        print("Errore: la modalita' batch su file richiede --counts")
        sys.exit(1)

    counts = [int(x) for x in args.counts.split(",")]
    hits = read_hits(args.hits_file, with_header=not args.no_header)
    half = sum(counts) // 2

    print(f"Composizione: {counts}  carte totali: {sum(counts)}  meta': {half}")
    print(f"Candidati da verificare: {len(hits)}  (limite di sicurezza: {args.max_turns} turni)\n")

    if not hits:
        print("Nessun candidato nel file.")
        return

    print("Costruzione tabella multinomiale...")
    table, dims = build_table(counts)

    n_cycles = 0
    n_terminated = 0
    n_inconclusive = 0

    for i, (rank, turns_screening) in enumerate(hits):
        deck = unrank(table, dims, rank, counts)
        result = find_cycle(deck[:half], deck[half:], args.max_turns)

        label = f"[{i+1}/{len(hits)}] rank={rank}  turni_screening={turns_screening} "
        print_result(label, result, args.max_turns)

        if result[0] == "cycle":
            n_cycles += 1
        elif result[0] == "terminated":
            n_terminated += 1
        else:
            n_inconclusive += 1

    print(f"\n=== RIEPILOGO ===")
    print(f"Cicli infiniti CONFERMATI:  {n_cycles}")
    print(f"Falsi positivi (finiscono): {n_terminated}")
    print(f"Inconclusivi:               {n_inconclusive}")


if __name__ == "__main__":
    main()
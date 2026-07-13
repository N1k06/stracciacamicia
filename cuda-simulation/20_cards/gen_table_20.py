"""
gen_table_20.py

Prepara i dati per una ricerca ESAUSTIVA completa su un mazzo ridotto a 20
carte, mantenendo la stessa proporzione del mazzo reale (7 parti "0" per ogni
parte di "1","2","3" -> 28:4:4:4 su 40 diventa 14:2:2:2 su 20).

Totale configurazioni distinte: 20! / (14! 2! 2! 2!) = 3.488.400 circa --
abbastanza piccolo da poter essere enumerato per intero sulla GPU in pochi
secondi/minuti, ma abbastanza grande da essere un test end-to-end realistico
della pipeline completa (unranking + simulate + kernel persistente) prima di
passare al mazzo reale da 40 carte.

Genera:
  - table_20.bin         tabella multinomiale con header (stesso formato dei
                          test precedenti)
  - sample_ranks_20.bin   campione di rank (casuali + estremi) per una
                          validazione spot-check indipendente
  - sample_turns_20.bin   turni di riferimento Python per lo stesso campione
                          (con header: max_turns, n_samples)
"""

import struct
import random
from straccia_common import build_table, tbl_lookup, unrank, simulate

COUNTS = [14, 2, 2, 2]
SAMPLE_SIZE = 5000
SAMPLE_MAX_TURNS = 20000  # generoso per il campione di validazione Python


def main():
    print(f"Composizione: {COUNTS} (14:2:2:2, stessa proporzione 7:1:1:1 del mazzo reale)")
    table, dims = build_table(COUNTS)
    total = tbl_lookup(table, dims, *COUNTS)
    print(f"Totale configurazioni distinte: {total}")

    deck_size = sum(COUNTS)
    half = deck_size // 2
    print(f"Carte totali: {deck_size}, meta': {half}")

    with open("table_20.bin", "wb") as fh:
        fh.write(struct.pack("<4i", *COUNTS))
        for v in table:
            fh.write(struct.pack("<Q", v))
    print(f"Scritto table_20.bin (header + {len(table)} x uint64)")

    # --- campione di validazione indipendente ---
    rng = random.Random(42)
    sample_ranks = {0, total - 1}
    while len(sample_ranks) < SAMPLE_SIZE:
        sample_ranks.add(rng.randrange(total))
    sample_ranks = sorted(sample_ranks)

    print(f"Calcolo riferimento Python per {len(sample_ranks)} rank campione (max_turns={SAMPLE_MAX_TURNS})...")
    sample_turns = []
    max_seen = 0
    hit_cap = 0
    for r in sample_ranks:
        deck = unrank(table, dims, r, COUNTS)
        turns = simulate(deck[:half], deck[half:], SAMPLE_MAX_TURNS)
        sample_turns.append(turns)
        if turns > max_seen:
            max_seen = turns
        if turns >= SAMPLE_MAX_TURNS:
            hit_cap += 1

    print(f"Turni massimi osservati nel campione: {max_seen}")
    print(f"Configurazioni nel campione che raggiungono il tetto ({SAMPLE_MAX_TURNS}): {hit_cap}")

    with open("sample_ranks_20.bin", "wb") as fh:
        for r in sample_ranks:
            fh.write(struct.pack("<Q", r))

    with open("sample_turns_20.bin", "wb") as fh:
        fh.write(struct.pack("<II", SAMPLE_MAX_TURNS, len(sample_ranks)))
        for t in sample_turns:
            fh.write(struct.pack("<I", t))

    print("Scritti sample_ranks_20.bin, sample_turns_20.bin")
    print("\nPronto per search_full_20.cu (ricerca esaustiva su GPU) e validate_sample_20.cu.")


if __name__ == "__main__":
    main()

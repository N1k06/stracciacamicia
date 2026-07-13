"""
gen_table_and_tests.py

Genera i file necessari per testare l'unranking su GPU:

  1. multinomial_table.bin    - tabella dei coefficienti multinomiali, da caricare
                                 in __constant__ memory nel kernel CUDA
  2. test_ranks.bin            - lista di rank di test (uint64 little-endian)
  3. reference_sequences.bin  - sequenze attese per ciascun rank di test, calcolate
                                 in Python (40 byte per sequenza, un byte per simbolo)
  4. reference_sequences.txt  - stessa cosa in formato leggibile per ispezione manuale

Prima di scrivere qualunque file, esegue un self-test in puro Python (unranking +
ranking di andata/ritorno) per garantire che la logica di riferimento sia corretta
PRIMA di usarla come ground truth per validare la GPU.
"""

import struct
import random
from math import factorial

# Composizione del mazzo reale di Straccia Camicia:
# 28 carte non vincenti (simbolo 0), 4 carte "vinci 1" (simbolo 1),
# 4 carte "vinci 2" (simbolo 2), 4 carte "vinci 3" (simbolo 3). Totale 40 carte.
COUNTS = [28, 4, 4, 4]
DIMS = [c + 1 for c in COUNTS]  # 29, 5, 5, 5 -> range possibile per ogni simbolo residuo


def multinomial(counts):
    """Coefficiente multinomiale n! / (c0! c1! c2! c3!) per un dato multiset residuo."""
    total = sum(counts)
    result = factorial(total)
    for v in counts:
        result //= factorial(v)
    return result


def build_table():
    """Costruisce la tabella linearizzata multinomial(c0,c1,c2,c3) per tutte le
    combinazioni possibili di simboli residui durante l'unranking."""
    table = [0] * (DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3])
    for c0 in range(DIMS[0]):
        for c1 in range(DIMS[1]):
            for c2 in range(DIMS[2]):
                for c3 in range(DIMS[3]):
                    idx = ((c0 * DIMS[1] + c1) * DIMS[2] + c2) * DIMS[3] + c3
                    table[idx] = multinomial([c0, c1, c2, c3])
    return table


def tbl_lookup(table, c0, c1, c2, c3):
    idx = ((c0 * DIMS[1] + c1) * DIMS[2] + c2) * DIMS[3] + c3
    return table[idx]


def unrank(table, rank, counts):
    """Unranking via lookup in tabella (stessa logica di unrank_multiset originale,
    ma senza ricalcolare i fattoriali ogni volta)."""
    c = list(counts)
    out = []
    for _ in range(sum(counts)):
        for sym in range(4):
            if c[sym] == 0:
                continue
            c[sym] -= 1
            perms = tbl_lookup(table, c[0], c[1], c[2], c[3])
            if rank < perms:
                out.append(sym)
                break
            rank -= perms
            c[sym] += 1
    return out


def rank_of(table, seq, counts):
    """Operazione inversa (ranking): dalla sequenza calcola il rank corrispondente.
    Usata per il test di consistenza round-trip: rank_of(unrank(r)) deve tornare r."""
    c = list(counts)
    rank = 0
    for sym_actual in seq:
        for sym in range(sym_actual):
            if c[sym] == 0:
                continue
            c[sym] -= 1
            rank += tbl_lookup(table, c[0], c[1], c[2], c[3])
            c[sym] += 1
        c[sym_actual] -= 1
    return rank


def self_test(table, counts, n_random=2000, seed=12345):
    """Round-trip test in puro Python: per un campione di rank (compresi gli estremi
    e valori casuali), verifica che unrank + rank_of tornino al rank di partenza."""
    total = tbl_lookup(table, *counts)
    rng = random.Random(seed)
    test_ranks = {0, total - 1}
    while len(test_ranks) < n_random + 2:
        test_ranks.add(rng.randrange(total))
    test_ranks = sorted(test_ranks)

    failures = []
    for r in test_ranks:
        seq = unrank(table, r, counts)
        r2 = rank_of(table, seq, counts)
        if r2 != r:
            failures.append((r, r2, seq))

    return total, test_ranks, failures


def main():
    print("Costruzione tabella multinomiale...")
    table = build_table()
    total = tbl_lookup(table, *COUNTS)
    print(f"Totale configurazioni distinte: {total}")

    print("Self-test round-trip in Python (unrank -> rank_of)...")
    total_check, test_ranks_py, failures = self_test(table, COUNTS, n_random=2000)
    assert total_check == total
    if failures:
        print(f"FALLITO: {len(failures)} discrepanze su {len(test_ranks_py)} rank testati.")
        for r, r2, seq in failures[:10]:
            print(f"  rank={r} -> unrank -> rank_of={r2} (seq={''.join(map(str, seq))})")
        raise SystemExit(1)
    print(f"OK: tutti i {len(test_ranks_py)} rank testati sono round-trip consistenti in Python.\n")

    # --- Scrittura tabella multinomiale (per la GPU, __constant__ memory) ---
    with open("multinomial_table.bin", "wb") as fh:
        for v in table:
            fh.write(struct.pack("<Q", v))
    print(f"Scritto multinomial_table.bin ({len(table)} x uint64 = {len(table) * 8} byte)")

    # --- Costruzione dei rank di test per la GPU ---
    # Includiamo: gli estremi, i batch_start del vero schema di scrematura
    # (step = 2*10^8, cosi' il test copre anche i punti che verranno effettivamente
    # usati nel run di produzione), e un campione di rank casuali.
    step = 200_000_000
    batch_starts = list(range(0, total, step))

    rng = random.Random(999)
    random_ranks = [rng.randrange(total) for _ in range(500)]

    test_ranks_list = sorted(set([0, total - 1] + batch_starts[:200] + random_ranks))
    print(f"Numero di rank di test totali: {len(test_ranks_list)}")

    with open("test_ranks.bin", "wb") as fh:
        for r in test_ranks_list:
            fh.write(struct.pack("<Q", r))

    # --- Sequenze di riferimento (calcolate in Python, "verita' di terra") ---
    with open("reference_sequences.bin", "wb") as fbin, open("reference_sequences.txt", "w") as ftxt:
        for r in test_ranks_list:
            seq = unrank(table, r, COUNTS)
            fbin.write(bytes(seq))  # 40 byte, uno per simbolo (valori 0-3)
            ftxt.write(f"{r}: {''.join(map(str, seq))}\n")

    print("Scritti test_ranks.bin, reference_sequences.bin, reference_sequences.txt")
    print("\nPronto per il test CUDA (test_unrank.cu).")


if __name__ == "__main__":
    main()
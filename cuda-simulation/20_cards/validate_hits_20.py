"""
validate_hits_20.py

Ricalcola in Python il numero di turni per OGNI candidato ("hit") trovato dalla
ricerca esaustiva su GPU (search_full_20.cu), e verifica che coincida
esattamente col valore riportato dalla GPU. Dato che ci aspettiamo pochi hit
(le configurazioni che raggiungono il tetto di turni), possiamo permetterci di
validarli TUTTI individualmente, invece di un campione.

Uso:
    python3 validate_hits_20.py table_20.bin hits_20.bin
"""

import struct
import sys
from straccia_common import build_table, unrank, simulate


def main():
    if len(sys.argv) != 3:
        print("Uso: python3 validate_hits_20.py table_20.bin hits_20.bin")
        sys.exit(1)

    table_path, hits_path = sys.argv[1], sys.argv[2]

    with open(table_path, "rb") as fh:
        counts = list(struct.unpack("<4i", fh.read(16)))
    table, dims = build_table(counts)  # ricostruita in Python (identica per costruzione)

    half = sum(counts) // 2

    with open(hits_path, "rb") as fh:
        max_turns, n = struct.unpack("<iI", fh.read(8))
        hits = []
        for _ in range(n):
            rank, turns_gpu = struct.unpack("<QI", fh.read(12))
            hits.append((rank, turns_gpu))

    print(f"Candidati da validare: {n}  (max_turns usato nella ricerca: {max_turns})")

    if n == 0:
        print("\nNessun candidato trovato dalla ricerca esaustiva: nulla da validare.")
        print("(Non significa necessariamente che non esistano loop: significa che nessuna")
        print(" configurazione di questo mazzo ridotto ha superato la soglia impostata.)")
        return

    mismatches = []
    max_turns_seen = 0
    for rank, turns_gpu in hits:
        deck = unrank(table, dims, rank, counts)
        turns_py = simulate(deck[:half], deck[half:], max_turns)
        max_turns_seen = max(max_turns_seen, turns_py)
        if turns_py != turns_gpu:
            mismatches.append((rank, turns_gpu, turns_py))

    print(f"Turni massimi confermati in Python tra i candidati: {max_turns_seen}")

    if mismatches:
        print(f"\nATTENZIONE: {len(mismatches)} discrepanze su {n} candidati:")
        for rank, tg, tp in mismatches[:15]:
            print(f"  rank={rank}  GPU={tg}  Python={tp}")
        sys.exit(1)
    else:
        print(f"\nTUTTI I {n} CANDIDATI CONFERMATI (GPU e Python coincidono esattamente).")


if __name__ == "__main__":
    main()

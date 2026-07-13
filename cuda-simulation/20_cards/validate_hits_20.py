"""
validate_hits_20.py

Ricalcola in Python il numero di turni per OGNI candidato ("hit") trovato dalla
ricerca esaustiva su GPU (search_full_20.cu), e verifica che coincida
esattamente col valore riportato dalla GPU. Dato che ci aspettiamo pochi hit
(le configurazioni che raggiungono il tetto di turni), possiamo permetterci di
validarli TUTTI individualmente, invece di un campione.

Oltre all'output a schermo, scrive un file di testo con il dettaglio di ogni
candidato (rank, sequenza di carte ricostruita, turni GPU vs Python, esito).

Uso:
    python3 validate_hits_20.py table_20.bin hits_20.bin [report.txt]

Se non specificato, il file di report si chiama "hits_validation_report.txt".
"""

import struct
import sys
from straccia_common import build_table, unrank, simulate


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Uso: python3 validate_hits_20.py table_20.bin hits_20.bin [report.txt]")
        sys.exit(1)

    table_path, hits_path = sys.argv[1], sys.argv[2]
    report_path = sys.argv[3] if len(sys.argv) == 4 else "hits_validation_report.txt"

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
        with open(report_path, "w") as fh:
            fh.write(f"Composizione: {counts}\n")
            fh.write(f"max_turns usato nella ricerca: {max_turns}\n")
            fh.write("Candidati trovati: 0\n")
            fh.write("Nessun candidato da riportare.\n")
        print(f"Scritto {report_path}")
        return

    rows = []  # (rank, sequenza, turns_gpu, turns_py, match)
    mismatches = []
    max_turns_seen = 0

    for rank, turns_gpu in hits:
        deck = unrank(table, dims, rank, counts)
        turns_py = simulate(deck[:half], deck[half:], max_turns)
        max_turns_seen = max(max_turns_seen, turns_py)
        match = (turns_py == turns_gpu)
        rows.append((rank, deck, turns_gpu, turns_py, match))
        if not match:
            mismatches.append((rank, turns_gpu, turns_py))

    print(f"Turni massimi confermati in Python tra i candidati: {max_turns_seen}")

    # --- scrittura del file di testo con il dettaglio di ogni candidato ---
    with open(report_path, "w") as fh:
        fh.write(f"Composizione: {counts}\n")
        fh.write(f"Carte totali: {sum(counts)}  meta': {half}\n")
        fh.write(f"max_turns usato nella ricerca: {max_turns}\n")
        fh.write(f"Candidati trovati: {len(rows)}\n")
        fh.write(f"Turni massimi confermati in Python: {max_turns_seen}\n")
        fh.write(f"Discrepanze GPU vs Python: {len(mismatches)}\n")
        fh.write("\n")
        fh.write(f"{'rank':>15}  {'turni_GPU':>10}  {'turni_Python':>12}  {'esito':>8}  sequenza\n")
        fh.write("-" * 100 + "\n")
        for rank, deck, turns_gpu, turns_py, match in rows:
            esito = "OK" if match else "MISMATCH"
            seq_str = "".join(map(str, deck))
            fh.write(f"{rank:>15}  {turns_gpu:>10}  {turns_py:>12}  {esito:>8}  {seq_str}\n")

    print(f"Scritto {report_path} con il dettaglio di tutti i {len(rows)} candidati")

    if mismatches:
        print(f"\nATTENZIONE: {len(mismatches)} discrepanze su {n} candidati:")
        for rank, tg, tp in mismatches[:15]:
            print(f"  rank={rank}  GPU={tg}  Python={tp}")
        sys.exit(1)
    else:
        print(f"\nTUTTI I {n} CANDIDATI CONFERMATI (GPU e Python coincidono esattamente).")


if __name__ == "__main__":
    main()
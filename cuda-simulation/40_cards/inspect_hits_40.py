"""
inspect_hits_40.py

Legge hits_40.bin (l'output cumulativo del programma di produzione
straccia_search_40.cu) e per ciascun candidato trovato:
  - ricostruisce la sequenza di 40 carte a partire dal rank
  - riverifica il conteggio turni in Python (stessa logica, stesso max_turns)
  - segnala eventuali discrepanze rispetto a quanto riportato dalla GPU

Puo' essere eseguito periodicamente durante un run lungo (il file hits_40.bin
cresce nel tempo, aggiunta dopo aggiunta) per tenere sotto controllo i
risultati senza aspettare la fine dell'intera ricerca.

Uso:
    python3 inspect_hits_40.py hits_40.bin [report.txt]

Nota: se il numero di candidati diventa molto grande (migliaia), la
riverifica completa in Python puo' richiedere piu' tempo; in tal caso valuta
di passare solo un campione (modifica MAX_TO_VERIFY qui sotto).
"""

import struct
import sys
from straccia_common import build_table, unrank, simulate

COUNTS = [28, 4, 4, 4]
MAX_TO_VERIFY = 5000  # se hits_40.bin contiene piu' candidati, verifica solo i primi N in Python


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Uso: python3 inspect_hits_40.py hits_40.bin [report.txt]")
        sys.exit(1)

    hits_path = sys.argv[1]
    report_path = sys.argv[2] if len(sys.argv) == 3 else "hits_40_report.txt"

    # hits_40.bin non ha un header (a differenza di hits_20.bin): e' semplicemente
    # una sequenza continua di record (uint64 rank, uint32 turns), scritti in
    # append da straccia_search_40.cu batch dopo batch.
    hits = []
    with open(hits_path, "rb") as fh:
        data = fh.read()
    record_size = 8 + 4
    n_total = len(data) // record_size
    if len(data) % record_size != 0:
        print(f"ATTENZIONE: {hits_path} ha una dimensione non multipla di {record_size} byte "
              f"({len(data)} byte, resto {len(data) % record_size}). Il file potrebbe essere "
              f"stato troncato da una scrittura interrotta a meta'.")

    for i in range(n_total):
        rank, turns_gpu = struct.unpack_from("<QI", data, i * record_size)
        hits.append((rank, turns_gpu))

    print(f"Candidati totali in {hits_path}: {n_total}")

    if n_total == 0:
        print("Nessun candidato trovato finora.")
        with open(report_path, "w") as fh:
            fh.write("Nessun candidato trovato finora.\n")
        return

    print("Costruzione tabella multinomiale (composizione 28/4/4/4)...")
    table, dims = build_table(COUNTS)
    half = sum(COUNTS) // 2

    to_verify = hits[:MAX_TO_VERIFY]
    if len(hits) > MAX_TO_VERIFY:
        print(f"Verifico solo i primi {MAX_TO_VERIFY} candidati su {len(hits)} totali "
              f"(modifica MAX_TO_VERIFY nello script per cambiare questo limite).")

    print(f"Riverifica in Python di {len(to_verify)} candidati (puo' richiedere tempo)...")

    rows = []
    mismatches = []
    max_turns_seen = 0

    # Il max_turns usato nella ricerca non e' salvato in hits_40.bin (a differenza
    # di hits_20.bin): se lo hai cambiato rispetto al default, passalo qui.
    MAX_TURNS_USED = 5000

    for idx, (rank, turns_gpu) in enumerate(to_verify):
        deck = unrank(table, dims, rank, COUNTS)
        turns_py = simulate(deck[:half], deck[half:], MAX_TURNS_USED)
        max_turns_seen = max(max_turns_seen, turns_py)
        match = (turns_py == turns_gpu)
        rows.append((rank, deck, turns_gpu, turns_py, match))
        if not match:
            mismatches.append((rank, turns_gpu, turns_py))
        if (idx + 1) % 500 == 0:
            print(f"  ...{idx + 1}/{len(to_verify)} verificati")

    print(f"\nTurni massimi confermati in Python: {max_turns_seen}")
    print(f"Discrepanze GPU vs Python: {len(mismatches)}")

    with open(report_path, "w") as fh:
        fh.write(f"Candidati totali nel file: {n_total}\n")
        fh.write(f"Candidati riverificati in Python: {len(to_verify)}\n")
        fh.write(f"Turni massimi confermati: {max_turns_seen}\n")
        fh.write(f"Discrepanze: {len(mismatches)}\n\n")
        fh.write(f"{'rank':>18}  {'turni_GPU':>10}  {'turni_Python':>12}  {'esito':>8}  sequenza\n")
        fh.write("-" * 110 + "\n")
        for rank, deck, turns_gpu, turns_py, match in rows:
            esito = "OK" if match else "MISMATCH"
            seq_str = "".join(map(str, deck))
            fh.write(f"{rank:>18}  {turns_gpu:>10}  {turns_py:>12}  {esito:>8}  {seq_str}\n")

    print(f"Scritto {report_path}")

    if mismatches:
        print(f"\nATTENZIONE: {len(mismatches)} discrepanze trovate, vedi {report_path} per il dettaglio.")
        sys.exit(1)
    else:
        print(f"\nTutti i {len(to_verify)} candidati riverificati coincidono con la GPU.")


if __name__ == "__main__":
    main()

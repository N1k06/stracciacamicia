"""
inspect_hits_40.py

Legge hits_40.bin e per ciascun candidato trovato:
  - ricostruisce la sequenza di 40 carte
  - simula la partita tenendo traccia degli stati di gioco
  - se lo stato si ripete, decreta matematicamente che la partita è INFINITA
  - genera un report delle partite effettivamente infinite.
"""

import struct
import sys
from straccia_common import build_table, unrank

COUNTS = [28, 4, 4, 4]
MAX_TO_VERIFY = 5000 

def check_infinite(p1_deck, p2_deck):
    """
    Simula la partita tenendo traccia di ogni stato.
    Ritorna: (is_infinite, turni_giocati)
    """
    p1 = list(p1_deck)
    p2 = list(p2_deck)
    pile = []
    turn = 1
    pending = 0
    seen = set()

    turns = 0

    # Invece di controllare che entrambi abbiano sempre carte,
    # andiamo avanti all'infinito e fermiamo solo se chi DEVE giocare è a secco.
    while True:
        # CONDIZIONE DI VITTORIA/SCONFITTA CORRETTA
        if turn == 1 and not p1:
            return False, turns
        if turn == 2 and not p2:
            return False, turns

        # Salvataggio dello stato per il rilevamento loop
        state = (tuple(p1), tuple(p2), tuple(pile), turn, pending)
        if state in seen:
            return True, turns
            
        seen.add(state)
        turns += 1

        # Pesca
        if turn == 1:
            c = p1.pop(0)
        else:
            c = p2.pop(0)

        pile.append(c)

        if c > 0:
            turn = 3 - turn
            pending = c
        else:
            if pending > 0:
                pending -= 1
                if pending == 0:
                    winner = 3 - turn
                    if winner == 1:
                        p1.extend(pile)
                    else:
                        p2.extend(pile)
                    pile = []
                    turn = winner
            else:
                turn = 3 - turn

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Uso: python3 inspect_hits_40.py hits_40.bin [report.txt]")
        sys.exit(1)

    hits_path = sys.argv[1]
    report_path = sys.argv[2] if len(sys.argv) == 3 else "hits_40_report.txt"

    hits = []
    try:
        with open(hits_path, "rb") as fh:
            data = fh.read()
    except FileNotFoundError:
        print(f"ERRORE: File {hits_path} non trovato.")
        sys.exit(1)
        
    record_size = 8 + 4
    n_total = len(data) // record_size
    if len(data) % record_size != 0:
        print(f"ATTENZIONE: {hits_path} ha una dimensione non multipla di {record_size} byte. "
              f"Il file potrebbe essere stato troncato.")

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
        print(f"Verifico solo i primi {MAX_TO_VERIFY} candidati su {len(hits)} totali...")

    print(f"Analisi dei cicli su {len(to_verify)} candidati (l'hashing può richiedere qualche secondo)...")

    rows = []
    infiniti_count = 0

    for idx, (rank, turns_gpu) in enumerate(to_verify):
        deck = unrank(table, dims, rank, COUNTS)
        
        is_inf, turni_totali = check_infinite(deck[:half], deck[half:])
        
        if is_inf:
            infiniti_count += 1
            
        rows.append((rank, deck, turns_gpu, is_inf, turni_totali))
        
        if (idx + 1) % 500 == 0:
            print(f"  ...{idx + 1}/{len(to_verify)} analizzati (Trovati {infiniti_count} loop)")

    print(f"\nVerifica completata: {infiniti_count} partite infinite matematicamente confermate.")

    with open(report_path, "w") as fh:
        fh.write(f"Candidati totali nel file: {n_total}\n")
        fh.write(f"Candidati analizzati: {len(to_verify)}\n")
        fh.write(f"Partite INFINITE trovate: {infiniti_count}\n\n")
        fh.write(f"{'rank':>18}  {'esito':>12}  {'turni_pre_loop':>16}  {'turni_gpu':>10}  sequenza\n")
        fh.write("-" * 110 + "\n")
        for rank, deck, turns_gpu, is_inf, turni_py in rows:
            esito = "INFINITO" if is_inf else "TERMINATA"
            seq_str = "".join(map(str, deck))
            fh.write(f"{rank:>18}  {esito:>12}  {turni_py:>16}  {turns_gpu:>10}  {seq_str}\n")

    print(f"Scritto {report_path}")

if __name__ == "__main__":
    main()
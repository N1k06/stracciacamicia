#!/bin/bash
# merge_and_status.sh
#
# Unisce i file hits_gpuN.bin di tutte le GPU in un unico file (il formato
# senza header di straccia_search_40.cu si concatena direttamente, senza
# bisogno di parsing) e mostra lo stato di avanzamento di ciascun processo,
# inclusa la PERCENTUALE COMPLESSIVA dell'intero spazio esplorata finora
# (letta da run_state.txt, scritto da launch_multi_gpu.sh al momento del
# lancio, che registra i confini reali assegnati a ciascuna GPU).
#
# Uso:
#   ./merge_and_status.sh [output=hits_40_merged.bin]

OUTPUT="${1:-hits_40_merged.bin}"

echo "=== Stato dei processi (checkpoint per GPU) ==="
for f in checkpoint_gpu*.txt; do
    [ -f "$f" ] || continue
    idx=$(echo "$f" | grep -oE '[0-9]+')
    val=$(cat "$f")
    echo "GPU $idx: batch_start=$val"
done

echo ""
if [ -f run_state.txt ]; then
    python3 -c "
import sys

with open('run_state.txt') as fh:
    lines = [l.strip() for l in fh if l.strip()]

total_space = int(lines[0])
resume_from = int(lines[1])
gpu_ranges = {}
for line in lines[2:]:
    parts = line.split()
    idx, start, end = int(parts[0]), int(parts[1]), int(parts[2])
    gpu_ranges[idx] = (start, end)

total_done = resume_from
per_gpu_lines = []
for idx, (start, end) in sorted(gpu_ranges.items()):
    try:
        with open(f'checkpoint_gpu{idx}.txt') as fh:
            checkpoint = int(fh.read().strip())
    except (FileNotFoundError, ValueError):
        checkpoint = start  # nessun batch completato ancora per questa GPU

    done_here = max(0, checkpoint - start)
    assigned = end - start
    pct_local = 100.0 * done_here / assigned if assigned > 0 else 0.0
    per_gpu_lines.append(f'  GPU {idx}: {done_here:,} / {assigned:,} nella propria porzione ({pct_local:.4f}%)')
    total_done += done_here

pct_total = 100.0 * total_done / total_space

print('=== Percentuale complessiva dell\'intero spazio ===')
for l in per_gpu_lines:
    print(l)
print()
if resume_from > 0:
    print(f'Gia\' completato prima di questo multi-GPU (resume_from): {resume_from:,}')
print(f'Totale configurazioni esplorate: {total_done:,} / {total_space:,}')
print(f'Percentuale COMPLESSIVA: {pct_total:.6f}%')
"
else
    echo "ATTENZIONE: run_state.txt non trovato (lanciato con una versione precedente di"
    echo "launch_multi_gpu.sh?). Impossibile calcolare la percentuale complessiva reale;"
    echo "vedi solo i batch_start grezzi sopra."
fi

echo ""
echo "=== Unione dei file di hit ==="
rm -f "$OUTPUT"
n_files=0
for f in hits_gpu*.bin; do
    [ -f "$f" ] || continue
    cat "$f" >> "$OUTPUT"
    n_files=$((n_files + 1))
    echo "  aggiunto $f ($(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") byte)"
done

if [ "$n_files" -eq 0 ]; then
    echo "Nessun file hits_gpuN.bin trovato."
    exit 1
fi

total_size=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")
n_hits=$((total_size / 12))
echo ""
echo "File unito: $OUTPUT ($total_size byte = $n_hits candidati totali)"
echo ""
echo "Prossimi passi:"
echo "  python3 inspect_hits_40.py $OUTPUT hits_40_report.txt"
echo "  python3 confirm_cycle.py --counts 28,4,4,4 $OUTPUT --no-header --max-turns 2000000 --report cycles_report.txt"
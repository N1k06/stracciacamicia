#!/bin/bash
# merge_and_status.sh
#
# Unisce i file hits_gpuN.bin di tutte le GPU in un unico file (il formato
# senza header di straccia_search_40.cu si concatena direttamente, senza
# bisogno di parsing) e mostra lo stato di avanzamento di ciascun processo.
#
# Uso:
#   ./merge_and_status.sh [output=hits_40_merged.bin]

OUTPUT="${1:-hits_40_merged.bin}"
TOTAL_SPACE=193584473082000

echo "=== Stato dei processi (checkpoint per GPU) ==="
for f in checkpoint_gpu*.txt; do
    [ -f "$f" ] || continue
    idx=$(echo "$f" | grep -oE '[0-9]+')
    val=$(cat "$f")
    echo "GPU $idx: batch_start=$val"
done

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

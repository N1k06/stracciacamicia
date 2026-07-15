#!/bin/bash
# launch_multi_gpu.sh
#
# Divide lo spazio totale delle configurazioni in N porzioni uguali (N = numero
# di GPU rilevate sull'istanza) e lancia un processo indipendente per ciascuna,
# isolato tramite CUDA_VISIBLE_DEVICES. Ogni processo ha il proprio checkpoint
# e il proprio file di hit, cosi' non c'e' alcuna necessita' di comunicazione
# o sincronizzazione tra le GPU (il problema e' perfettamente parallelizzabile:
# ogni batch/porzione e' indipendente dalle altre).
#
# Uso:
#   ./launch_multi_gpu.sh [max_turns=5000] [time_budget_seconds=360000] [target_batch_seconds=60]
#
# Presuppone che straccia_search_40.cu sia gia' compilato come ./straccia_search_40
# (vedi VASTAI_INSTRUCTIONS.md per la compilazione) e che multinomial_table.bin
# sia nella directory corrente.

set -e

MAX_TURNS="${1:-5000}"
TIME_BUDGET="${2:-360000}"
TARGET_BATCH_SECONDS="${3:-60}"
TABLE="multinomial_table.bin"
TOTAL_SPACE=193584473082000   # multinomial(28,4,4,4), costante nota per il mazzo reale

if [ ! -f "$TABLE" ]; then
    echo "Errore: $TABLE non trovato nella directory corrente."
    exit 1
fi
if [ ! -x "./straccia_search_40" ]; then
    echo "Errore: ./straccia_search_40 non trovato o non eseguibile. Compilalo prima."
    exit 1
fi

N_GPUS=$(nvidia-smi -L | wc -l)
if [ "$N_GPUS" -eq 0 ]; then
    echo "Errore: nessuna GPU rilevata da nvidia-smi."
    exit 1
fi

echo "GPU rilevate: $N_GPUS"
echo "Spazio totale: $TOTAL_SPACE configurazioni"
echo "Porzione per GPU: $((TOTAL_SPACE / N_GPUS)) configurazioni circa"
echo ""

mkdir -p logs

for ((i=0; i<N_GPUS; i++)); do
    RANGE_START=$(( TOTAL_SPACE * i / N_GPUS ))
    RANGE_END=$(( TOTAL_SPACE * (i+1) / N_GPUS ))
    # L'ultima GPU copre fino alla fine esatta, senza arrotondamenti persi
    if [ "$i" -eq $((N_GPUS - 1)) ]; then
        RANGE_END=0   # 0 = "fino alla fine dello spazio totale" per il programma
    fi

    CHECKPOINT="checkpoint_gpu${i}.txt"
    HITS="hits_gpu${i}.bin"
    LOG="logs/gpu${i}.log"

    echo "GPU $i: range [$RANGE_START, ${RANGE_END:-fine}) -> checkpoint=$CHECKPOINT hits=$HITS log=$LOG"

    CUDA_VISIBLE_DEVICES=$i nohup ./straccia_search_40 \
        "$TABLE" 10000000 "$MAX_TURNS" 100000 "$TIME_BUDGET" \
        "$CHECKPOINT" "$HITS" "$TARGET_BATCH_SECONDS" 2048 256 \
        "$RANGE_START" "$RANGE_END" \
        > "$LOG" 2>&1 &

    echo "  -> lanciato con PID $!"
done

echo ""
echo "Tutti i $N_GPUS processi sono stati lanciati in background."
echo "Monitora con:  tail -f logs/gpu0.log   (o il numero che ti interessa)"
echo "Controlla tutti i processi con:  jobs -l"
echo "Controlla il progresso di tutte le GPU con:  for f in checkpoint_gpu*.txt; do echo -n \"\$f: \"; cat \$f; done"

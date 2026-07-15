#!/bin/bash
# launch_multi_gpu.sh
#
# Divide la porzione RIMANENTE dello spazio delle configurazioni (dal punto di
# ripresa RESUME_FROM fino alla fine) in N parti uguali (N = numero di GPU
# rilevate sull'istanza) e lancia un processo indipendente per ciascuna,
# isolato tramite CUDA_VISIBLE_DEVICES. Ogni processo ha il proprio checkpoint
# e il proprio file di hit, cosi' non c'e' alcuna necessita' di comunicazione
# o sincronizzazione tra le GPU (il problema e' perfettamente parallelizzabile:
# ogni batch/porzione e' indipendente dalle altre).
#
# Uso:
#   ./launch_multi_gpu.sh [max_turns=5000] [time_budget_seconds=360000] [target_batch_seconds=60] [resume_from=0] [batch_size=10000000]
#
# target_batch_seconds: metti a 0 per DISATTIVARE l'auto-tuning e usare
# batch_size come valore FISSO per tutta l'esecuzione (invece che come seed
# iniziale che poi si auto-regola).
#
# resume_from: usalo per riprendere da un progresso gia' fatto altrove (es. un
# checkpoint.txt di un run precedente su Colab a singola GPU) invece di
# ripartire dall'inizio dell'intero spazio. Leggi il valore dal vecchio
# checkpoint.txt e passalo qui -- lo script dividera' equamente tra le GPU
# solo la porzione [resume_from, totale), non l'intero spazio da zero.
#
# Presuppone che straccia_search_40.cu sia gia' compilato come ./straccia_search_40
# (vedi VASTAI_INSTRUCTIONS.md per la compilazione) e che multinomial_table.bin
# sia nella directory corrente.

set -e

MAX_TURNS="${1:-5000}"
TIME_BUDGET="${2:-360000}"
TARGET_BATCH_SECONDS="${3:-60}"
RESUME_FROM="${4:-0}"
BATCH_SIZE="${5:-10000000}"
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
if [ "$RESUME_FROM" -ge "$TOTAL_SPACE" ]; then
    echo "Errore: resume_from ($RESUME_FROM) e' >= dello spazio totale ($TOTAL_SPACE). Nulla da fare."
    exit 1
fi

# --- Controllo di sicurezza: checkpoint preesistenti da un lancio precedente ---
# Un checkpoint gia' presente viene SEMPRE riusato dal programma (ha priorita'
# sul range_start assegnato qui). Se i confini di questo lancio sono diversi
# da quelli del lancio che ha generato quei checkpoint (es. numero di GPU
# diverso, resume_from diverso), il valore riletto puo' risultare incoerente
# con la nuova porzione assegnata -- tipicamente si manifesta come percentuali
# di completamento assurde (es. gia' al 70% dopo pochi secondi).
EXISTING=$(ls checkpoint_gpu*.txt 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "⚠️  ATTENZIONE: trovati checkpoint preesistenti in questa directory:"
    for f in $EXISTING; do
        echo "    $f -> $(cat "$f" 2>/dev/null || echo '(vuoto o illeggibile)')"
    done
    echo ""
    echo "Questi verranno RIUSATI cosi' come sono (hanno priorita' sul range_start"
    echo "calcolato qui). Se provengono da un lancio con un numero di GPU o un"
    echo "resume_from DIVERSI da quello attuale, il progresso letto potrebbe non"
    echo "corrispondere alla nuova suddivisione. Se non e' quello che vuoi:"
    echo "    rm -f checkpoint_gpu*.txt hits_gpu*.bin run_state.txt"
    echo "e rilancia."
    echo ""
    read -p "Vuoi procedere comunque riusando questi checkpoint? [y/N] " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Interrotto. Ripulisci i checkpoint se vuoi un lancio pulito, poi rilancia."
        exit 1
    fi
fi

N_GPUS=$(nvidia-smi -L | wc -l)
if [ "$N_GPUS" -eq 0 ]; then
    echo "Errore: nessuna GPU rilevata da nvidia-smi."
    exit 1
fi

REMAINING=$(( TOTAL_SPACE - RESUME_FROM ))

echo "GPU rilevate: $N_GPUS"
echo "Spazio totale: $TOTAL_SPACE configurazioni"
if [ "$RESUME_FROM" -gt 0 ]; then
    echo "Ripresa da: $RESUME_FROM (progresso gia' fatto altrove)"
    echo "Porzione rimanente da dividere: $REMAINING configurazioni ($(awk "BEGIN{printf \"%.2f\", 100.0*$RESUME_FROM/$TOTAL_SPACE}")% gia' completato)"
fi
echo "Porzione per GPU: $((REMAINING / N_GPUS)) configurazioni circa"
if [ "$TARGET_BATCH_SECONDS" = "0" ]; then
    echo "Batch size: FISSO a $BATCH_SIZE (auto-tuning disattivato)"
else
    echo "Batch size: auto-tuning attivo (seed $BATCH_SIZE, target ${TARGET_BATCH_SECONDS}s/batch)"
fi
echo ""

mkdir -p logs
: > run_state.txt   # azzera/crea il file di metadati per questo lancio
echo "$TOTAL_SPACE" >> run_state.txt
echo "$RESUME_FROM" >> run_state.txt

for ((i=0; i<N_GPUS; i++)); do
    RANGE_START=$(( RESUME_FROM + REMAINING * i / N_GPUS ))
    RANGE_END=$(( RESUME_FROM + REMAINING * (i+1) / N_GPUS ))
    # L'ultima GPU copre fino alla fine esatta, senza arrotondamenti persi
    EFFECTIVE_END=$RANGE_END
    if [ "$i" -eq $((N_GPUS - 1)) ]; then
        RANGE_END=0        # 0 = "fino alla fine dello spazio totale" per il programma
        EFFECTIVE_END=$TOTAL_SPACE   # valore reale (non il sentinella 0) da salvare nei metadati
    fi

    # Salva il confine REALE (mai 0-come-sentinella) cosi' merge_and_status.sh
    # puo' calcolare quanto ciascuna GPU ha effettivamente da fare, senza dover
    # replicare la logica del sentinella "0 = fino alla fine".
    echo "$i $RANGE_START $EFFECTIVE_END" >> run_state.txt

    CHECKPOINT="checkpoint_gpu${i}.txt"
    HITS="hits_gpu${i}.bin"
    LOG="logs/gpu${i}.log"

    echo "GPU $i: range [$RANGE_START, ${RANGE_END:-fine}) -> checkpoint=$CHECKPOINT hits=$HITS log=$LOG"

    CUDA_VISIBLE_DEVICES=$i nohup stdbuf -oL -eL ./straccia_search_40 \
        "$TABLE" "$BATCH_SIZE" "$MAX_TURNS" 100000 "$TIME_BUDGET" \
        "$CHECKPOINT" "$HITS" "$TARGET_BATCH_SECONDS" 2048 256 \
        "$RANGE_START" "$RANGE_END" \
        > "$LOG" 2>&1 &

    echo "  -> lanciato con PID $!"
done

echo ""
echo "Tutti i $N_GPUS processi sono stati lanciati in background."
echo "Monitora con:  tail -f logs/gpu0.log   (o il numero che ti interessa)"
echo "Controlla tutti i processi con:  jobs -l"
echo "Controlla il progresso complessivo con:  ./merge_and_status.sh"
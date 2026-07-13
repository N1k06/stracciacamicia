# Come eseguire il test su Google Colab

## 1. Prepara il runtime

- Apri un nuovo notebook su https://colab.research.google.com
- `Runtime` → `Change runtime type` → `Hardware accelerator` → seleziona una GPU (per
  questo primo test va benissimo anche il T4 gratuito: il test coinvolge solo 701
  configurazioni, non serve potenza)
- Verifica che la GPU sia effettivamente assegnata:
  ```
  !nvidia-smi
  ```

## 2. Carica i due file su Colab

Nel pannello a sinistra (icona cartella) trascina i due file:
- `gen_table_and_tests.py`
- `test_unrank.cu`

oppure, da una cella:
```python
from google.colab import files
uploaded = files.upload()  # seleziona i due file dal tuo computer
```

## 3. Genera la tabella e i dati di riferimento

```
!python3 gen_table_and_tests.py
```

Dovresti vedere lo stesso output ottenuto in locale:
```
Costruzione tabella multinomiale...
Totale configurazioni distinte: 193584473082000
Self-test round-trip in Python (unrank -> rank_of)...
OK: tutti i 2002 rank testati sono round-trip consistenti in Python.

Scritto multinomial_table.bin (3625 x uint64 = 29000 byte)
Numero di rank di test totali: 701
Scritti test_ranks.bin, reference_sequences.bin, reference_sequences.txt

Pronto per il test CUDA (test_unrank.cu).
```

Se questo self-test fallisse su Colab (non dovrebbe, è puro Python), fermati qui:
significherebbe un problema di ambiente, non di logica.

## 4. Verifica la compute capability della tua GPU

```
!nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Esempi tipici su Colab: T4 → `sm_75`, A100 → `sm_80`, L4 → `sm_89`.

## 5. Compila il test CUDA

```
!nvcc -O3 -arch=sm_75 --ptxas-options=-v test_unrank.cu -o test_unrank
```

(sostituisci `sm_75` con il valore corretto trovato al passo 4). L'opzione
`--ptxas-options=-v` stampa l'uso di registri per il kernel — controlla che non
compaiano `spill stores`/`spill loads` diversi da zero (con questo test, array così
piccoli e pochi thread, non dovrebbero comunque comparire problemi di questo tipo;
sarà più rilevante quando aggiungeremo `simulate()`).

## 6. Esegui il test

```
!./test_unrank multinomial_table.bin test_ranks.bin reference_sequences.bin
```

## 7. Cosa aspettarsi

Se tutto funziona:
```
Caricati 701 rank di test.

=== RISULTATO TEST ===
Rank testati: 701
Sequenze GPU vs riferimento Python: 701 OK, 0 MISMATCH
Round-trip rank->unrank->rank su GPU: 701 OK, 0 MISMATCH

TUTTI I TEST SUPERATI.
```

Se qualcosa non torna, il programma stampa su stderr i primi 10 casi di mismatch
(sia per il confronto diretto con Python sia per il round-trip), con il rank
coinvolto e le due sequenze a confronto — utile per isolare rapidamente dove
diverge la logica (es. un ordine dei simboli invertito, un errore di indicizzazione
nella tabella).

## Prossimo passo

Una volta che questo test passa in modo pulito, il prossimo blocco da validare è
`simulate()` (la logica di gioco), seguendo la stessa strategia: riferimento in
Python puro, confronto sistematico con l'output della GPU su un campione di
configurazioni, prima di lanciare qualunque batch a piena scala.
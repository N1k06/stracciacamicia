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

## Prossimo passo: validare simulate() (§7 del README)

Una volta che il test di unranking passa in modo pulito, si passa alla logica di
gioco vera e propria. Qui la strategia cambia leggermente: invece di un campione
di rank, enumeriamo **tutte** le configurazioni possibili di due mazzi ridotti
(8 e 10 carte), dato che lo spazio è abbastanza piccolo da testarlo per intero.

### 1. Genera i dati di test esaustivi

Carica anche `gen_game_test_data.py` su Colab, poi:

```
!python3 gen_game_test_data.py
```

Output atteso:
```
=== Composizione 'small_a': counts=[2, 2, 2, 2], carte totali=8 ===
Configurazioni distinte da enumerare: 2520
Scritto table_small_a.bin (header + 81 x uint64)
Turni massimi osservati: 15
Configurazioni che raggiungono il tetto (2000 turni): 0
Scritti turns_small_a.bin, turns_small_a.txt

=== Composizione 'small_b': counts=[4, 2, 2, 2], carte totali=10 ===
Configurazioni distinte da enumerare: 18900
Scritto table_small_b.bin (header + 135 x uint64)
Turni massimi osservati: 32
Configurazioni che raggiungono il tetto (2000 turni): 0
Scritti turns_small_b.bin, turns_small_b.txt

Tutti i dati di test generati. Pronto per test_simulate.cu.
```

### 2. Compila test_simulate.cu

Carica anche `test_simulate.cu`, poi:

```
!nvcc -O3 -arch=sm_75 --ptxas-options=-v test_simulate.cu -o test_simulate
```

(sostituisci `sm_75` con la compute capability trovata al passo 4 della sezione
precedente).

### 3. Esegui il test su entrambe le composizioni

```
!./test_simulate table_small_a.bin turns_small_a.bin
!./test_simulate table_small_b.bin turns_small_b.bin
```

Output atteso per ciascuna:
```
Composizione: [2,2,2,2]  carte totali=8  meta'=4
Configurazioni da testare (esaustivo): 2520  max_turns=2000

=== RISULTATO TEST ===
Configurazioni testate: 2520
Corrispondenze: 2520 OK, 0 MISMATCH

TUTTI I TEST SUPERATI (validazione esaustiva).
```

Questo è un test **esaustivo**, non a campione: se passa su entrambe le
composizioni, hai una copertura molto solida della logica di `simulate()` —
inclusi i casi limite come catene di carte vincenti che si susseguono, e
giocatori che esauriscono le carte a metà di una risposta dovuta.

Se qualcosa non torna, il programma stampa su stderr i primi 15 rank in
mismatch con il conteggio turni GPU vs Python, utile per isolare rapidamente
la configurazione problematica e, tramite il rank, ricostruire la sequenza
esatta di carte con `unrank_multiset` in Python per ispezionarla a mano.

## Prossimo passo dopo questo

Una volta che anche `simulate()` passa la validazione esaustiva sui mazzi
ridotti, si converte la rappresentazione da array semplice a bit-packed
(paragrafo 8 del README) e si ripete la stessa validazione, per assicurarsi
che la conversione non introduca bug. Solo dopo si integra tutto nel kernel
persistente per il run a piena scala.

---

# Ricerca esaustiva completa: mazzo ridotto a 20 carte

Prima di passare al mazzo reale da 40 carte (~1,9·10¹⁴ configurazioni), questo
passo copre per intero un mazzo ridotto a 20 carte con la stessa proporzione
del mazzo reale (14:2:2:2, cioè 7:1:1:1 come 28:4:4:4 su 40). Il totale è
**3.488.400 configurazioni**: abbastanza per essere un vero test end-to-end
della pipeline di produzione (unranking + simulate + kernel persistente con
coda di lavoro atomica), ma piccolo abbastanza da coprirlo tutto in un solo
lancio, in pochi secondi/minuti anche su una T4 gratuita.

A differenza dei passi precedenti, qui l'obiettivo non è solo validare la
logica (già fatto esaustivamente sopra), ma anche **trovare candidati reali**
e testare il throughput della pipeline completa.

## 1. Carica i file necessari

- `straccia_common.py` (modulo condiviso, richiesto da `gen_table_20.py` e `validate_hits_20.py`)
- `gen_table_20.py`
- `search_full_20.cu`
- `validate_sample_20.cu`
- `validate_hits_20.py`

## 2. Genera i dati

```
!python3 gen_table_20.py
```

Output atteso:
```
Composizione: [14, 2, 2, 2] (14:2:2:2, stessa proporzione 7:1:1:1 del mazzo reale)
Totale configurazioni distinte: 3488400
Carte totali: 20, meta': 10
Scritto table_20.bin (header + 405 x uint64)
Calcolo riferimento Python per 5000 rank campione (max_turns=20000)...
Turni massimi osservati nel campione: 349
Configurazioni nel campione che raggiungono il tetto (20000): 0
Scritti sample_ranks_20.bin, sample_turns_20.bin

Pronto per search_full_20.cu (ricerca esaustiva su GPU) e validate_sample_20.cu.
```

(i numeri esatti del campione possono variare leggermente se cambi il seed,
ma l'ordine di grandezza dovrebbe essere simile)

## 3. Validazione indipendente sul campione (facoltativa ma consigliata)

```
!nvcc -O3 -arch=sm_75 validate_sample_20.cu -o validate_sample_20
!./validate_sample_20 table_20.bin sample_ranks_20.bin sample_turns_20.bin
```

Deve dare `5000 OK, 0 MISMATCH`.

## 4. Ricerca esaustiva completa su GPU

```
!nvcc -O3 -arch=sm_75 search_full_20.cu -o search_full_20
!./search_full_20 table_20.bin
```

Questo usa i default: `max_turns=5000`, `hit_threshold=5000` (stesso criterio
del progetto principale: "candidato potenzialmente infinito" = raggiunge il
tetto di turni), `max_hits=100000`. Puoi personalizzarli:

```
!./search_full_20 table_20.bin 5000 5000 100000 hits_20.bin
```

Output atteso (i numeri esatti dipendono dai risultati reali):
```
Composizione: [14,2,2,2]  carte=20  meta'=10
Totale configurazioni da esaminare: 3488400
max_turns=5000  hit_threshold=5000  max_hits=100000

=== RISULTATO RICERCA ===
Tempo GPU: 0.XXX secondi
Throughput: XX.XX milioni di configurazioni/secondo
Turni massimi trovati: NNN
Configurazioni con turni >= 5000 (candidate): N

Distribuzione turni:
  <10:            ...
  10-49:          ...
  50-199:         ...
  200-999:        ...
  1000-4999: ...
  >= 5000 (tetto):  N

Scritti N hit in hits_20.bin
```

Se il numero di candidati è 0, significa che nessuna configurazione di questo
mazzo ridotto forma un ciclo abbastanza lungo da superare 5000 turni — un
risultato legittimo e comunque interessante (specialmente se lo confronti con
la letteratura esistente sulle varianti ridotte di questo tipo di gioco). Se
invece trovi candidati, il passo successivo li verifica uno per uno.

## 5. Validazione mirata di ogni singolo candidato trovato

```
!python3 validate_hits_20.py table_20.bin hits_20.bin
```

Dato che ci aspettiamo pochi candidati (non milioni), questo script li
ricalcola **tutti** in Python (non solo un campione) e verifica che il numero
di turni coincida esattamente con quello riportato dalla GPU. Se anche questo
passa, hai una conferma end-to-end molto solida: la pipeline completa
(unranking + simulate + kernel persistente) funziona correttamente anche a
scala di milioni di configurazioni, non solo sulle migliaia testate prima.

## 6. Cosa fare con i risultati

Se trovi candidati con `turni >= 5000`, puoi ricostruire la configurazione
completa di carte a partire dal rank con `unrank` (in `straccia_common.py`),
ad esempio:

```python
import struct
from straccia_common import build_table, unrank

counts = [14, 2, 2, 2]
table, dims = build_table(counts)
rank = ...  # uno dei rank trovati in hits_20.bin
deck = unrank(table, dims, rank, counts)
print(''.join(map(str, deck)))
```

Se conosci risultati pubblicati per varianti ridotte di questo gioco, questo è
il punto per confrontare: stesso numero di configurazioni cicliche trovate?
Stessa lunghezza dei cicli più lunghi? Discrepanze qui meriterebbero
un'indagine più a fondo prima di fidarsi del run a piena scala da 40 carte.

## Prossimo passo dopo questo

Se tutto torna (validazione a campione OK, validazione candidati OK, e
idealmente un riscontro con risultati noti in letteratura, se disponibili),
il passo successivo è convertire `simulate()` alla rappresentazione
bit-packed (paragrafo 8 del README) e ripetere questo stesso identico test
sul mazzo da 20 carte, confrontando che i risultati (candidati trovati,
turni massimi, istogramma) siano **identici** tra le due versioni. Solo dopo
si passa al mazzo reale da 40 carte, suddiviso nei circa 967.922 batch.
# Stracciacamicia infinita

Repository per la prototipazione e la ricerca di configurazioni iniziali che generano cicli infiniti nella variante italiana del gioco "Straccia Camicia" (simile a "Beggar My Neighbour").

Basato su lavori precedenti e repository affini:
- [alessandro-gentilini](https://github.com/alessandro-gentilini/pelagaletto2): ottima introduzione al problema e analisi dei cicli con 20 carte.
- [drago-96](https://github.com/drago-96/cavacamisa): ricerca di partite infinite e ritrovamento del primo ciclo con 40 carte.

Questo progetto unisce:
- generazione e campionamento di permutazioni multiset per il mazzo reale da 40 carte
- validazione Python delle funzioni di unranking e simulazione
- implementazioni CUDA per test e ricerca su GPU
- esperimenti su mazzi ridotti da 20 carte come passaggio di validazione

## Struttura del repository

- `batches-generation/`
  - `gen_perm_batches.py`: script Python che genera permutazioni distinte del mazzo da 40 carte con conteggi `[28, 4, 4, 4]` e le stampa a intervalli regolari.
  - `perm_batches_200_000_000.txt`: esempio di output/serie di permutazioni pre-generate.

- `cuda-simulation/`
  - `README.md`: istruzioni dettagliate per la validazione e il test su Google Colab.
  - `20_cards/`: implementazioni e script per la fase di sviluppo e test su mazzi ridotti di 20 carte.
  - `40_cards/`: codice di simulazione e analisi per il mazzo reale di 40 carte.

## Contenuti principali

- `cuda-simulation/20_cards/straccia_common.py`: implementazione Python di riferimento per l'unranking e la simulazione delle regole del gioco.
- `cuda-simulation/20_cards/gen_table_20.py`: genera tabelle multinomiali e dati di riferimento usati per validare la logica CUDA.
- `cuda-simulation/20_cards/test_unrank.cu`, `validate_hits_20.py`, `validate_sample_20.cu`: test di correttezza e confronto tra GPU e riferimento Python.
- `cuda-simulation/20_cards/search_full_20.cu`: versione GPU per enumerazione e ricerca su mazzi da 20 carte.
- `cuda-simulation/40_cards/straccia_common.py`: versione Python di riferimento condivisa per le strutture dati e la simulazione.
- `cuda-simulation/40_cards/straccia_search_40.cu`: kernel CUDA progettato per la ricerca ad alte prestazioni sul mazzo completo da 40 carte.
- `cuda-simulation/40_cards/confirm_cycle.py`, `inspect_hits_40.py`: script di analisi per validare i risultati e ispezionare le configurazioni trovate.

## Obiettivo del progetto

L'obiettivo non è creare un gioco completo, ma esplorare e validare una pipeline di ricerca per trovare configurazioni iniziali che producono una partita infinita. Il flusso tipico è:

1. validare in Python la logica di unranking e simulazione su mazzi ridotti;
2. testare le versioni CUDA contro i riferimenti Python;
3. estendere la ricerca al mazzo completo da 40 carte;
4. analizzare e confermare eventuali configurazioni cicliche trovate.

## Come iniziare

1. Leggi `cuda-simulation/README.md` per i dettagli di setup e validazione su GPU.
2. Usa `batches-generation/gen_perm_batches.py` per generare porzioni di permutazioni del mazzo da 40 carte.
3. Esplora `cuda-simulation/20_cards` per capire la pipeline di validazione prima di passare al codice `40_cards`.

> Nota: la maggior parte dei file CUDA e dei test è orientata a verificare la correttezza della logica e ad analizzare il comportamento del gioco, non solo a eseguire partite singole.

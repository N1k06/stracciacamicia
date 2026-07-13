# Ricerca partite infinite in Straccia Camicia (CUDA)

Questo documento spiega **da zero** tutto il progetto: cosa stiamo cercando, perché,
come funziona la matematica dell'enumerazione dei mazzi, come funziona l'esecuzione
su GPU, e perché ogni scelta implementativa è stata fatta. È scritto per essere
seguito in ordine, anche senza conoscenze pregresse di CUDA.

---

## 1. Il problema in termini semplici

**Straccia Camicia** (variante italiana di *Beggar-My-Neighbour*) è un gioco di carte
completamente deterministico: una volta fissato l'ordine delle 40 carte nel mazzo
iniziale (e quindi come vengono divise tra i due giocatori), **l'intera partita è
già decisa**. Non ci sono scelte da fare durante il gioco: si gioca sempre la carta
in cima alla propria mano, seguendo regole fisse.

Questo significa che l'intero "spazio di ricerca" è finito: è l'insieme di tutte le
possibili configurazioni iniziali del mazzo. Il nostro obiettivo è **scandire tutte
queste configurazioni** (o un campione rappresentativo via simmetrie, se in futuro
vorrai ridurre lo spazio) e per ciascuna simulare la partita, registrando quelle che
**sembrano non finire mai** (le chiamiamo "potenzialmente infinite": non terminano
entro un tetto di 5000 turni).

Perché "potenzialmente"? Perché con un tetto arbitrario di turni non puoi distinguere
con certezza assoluta un gioco che gira in un ciclo perenne da uno che semplicemente
impiega più di 5000 turni per finire in modo naturale. Per questo la strategia è a
due fasi:

1. **Fase 1 (questo documento):** scrematura veloce su GPU di *tutte* le
   configurazioni, per isolare quelle che superano 5000 turni. Questo elimina la
   stragrande maggioranza dei casi (che finiscono in poche decine/centinaia di
   turni) e produce una lista molto più piccola di "candidati".
2. **Fase 2 (successiva, non coperta qui):** analisi più approfondita e rigorosa
   sui soli candidati, ad esempio con rilevamento di cicli esatto (vedi §9), per
   distinguere i cicli veri dalle partite semplicemente molto lunghe.

---

## 2. Quante configurazioni ci sono? (la matematica dell'enumerazione)

### 2.1 Perché non è semplicemente 40!

Il mazzo ha 40 carte, ma non tutte sono distinguibili ai fini del gioco: quello che
conta per l'andamento della partita non è *quale* carta specifica (es. "3 di
coppe") occupa una posizione, ma **il suo valore di gioco**:

- `0` = carta "non vincente" (non forza risposta) — ce ne sono **28**
- `1` = carta che forza l'avversario a rispondere con **1** carta — ce ne sono **4**
- `2` = carta che forza risposta con **2** carte — ce ne sono **4**
- `3` = carta che forza risposta con **3** carte — ce ne sono **4**

Totale: 28+4+4+4 = 40. ✓

Quindi lo spazio di ricerca non è l'insieme delle permutazioni di 40 oggetti
distinti (40! ≈ 8·10⁴⁷, un numero assurdo), ma l'insieme delle **sequenze distinte**
di questo *multiset* (insieme con ripetizioni) di simboli {0,1,2,3}. Il numero di
tali sequenze distinte è dato dal **coefficiente multinomiale**:

```
                    40!
totale = ─────────────────────────
          28! · 4! · 4! · 4!
```

Questo si calcola in Python con `math.factorial` senza problemi, perché Python
gestisce interi arbitrariamente grandi. Il risultato è:

```
totale ≈ 1.936 × 10^14   (circa 193 mila miliardi di configurazioni)
```

Questo è il numero di **mazzi distinti** da esaminare. È un numero enorme ma finito,
ed è il motivo per cui serve una GPU: anche a centinaia di milioni di simulazioni al
secondo, servono giorni/settimane di calcolo.

### 2.2 Cos'è "l'unranking" e perché ci serve

Con 1.9·10¹⁴ configurazioni, non possiamo generarle tutte in una lista e salvarle
su disco (occuperebbero terabyte). Invece, usiamo un trucco standard di
combinatoria: **numeriamo** tutte le configurazioni distinte da `0` a `totale-1`
(questa numerazione si chiama *rank*, "grado"), e scriviamo una funzione che,
dato un rank, **ricostruisce al volo** la configurazione corrispondente, senza
doverle generare tutte in ordine. Questa funzione si chiama **unranking**.

Il tuo script Python (`unrank_multiset`) fa esattamente questo:

```python
def unrank_multiset(rank, multiset):
    counts = Counter(multiset)   # quante carte di ogni tipo restano da piazzare
    result = []
    for _ in range(length):       # per ogni posizione nella sequenza (0..39)
        for char in sorted(counts):  # prova i simboli in ordine 0,1,2,3
            if counts[char] == 0:
                continue
            counts[char] -= 1
            perms = multinomial(counts)  # quante sequenze iniziano con questo prefisso
            if rank < perms:
                result.append(char)   # il rank cade in questo blocco: fissa il simbolo
                break
            else:
                rank -= perms          # salta oltre questo blocco, prova il prossimo simbolo
                counts[char] += 1
    return ''.join(result)
```

**Intuizione:** immagina di elencare tutte le 1.9·10¹⁴ sequenze in ordine
lessicografico (come in un dizionario, con `0 < 1 < 2 < 3`). Tutte quelle che
iniziano con `0` vengono prima di tutte quelle che iniziano con `1`, e così via.
Il numero di sequenze che iniziano con un dato prefisso è ancora un coefficiente
multinomiale (sui simboli rimanenti). Quindi, per trovare la sequenza numero
`rank`, all'inizio decidiamo il primo simbolo controllando in quale "blocco" cade
il rank; poi ripetiamo lo stesso ragionamento per il secondo simbolo, e così via
per tutte le 40 posizioni.

Questo è esattamente ciò che serve per la GPU: **ogni thread riceve solo un numero
intero (il rank) e ricostruisce da solo la propria configurazione**, senza bisogno
di ricevere 40 byte trasferiti dalla CPU.

---

## 3. Perché la generazione va fatta sulla GPU e non sulla CPU

Un batch, come definito nel tuo script, contiene `step = 2·10⁸` (200 milioni) di
configurazioni. Se le generassi in Python e le trasferissi alla GPU:

- 200.000.000 configurazioni × 40 byte/configurazione ≈ **8 GB per batch**
- il trasferimento su bus PCIe (anche veloce, ~20 GB/s) richiederebbe **~0.4
  secondi solo per il trasferimento**, per ogni batch, ripetuto ~967.922 volte
  (vedi §4) — un collo di bottiglia enorme e del tutto evitabile
- la CPU singola impiegherebbe comunque molto più tempo a generare 200M
  configurazioni di quanto la GPU impieghi a simularle

La soluzione: **portiamo l'unranking dentro il kernel CUDA**. Ogni thread GPU
riceve solo `batch_start` (un singolo intero a 64 bit, uguale per tutti i thread
del lancio) e calcola `rank = batch_start + tid` (dove `tid` è l'indice del
thread). Da questo rank, il thread ricostruisce da solo, internamente, la propria
sequenza di 40 carte, usando lo stesso algoritmo di unranking, ma scritto in
C++/CUDA invece che in Python.

**Cosa si trasferisce quindi da host a device?** Un singolo intero a 64 bit per
lancio di kernel. Tutto il resto avviene in parallelo, internamente alla GPU.

---

## 4. Quanti batch servono in totale?

```
numero di batch = totale / step = 1.936 × 10^14 / 2 × 10^8 ≈ 967.922 batch
```

Sono quasi un milione di lanci di kernel. Non è un problema di per sé (una GPU
moderna lancia kernel con overhead di pochi microsecondi), ma va tenuto conto
nella progettazione del ciclo host (uso di CUDA streams, I/O bufferizzato — vedi
§10).

---

## 5. La tabella dei coefficienti multinomiali (necessaria per l'unranking su GPU)

### 5.1 Il problema del fattoriale su GPU

Il codice CUDA gira con interi a **precisione fissa** (tipicamente `uint64_t`,
cioè un intero senza segno a 64 bit, che può rappresentare valori fino a
~1.8·10¹⁹). Il tuo Python usa `factorial()` su interi *arbitrariamente grandi*
(Python gestisce nativamente numeri enormi), ma `40!` da solo vale
**~8.16 × 10⁴⁷**, ben oltre quello che un `uint64_t` può contenere. Quindi non
possiamo calcolare `factorial()` direttamente in CUDA con questo approccio.

### 5.2 La soluzione: precalcolare solo i valori che servono davvero

Osservazione chiave: durante l'unranking, l'unica cosa che ci serve in ogni
istante è `multinomial(counts_residui)`, cioè il coefficiente multinomiale
calcolato sui simboli **rimasti da piazzare**, non sul totale di 40 carte.

I possibili valori di `counts_residui` sono limitati:

- `c0` (zeri rimasti) può andare da 0 a 28 → **29 valori possibili**
- `c1` (uno rimasti) può andare da 0 a 4 → **5 valori possibili**
- `c2` (due rimasti) → **5 valori possibili**
- `c3` (tre rimasti) → **5 valori possibili**

Quindi lo spazio di tutte le combinazioni possibili di `(c0,c1,c2,c3)` è:

```
29 × 5 × 5 × 5 = 3625 combinazioni
```

Per ciascuna di queste 3625 combinazioni, il valore `multinomial(c0,c1,c2,c3)` è
al massimo `multinomial(28,4,4,4) ≈ 1.9·10¹⁴`, che **entra comodamente** in un
`uint64_t` (che arriva fino a ~1.8·10¹⁹). Quindi possiamo **precalcolare tutti e
3625 i valori una volta sola, su CPU (in Python, dove i fattoriali grandi non
sono un problema), e caricarli come tabella fissa nella GPU**.

```python
from math import factorial

def f(c0, c1, c2, c3):
    n = c0 + c1 + c2 + c3
    if n == 0:
        return 1
    return factorial(n) // (factorial(c0) * factorial(c1) * factorial(c2) * factorial(c3))

table = []
for c0 in range(29):
    for c1 in range(5):
        for c2 in range(5):
            for c3 in range(5):
                table.append(f(c0, c1, c2, c3))

# 'table' ha 3625 elementi, ognuno rappresentabile in uint64_t.
# Va salvato in un formato che il codice C++/CUDA può caricare, ad es. binario:
import struct
with open('multinomial_table.bin', 'wb') as fh:
    for v in table:
        fh.write(struct.pack('<Q', v))  # '<Q' = uint64 little-endian
```

### 5.3 `__constant__` memory: cos'è e perché usarla

Nella terminologia CUDA, `__constant__` è uno spazio di memoria speciale
(dimensione totale tipica: 64 KB su quasi tutte le GPU NVIDIA) che:

- è **di sola lettura** dal punto di vista del kernel (si scrive solo dall'host,
  prima del lancio)
- ha una **cache dedicata**, molto veloce quando **tutti i thread di un warp
  leggono lo stesso indirizzo nello stesso momento** (broadcast) — che è
  esattamente il nostro caso, perché tutti i thread consultano la stessa tabella

La nostra tabella ha 3625 × 8 byte = **29.000 byte (~29 KB)**, ben dentro il
limite di 64 KB. È l'uso da manuale di `__constant__` memory.

```cpp
__constant__ uint64_t d_table[29 * 5 * 5 * 5];  // 3625 elementi
```

Sull'host, prima di lanciare i kernel, questa tabella va copiata una volta sola
con `cudaMemcpyToSymbol`:

```cpp
uint64_t h_table[3625];
// ... leggi h_table da 'multinomial_table.bin' generato in Python ...
cudaMemcpyToSymbol(d_table, h_table, sizeof(h_table));
```

### 5.4 Funzione di accesso alla tabella

Per leggere il valore corrispondente a una data quadrupla `(c0,c1,c2,c3)`, va
convertita in un indice lineare dell'array. Usiamo la stessa logica con cui è
stata costruita in Python (nested loop `c0, c1, c2, c3`):

```cpp
__device__ __forceinline__ uint64_t tbl(int c0, int c1, int c2, int c3) {
    // indice lineare coerente con l'ordine di costruzione in Python:
    // for c0 in range(29): for c1 in range(5): for c2 in range(5): for c3 in range(5)
    return d_table[((c0 * 5 + c1) * 5 + c2) * 5 + c3];
}
```

`__device__` significa "questa funzione gira sulla GPU, chiamata da altro codice
GPU". `__forceinline__` chiede al compilatore di espandere il corpo della
funzione direttamente nel punto di chiamata (niente overhead di funzione
separata) — utile perché questa funzione viene chiamata moltissime volte per
ogni singola simulazione.

### 5.5 Unranking in CUDA, spiegato riga per riga

```cpp
__device__ void unrank40(uint64_t rank, uint8_t out[40]) {
    int c[4] = {28, 4, 4, 4};  // quante carte di ogni simbolo restano da piazzare

    for (int pos = 0; pos < 40; pos++) {        // per ciascuna delle 40 posizioni...
        for (int sym = 0; sym < 4; sym++) {      // ...prova i simboli in ordine 0,1,2,3
            if (c[sym] == 0) continue;           // simbolo esaurito: salta

            c[sym]--;                            // ipotizza di piazzare 'sym' qui
            uint64_t perms = tbl(c[0], c[1], c[2], c[3]);  // quante sequenze restano
                                                             // possibili con questa scelta

            if (rank < perms) {
                out[pos] = sym;                  // il rank cade in questo blocco: conferma
                break;                            // e passa alla posizione successiva
            }
            rank -= perms;                        // altrimenti salta oltre questo blocco...
            c[sym]++;                              // ...e ripristina il conteggio per riprovare
        }
    }
}
```

Questo è l'esatto equivalente C++ del tuo `unrank_multiset` in Python, con due
differenze pratiche:

1. `counts` è un array fisso `int c[4]` invece di un `Counter` (dizionario) —
   più veloce e non richiede allocazioni dinamiche, cruciale su GPU dove
   l'allocazione dinamica per-thread è costosa o non disponibile.
2. `multinomial(counts)` è sostituito dalla lettura in tabella `tbl(...)` invece
   di essere ricalcolato ogni volta da zero — l'operazione più costosa (calcolo
   di fattoriali) è già stata fatta una volta per tutte su CPU in Python.

`out[40]` è un array di 40 `uint8_t` (interi a 8 bit senza segno, valori 0-255,
qui usati solo per 0-3) che al termine della funzione contiene la sequenza
completa di 40 simboli — l'equivalente della stringa restituita da
`unrank_multiset` in Python.

---

## 6. Le regole del gioco, formalizzate senza ambiguità

Prima di scrivere `simulate()`, formalizziamo con precisione le regole che hai
descritto, perché ogni dettaglio cambia il risultato della simulazione:

1. **Distribuzione iniziale.** Il mazzo di 40 carte si divide in due metà da 20.
   Nel nostro schema, la sequenza `deck[0..39]` prodotta dall'unranking va divisa
   in due mani. Se la tua definizione di "prima metà / seconda metà" è
   *posizionale* (le prime 20 carte a un giocatore, le ultime 20 all'altro,
   coerente con "il mazzo si divide in due metà"), la divisione corretta è:
   ```
   handA = deck[0..19]   (prime 20 posizioni)
   handB = deck[20..39]  (ultime 20 posizioni)
   ```
   **Questo è diverso** da una distribuzione alternata (carta 0 ad A, carta 1 a
   B, carta 2 ad A, ...), che avevo usato in un esempio di codice precedente per
   errore/genericità. Dato che hai descritto esplicitamente "il mazzo si divide
   in due metà", il codice finale userà la divisione posizionale in blocchi di
   20, **non** quella alternata. Questo è un punto critico da verificare: se la
   tua idea di "metà" è diversa (es. basata sul mazziere che distribuye
   alternando), va corretto di conseguenza — è un singolo cambiamento
   localizzato nel codice.

2. **Turno base.** Il giocatore "in testa" (che ha l'iniziativa) gioca la carta
   in cima alla propria mano e la mette nel mazzetto centrale (la pila di carte
   giocate, non ancora vinta da nessuno).

3. **Carta non vincente (valore 0).** Se la carta giocata è "normale"
   (non 1, 2 o 3), il turno passa semplicemente all'avversario, che ora gioca
   lui la carta in cima alla propria mano nel mazzetto — **l'iniziativa si
   sposta**, senza che nessuno vinca nulla.

4. **Carta vincente (valore 1, 2 o 3).** Se la carta giocata è vincente,
   l'avversario deve rispondere giocando immediatamente, una dopo l'altra, un
   numero di carte pari al valore (1, 2 o 3), mettendole anch'esse nel mazzetto.
   Due sotto-casi:
   - **Se durante questa risposta esce a sua volta una carta vincente** (prima
     di aver esaurito il numero dovuto), la sequenza di risposta si interrompe
     immediatamente: ora è **l'altro giocatore** (l'attaccante originale) che
     deve rispondere con il numero di carte richiesto dalla nuova carta
     vincente. Questo può incatenarsi più volte (una risposta genera un'altra
     carta vincente, che genera un'altra risposta, e così via) — modellato nel
     codice come un ciclo con "ruoli che si scambiano" (vedi §7).
   - **Se il difensore risponde con il numero dovuto di carte senza che nessuna
     sia vincente**, l'attaccante originale **vince il mazzetto**: tutte le
     carte accumulate nel mazzetto centrale vanno in fondo alla sua mano.
     L'attaccante mantiene (o riacquisisce) l'iniziativa per il turno
     successivo.

5. **Fine anticipata per esaurimento carte.** Se in un qualunque momento — non
   solo all'inizio di un turno, ma anche **a metà di una sequenza di
   risposta** — un giocatore deve giocare una carta ma la sua mano è vuota,
   **perde immediatamente la partita**: l'avversario vince tutto ciò che resta
   (mazzetto + eventuali carte residue).

6. **Ordine di reinserimento delle carte vinte.** Quando un giocatore vince il
   mazzetto, le carte vengono aggiunte **in fondo alla sua mano**, nell'ordine
   in cui sono state giocate durante quel turno/sequenza (la prima carta
   giocata nel mazzetto finisce più vicina alla "cima" del blocco appena
   aggiunto, cioè sarà la prima delle carte vinte a essere rigiocata in
   futuro). Questa è la convenzione **FIFO** (First In, First Out): il
   mazzetto si comporta come una coda, non come una pila.
   ⚠️ **Questo è il dettaglio più delicato per la correttezza matematica**: se
   la tua variante usa un ordine diverso (es. le carte vengono mescolate, o
   inserite in ordine inverso, o la carta vincente finale va in cima invece che
   in fondo), il codice di `simulate()` in §7 va adattato — è comunque un
   cambiamento localizzato a una singola funzione.

7. **Condizione di vittoria.** Il gioco termina quando uno dei due giocatori
   resta con **zero carte** in mano: l'altro ha vinto (ha "tutte le carte",
   come hai detto tu).

---

## 7. `simulate()`: la funzione che gioca una partita intera

### 7.1 Perché una coda FIFO e non un array semplice

Sia le mani dei giocatori sia il mazzetto centrale si comportano come **code**:
si pesca dalla cima (front) e si aggiunge in fondo (back). In C++/CUDA, la
struttura dati più naturale per questo è una **coda circolare** (circular
buffer): un array di dimensione fissa con due indici, `head` (dove pescare) e
`size` (quante carte contiene), che "avvolgono" (wrap-around) quando superano la
fine dell'array. Questo evita di dover spostare fisicamente gli elementi ad ogni
pescata/inserimento (che sarebbe lento, O(n) per operazione).

### 7.2 Codice completo, spiegato passo passo

```cpp
// Rappresentazione "logica" con array di byte (verrà bit-packed più avanti, §8)
struct Queue40 {
    uint8_t data[40];
    int head;   // indice della carta in cima
    int size;   // quante carte ci sono attualmente
};

// Pesca (rimuove e restituisce) la carta in cima alla coda
__device__ __forceinline__ int q_pop_front(Queue40 &q) {
    int v = q.data[q.head];
    q.head = (q.head + 1) % 40;   // avanza circolarmente
    q.size--;
    return v;
}

// Aggiunge una carta in fondo alla coda
__device__ __forceinline__ void q_push_back(Queue40 &q, int v) {
    int slot = (q.head + q.size) % 40;   // posizione libera subito dopo l'ultima carta
    q.data[slot] = v;
    q.size++;
}

__device__ int simulate(uint8_t dealA[20], uint8_t dealB[20], int max_turns) {
    // --- inizializzazione delle due mani a partire dal mazzo ricostruito ---
    Queue40 handA{}, handB{}, pile{};
    handA.head = 0; handA.size = 20;
    handB.head = 0; handB.size = 20;
    pile.head  = 0; pile.size  = 0;
    for (int i = 0; i < 20; i++) handA.data[i] = dealA[i];
    for (int i = 0; i < 20; i++) handB.data[i] = dealB[i];

    Queue40* hand[2] = { &handA, &handB };  // hand[0] = giocatore A, hand[1] = giocatore B

    int turn = 0;      // conta ogni singola carta giocata (non "round")
    int leader = 0;     // indice (0 o 1) di chi ha l'iniziativa in questo turno

    // Il gioco prosegue finché entrambi hanno carte e non abbiamo superato il tetto
    while (hand[0]->size > 0 && hand[1]->size > 0 && turn < max_turns) {

        int attacker = leader;
        int defender = 1 - leader;

        // L'attaccante gioca la prima carta del turno
        int v = q_pop_front(*hand[attacker]);
        q_push_back(pile, v);
        turn++;

        if (hand[defender]->size == 0) {
            // Il difensore non ha nemmeno una carta per rispondere: l'attaccante
            // vince tutto (mazzetto + eventuali carte residue del difensore,
            // che sono zero). La partita finisce qui.
            break;
        }

        if (v == 0) {
            // Carta non vincente: l'iniziativa passa semplicemente al difensore,
            // senza che nessuno vinca il mazzetto (che resta lì, si accumula
            // per il prossimo eventuale scontro con carta vincente).
            leader = defender;
            continue;
        }

        // --- v in {1,2,3}: il difensore deve rispondere 'v' carte ---
        // Modelliamo l'intera catena di eventuali inversioni con un unico
        // ciclo, usando 'pending' come "debito" di carte ancora dovute.
        int pending = v;
        while (pending > 0) {
            if (hand[defender]->size == 0) {
                // Il difensore finisce le carte a metà della risposta dovuta:
                // perde immediatamente. L'attaccante corrente vince la partita.
                break;
            }
            int rv = q_pop_front(*hand[defender]);
            q_push_back(pile, rv);
            turn++;
            pending--;

            if (rv != 0) {
                // La carta di risposta è a sua volta vincente: i ruoli si
                // scambiano. Chi era il difensore ora "attacca" e l'attaccante
                // originale deve rispondere 'rv' carte. Il ciclo continua con
                // i ruoli invertiti finché non si esaurisce una catena di
                // carte vincenti o qualcuno finisce le carte.
                int tmp = attacker; attacker = defender; defender = tmp;
                pending = rv;
            }
        }

        if (hand[defender]->size == 0) {
            // Uscita dal while per esaurimento carte del difensore corrente:
            // l'attaccante corrente vince la partita immediatamente.
            leader = attacker;
            break;
        }

        // Nessuna carta vincente ha interrotto la catena: il difensore ha
        // risposto correttamente 'pending' volte (fino a 0) senza carte
        // vincenti. L'attaccante (l'ultimo che ha "vinto lo scambio") si
        // prende l'intero mazzetto, in ordine FIFO (§6, punto 6).
        while (pile.size > 0) {
            int c = q_pop_front(pile);
            q_push_back(*hand[attacker], c);
        }
        leader = attacker;   // l'attaccante mantiene l'iniziativa nel turno successivo
    }

    return turn;   // numero di carte giocate in totale (usato come "durata" della partita)
}
```

### 7.3 Interpretazione del valore restituito

`simulate()` restituisce `turn`, il numero di singole carte giocate durante la
partita (non il numero di "round" o "scambi" — ogni carta giocata, sia iniziale
sia di risposta, incrementa il contatore). Se `turn >= max_turns` (5000 nel
nostro caso), consideriamo la partita "potenzialmente infinita" e la registriamo.

### 7.4 ⚠️ Punti da confermare con te prima del run completo

Il codice sopra codifica delle scelte specifiche che vanno **validate contro le
regole reali** che hai in mente, perché cambiano il risultato:

- **Divisione del mazzo**: ho usato blocchi di 20 posizionali (§6, punto 1). Se
  la tua definizione di "due metà" è diversa, va corretta la riga di split
  nel kernel (§10.2).
- **Direzione del FIFO**: ho assunto che la prima carta giocata nel mazzetto sia
  la prima ad essere reinserita (quindi la prima ad essere rigiocata in
  futuro). Verifica che corrisponda alle regole reali.
- **Chi gioca per primo nella partita** (`leader = 0` all'inizio): assunzione
  arbitraria che va confermata (es. "il giocatore che ha la prima metà del
  mazzo gioca per primo").

---

## 8. Perché conviene il bit-packing (spiegazione da zero)

### 8.1 Cos'è un registro e perché è importante

Una GPU esegue migliaia di thread contemporaneamente. Ogni thread ha a
disposizione un piccolo numero di **registri** — celle di memoria fisicamente
dentro il processore, velocissime da leggere/scrivere (accesso in 1 ciclo di
clock). Il numero di registri per thread è limitato (dipende dalla GPU, tipico
~255 registri a 32 bit per thread al massimo, ma di fatto molti meno se vuoi
tenere tanti thread attivi insieme).

Quando una variabile (es. un array) **non entra nei registri disponibili**, il
compilatore la sposta in **local memory**: nonostante il nome suggerisca
qualcosa di "vicino" e veloce, la local memory è **fisicamente nella memoria
globale della GPU** (la stessa DRAM esterna al chip usata per i grandi buffer di
dati), semplicemente con un indirizzamento privato per ogni thread. Leggere/
scrivere in local memory è **ordini di grandezza più lento** che usare registri
(centinaia di cicli di clock invece di 1).

### 8.2 Perché un array come `uint8_t data[40]` con indice variabile spilla

Il compilatore riesce a tenere un array in registri **solo se può determinare a
tempo di compilazione quali indici verranno acceduti** (es. un loop
completamente "srotolato" con indici costanti). Nel nostro caso, `q.head` cambia
ad ogni operazione, in modo dipendente dai dati della partita (quindi
imprevedibile a compile-time) — il compilatore **non ha altra scelta** che
allocare l'array in local memory.

Con array di ~40+40+40 = 120 byte per thread, moltiplicato per migliaia di
thread attivi contemporaneamente, e con accessi a indirizzi diversi da thread a
thread (quindi **non coalescenti** — vedi §8.4), il traffico verso la memoria
globale della GPU può facilmente diventare il vero collo di bottiglia
dell'intero programma, molto più grave della semplice divergenza dei warp.

### 8.3 La soluzione: comprimere lo stato in variabili scalari a 64 bit

Ogni carta ha solo 4 valori possibili (0,1,2,3), quindi basta **2 bit** per
rappresentarla (2² = 4). Una variabile `uint64_t` (intero a 64 bit) può quindi
contenere fino a 64/2 = **32 carte**. Per rappresentare una mano di 40 carte
usiamo **due** variabili `uint64_t`: `w0` per le prime 32 carte (slot 0-31) e
`w1` per le restanti 8 (slot 32-39). Usare esattamente 32 slot per parola (e non
un numero arbitrario) evita che un campo da 2 bit finisca "a cavallo" tra due
parole a 64 bit, il che complicherebbe di molto l'estrazione/inserimento.

Il punto cruciale: **operazioni come shift (`<<`, `>>`) e mask (`&`) su variabili
scalari sono istruzioni aritmetiche native della GPU**, eseguite dalla ALU
(unità aritmetico-logica) in un singolo ciclo di clock — esattamente come
un'addizione. Non c'è alcun accesso a memoria coinvolto: `w0` e `w1` sono
semplicemente due variabili che il compilatore può tenere comodamente in
registri per tutta la vita del thread, anche se il loro *contenuto* cambia ad
ogni turno di gioco.

### 8.4 Nota sulla "coalescenza" (approfondimento)

Un dettaglio in più, per completezza: quando più thread di uno stesso warp (un
gruppo di 32 thread eseguiti in lockstep sulla GPU) accedono alla memoria
globale, l'hardware è molto più efficiente se questi accessi sono "vicini" tra
loro (idealmente, indirizzi consecutivi) — si dice che l'accesso è **coalescente**.
Con array in local memory indicizzati da uno stato di gioco diverso per ogni
thread, gli accessi sono sparsi e imprevedibili: **non coalescenti**, quindi
molto più lenti del teorico. Il bit-packing elimina il problema alla radice,
perché non c'è più alcun accesso a memoria da coalescere: tutto avviene in
registri.

### 8.5 Implementazione completa

```cpp
struct Queue40 {
    uint64_t w0, w1;   // 40 carte × 2 bit = 80 bit, distribuiti su due parole a 64 bit
    int head;           // indice logico (0-39) della carta in cima
    int size;            // quante carte sono attualmente presenti
};

// Legge il valore (0-3) nello slot logico 'slot' (0-39)
__device__ __forceinline__ int q_get_slot(const Queue40 &q, int slot) {
    if (slot < 32) return (int)((q.w0 >> (slot * 2)) & 3ULL);
    else           return (int)((q.w1 >> ((slot - 32) * 2)) & 3ULL);
}

// Scrive il valore 'val' (0-3) nello slot logico 'slot'
__device__ __forceinline__ void q_set_slot(Queue40 &q, int slot, int val) {
    uint64_t mask = 3ULL, v = (uint64_t)val & 3ULL;
    if (slot < 32) {
        int sh = slot * 2;
        q.w0 = (q.w0 & ~(mask << sh)) | (v << sh);
    } else {
        int sh = (slot - 32) * 2;
        q.w1 = (q.w1 & ~(mask << sh)) | (v << sh);
    }
}

__device__ __forceinline__ int q_pop_front(Queue40 &q) {
    int v = q_get_slot(q, q.head);
    q.head++; if (q.head == 40) q.head = 0;   // wrap-around circolare
    q.size--;
    return v;
}

__device__ __forceinline__ void q_push_back(Queue40 &q, int v) {
    int slot = q.head + q.size;
    if (slot >= 40) slot -= 40;
    q_set_slot(q, slot, v);
    q.size++;
}
```

`simulate()` resta **identico** nella struttura logica descritta in §7 — cambia
solo il tipo `Queue40` e le funzioni di supporto, che ora manipolano bit invece
di array di byte. Questo è importante: **la logica di gioco e la sua
rappresentazione dati sono disaccoppiate**, quindi puoi validare prima la logica
con la versione "semplice" ad array (più facile da debuggare) e poi passare
alla versione bit-packed per le performance, confrontando che diano lo stesso
identico risultato (vedi §11).

### 8.6 Come verificare che il compilatore non spilli comunque

Dopo aver compilato, controlla sempre:

```bash
nvcc -arch=sm_86 --ptxas-options=-v -c kernel.cu -o kernel.o
```

(sostituisci `sm_86` con l'architettura della tua GPU — vedi §12.1 per come
scoprirla). L'output mostra righe tipo:

```
ptxas info    : Used 48 registers, 0 bytes spill stores, 0 bytes spill loads
```

Se vedi `spill stores`/`spill loads` **diversi da zero**, il compilatore sta
comunque usando local memory per qualcosa — in tal caso può aiutare:
- usare `__launch_bounds__(threadsPerBlock)` per guidare il compilatore verso un
  budget di registri più realistico;
- ridurre variabili temporanee non necessarie nel corpo di `simulate()`;
- verificare che non ci siano array residui non convertiti in bit-packing.

---

## 9. Rilevamento anticipato di cicli (ottimizzazione facoltativa ma importante)

### 9.1 Perché conviene

Molte partite "potenzialmente infinite" in realtà **entrano in un ciclo** molto
prima di 5000 turni: lo stato del gioco (chi ha quali carte, in che ordine,
nel mazzetto e nelle mani) si ripete esattamente identico a un certo punto, il
che garantisce che si ripeterà **per sempre** in modo identico da lì in poi
(perché il gioco è deterministico: stesso stato → stesse mosse future).

Se rileviamo un ciclo, possiamo **fermare la simulazione molto prima** di 5000
turni, con la certezza (non solo il sospetto) che la partita non finirà mai.
Questo velocizza enormemente lo screening, perché il costo medio per
simulazione crolla.

### 9.2 Come farlo: hashing dello stato + rilevamento di cicli di Brent

L'idea: ad ogni turno, calcoliamo un **hash** (un numero che riassume in modo
compatto) dell'intero stato del gioco (contenuto di `handA`, `handB`, `pile`,
più `leader`). Se lo stesso hash si ripresenta identico, quasi certamente lo
stato è tornato identico a uno precedente (con probabilità di falso positivo
trascurabile se l'hash è a 64 bit e ben distribuito), e possiamo interrompere
la simulazione dichiarando "ciclo confermato" invece di "sconosciuto dopo 5000
turni".

Con `Queue40` bit-packed, calcolare un hash dello stato è economico: basta
combinare `w0`, `w1`, `head`, `size` di tutte e tre le code, ad esempio con una
combinazione XOR-shift o una funzione hash a 64 bit standard (es. splitmix64).

**Nota implementativa**: questa parte non è ancora inclusa nel codice sopra,
perché richiede una scelta esplicita di algoritmo di rilevamento cicli
(Floyd "tartaruga e lepre" oppure Brent, entrambi O(1) di memoria aggiuntiva
per thread, adatti a GPU) e va discussa a parte quando vorrai implementarla.
Per la Fase 1 (screening puro a 5000 turni) **non è strettamente necessaria**:
è un'ottimizzazione di velocità, non di correttezza del risultato finale.

---

## 10. Il kernel CUDA completo: host + device

### 10.1 Concetti CUDA di base (per chi parte da zero)

- **Kernel**: una funzione scritta con la parola chiave `__global__`, che viene
  eseguita in parallelo da moltissimi thread sulla GPU. Si lancia dalla CPU
  (host) con la sintassi `nome_kernel<<<blocchi, thread_per_blocco>>>(argomenti)`.
- **Thread, blocco, griglia**: i thread sono raggruppati in **blocchi** (fino a
  1024 thread per blocco sulle GPU moderne); i blocchi sono a loro volta
  raggruppati in una **griglia**. Ogni thread può calcolare il proprio indice
  globale univoco come `blockIdx.x * blockDim.x + threadIdx.x`.
- **Warp**: la GPU esegue i thread in gruppi fissi di 32 (un *warp*) in modo
  sincronizzato (SIMT — *Single Instruction, Multiple Threads*): tutti i 32
  thread di un warp eseguono la stessa istruzione nello stesso momento, su dati
  diversi. Se, a causa di un `if`, alcuni thread del warp devono eseguire un
  ramo di codice diverso dagli altri, la GPU esegue **entrambi i rami in
  sequenza**, mascherando i thread non coinvolti in ciascun ramo — questo si
  chiama **divergenza di warp** ed è un costo prestazionale (tempo raddoppiato o
  peggio, a seconda di quanti rami diversi ci sono).
- **`atomicAdd`**: un'operazione che incrementa in modo sicuro un contatore
  condiviso tra migliaia di thread che potrebbero scriverci sopra
  contemporaneamente — senza `atomicAdd`, due thread che leggono e scrivono lo
  stesso contatore nello stesso istante potrebbero "perdersi" a vicenda
  (race condition), risultando in un conteggio finale sbagliato.

### 10.2 Perché un kernel "persistente" con coda dinamica di lavoro

**Il problema**: se assegni staticamente un rank fisso a ciascun thread (es.
`rank = batch_start + blockIdx.x*blockDim.x + threadIdx.x`), i thread di uno
stesso warp restano "agganciati" insieme: il warp intero **deve aspettare** che
il thread più lento finisca la propria simulazione prima di poter procedere
oltre. Dato che alcune partite finiscono in pochi turni e altre (quelle che
cerchiamo!) arrivano fino a 5000 turni, la varianza è enorme — nella peggiore
delle ipotesi, un intero warp da 32 thread rimane bloccato per 5000 turni anche
se 31 dei suoi thread avrebbero finito in 10 turni. L'efficienza reale crolla.

**La soluzione**: invece di assegnare un rank fisso per thread, lanciamo un
numero **fisso** di thread (calibrato sulla capacità della GPU, non sulla
dimensione del batch) che restano attivi in un ciclo, e ogni volta che un
thread finisce una simulazione, **va a "pescare" il prossimo rank disponibile**
da un contatore globale condiviso, incrementato atomicamente. In questo modo,
un thread che finisce presto non aspetta il warp: passa subito alla
configurazione successiva. Il carico si bilancia dinamicamente su tutta la GPU.

```cpp
// Contatore globale condiviso: "il prossimo rank (relativo al batch) da
// processare". Risiede in memoria globale della GPU, visibile a tutti i thread.
__device__ unsigned long long g_next_rank;

__global__ void persistent_search(
    uint64_t batch_start,     // rank assoluto di partenza per questo batch
    uint64_t batch_size,      // quante configurazioni contiene il batch (es. 2*10^8)
    uint64_t *hit_ranks,       // buffer di output: rank delle partite "infinite" trovate
    unsigned int *hit_count,    // contatore di quante ne sono state trovate finora
    unsigned int max_hits,       // capacità massima del buffer hit_ranks
    int max_turns)                // soglia (5000) oltre la quale consideriamo "infinita"
{
    while (true) {
        // Ogni thread chiede atomicamente "qual è il prossimo lavoro da fare?"
        uint64_t offset = atomicAdd(&g_next_rank, 1ULL);
        if (offset >= batch_size) return;   // lavoro esaurito per questo batch: il thread termina

        uint64_t rank = batch_start + offset;

        // --- ricostruzione della configurazione a partire dal rank ---
        uint8_t deck[40];
        unrank40(rank, deck);

        // --- divisione in due mani da 20 (§6, punto 1: blocchi posizionali) ---
        uint8_t handA_init[20], handB_init[20];
        for (int i = 0; i < 20; i++) handA_init[i] = deck[i];
        for (int i = 0; i < 20; i++) handB_init[i] = deck[20 + i];

        // --- simulazione della partita ---
        int turns = simulate(handA_init, handB_init, max_turns);

        // --- se la partita non è finita entro il limite, la registriamo ---
        if (turns >= max_turns) {
            unsigned int idx = atomicAdd(hit_count, 1);
            if (idx < max_hits) {
                hit_ranks[idx] = rank;
            }
            // Se idx >= max_hits, il buffer è saturo: l'host se ne accorge
            // confrontando hit_count con max_hits dopo il lancio (§10.4) e può
            // rilanciare il batch con un buffer più grande.
        }
    }
}
```

**Nota**: `deck`, `handA_init`, `handB_init` sono array a 40/20 byte con **indici
costanti a compile-time** (i loop `for (int i = 0; i < 20; i++)` sono
completamente prevedibili), quindi il compilatore riesce quasi sempre a tenerli
in registri anche senza bit-packing esplicito qui — il vero problema di
local memory era dentro `simulate()`, dove gli indici (`head`, `slot`) dipendono
dallo stato di gioco e cambiano in modo imprevedibile turno per turno (§8.2).

### 10.3 Quanti blocchi/thread lanciare?

A differenza dello schema "un thread per configurazione", qui il numero di
thread lanciati **non deve coincidere** con `batch_size` (200 milioni): lanciamo
invece un numero di thread pari a quanti la GPU può eseguire efficientemente in
parallelo (tipicamente qualche decina di migliaia, calibrato sul numero di SM —
*Streaming Multiprocessor* — della tua GPU specifica), e li lasciamo "in
loop" a pescare lavoro dalla coda finché non è esaurito.

```cpp
int threads_per_block = 256;
int num_blocks = 2048;   // valore di partenza ragionevole per GPU di fascia alta;
                          // va tarato sperimentalmente (vedi §12.3) misurando il
                          // throughput a diversi valori
```

### 10.4 Codice host: il ciclo sui batch

```cpp
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <fstream>

int main() {
    const unsigned int MAX_HITS = 1 << 16;   // 65536 slot; da tarare in base a
                                               // quanti "hit" ti aspetti per batch
    const int MAX_TURNS = 5000;
    const uint64_t BATCH_SIZE = 200000000ULL;  // 2*10^8, come nello script Python

    // --- caricamento della tabella multinomiale precalcolata in Python (§5.2) ---
    uint64_t h_table[3625];
    {
        std::ifstream fh("multinomial_table.bin", std::ios::binary);
        fh.read(reinterpret_cast<char*>(h_table), sizeof(h_table));
    }
    cudaMemcpyToSymbol(d_table, h_table, sizeof(h_table));

    // --- allocazione dei buffer sulla GPU, riutilizzati per tutti i batch ---
    uint64_t *d_hits;
    unsigned int *d_count;
    cudaMalloc(&d_hits, MAX_HITS * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(unsigned int));

    // --- lettura della lista di batch_start dal file generato dallo script Python ---
    std::vector<uint64_t> batch_starts = load_batch_starts_from_file("batches.txt");

    std::ofstream out("hits.bin", std::ios::binary);

    for (uint64_t batch_start : batch_starts) {
        // azzera contatore e "puntatore di lavoro" prima di ogni batch
        cudaMemset(d_count, 0, sizeof(unsigned int));
        unsigned long long zero = 0;
        cudaMemcpyToSymbol(g_next_rank, &zero, sizeof(zero));

        persistent_search<<<2048, 256>>>(
            batch_start, BATCH_SIZE, d_hits, d_count, MAX_HITS, MAX_TURNS
        );
        cudaDeviceSynchronize();   // attende che il kernel finisca prima di leggere i risultati

        unsigned int h_count;
        cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (h_count > MAX_HITS) {
            fprintf(stderr,
                "ATTENZIONE: buffer saturo per batch_start=%llu (trovati %u, capacita' %u).\n"
                "Alcuni hit sono andati persi. Aumenta MAX_HITS e rilancia questo batch.\n",
                (unsigned long long)batch_start, h_count, MAX_HITS);
            h_count = MAX_HITS;   // leggiamo solo quelli effettivamente scritti
        }

        std::vector<uint64_t> h_hits(h_count);
        cudaMemcpy(h_hits.data(), d_hits, h_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        out.write(reinterpret_cast<char*>(h_hits.data()), h_count * sizeof(uint64_t));

        printf("batch_start=%llu: %u hit trovati\n",
               (unsigned long long)batch_start, h_count);
    }

    cudaFree(d_hits);
    cudaFree(d_count);
    return 0;
}
```

**Cosa viene salvato su disco**: solo il `rank` (8 byte) di ogni configurazione
"potenzialmente infinita" trovata — **non** le 40 carte per esteso. Questo è
sufficiente, perché puoi sempre ricostruire la configurazione completa in un
secondo momento (in Python o in C++) applicando di nuovo `unrank_multiset(rank,
...)` sui soli rank salvati, che saranno una piccola frazione del totale.

---

## 11. Validazione: perché è indispensabile e come farla

**Prima di lanciare il run completo su 1.9·10¹⁴ configurazioni**, è essenziale
verificare che il codice CUDA (unranking + simulazione) produca **esattamente**
gli stessi risultati di una versione di riferimento in Python, su un caso
gestibile a mano.

### 11.1 Perché non fidarsi "a occhio"

Bug sottili nell'unranking (es. un ordine dei simboli invertito, un off-by-one
nell'indice del bit-packing) o nella logica di `simulate()` (es. una condizione
di uscita sbagliata) possono produrre risultati **plausibili ma sbagliati** —
partite che sembrano finire correttamente ma con un conteggio di turni diverso
da quello reale, oppure con l'esito (chi vince) invertito. Con miliardi di
simulazioni, questi errori non sono visibili "a campione" senza un confronto
sistematico.

### 11.2 Procedura di validazione consigliata

1. **Scegli un mazzo ridotto**, ad esempio 8 carte invece di 40 (es. 4 zeri,
   2 "1", 1 "2", 1 "3" — o qualunque composizione che mantenga proporzioni
   simili), per cui il numero totale di configurazioni distinte è piccolo
   (poche migliaia), enumerabile per intero in pochi secondi.
2. **Scrivi una versione Python di riferimento** di `simulate()`, che implementi
   *esattamente* la stessa logica descritta in §6-7, ma in modo diretto e
   facile da ispezionare (senza preoccuparti di performance).
3. **Per ogni rank da 0 a totale-1** (dello spazio ridotto a 8 carte):
   - genera la configurazione con `unrank_multiset` (Python) e con `unrank40`
     (CUDA, compilato per girare anche su un mazzo di dimensione ridotta —
     puoi parametrizzare `unrank40` per accettare la composizione come
     argomento invece di averla fissa a 28/4/4/4, solo per questa fase di
     test);
   - confronta che le due sequenze di carte generate siano **identiche**,
     carta per carta;
   - simula la partita con entrambe le versioni (Python e CUDA) e confronta
     che il numero di turni restituito sia **identico**.
4. Se **tutti** i confronti coincidono (idealmente migliaia di casi, coprendo
   quindi anche configurazioni "rare" come quelle con partite molto lunghe),
   puoi fidarti che la logica sia corretta, e passare al mazzo completo da 40
   carte.
5. Ripeti un confronto simile (magari su un batch più piccolo, es. 10⁶ invece
   di 2·10⁸) anche dopo aver introdotto il bit-packing (§8), per verificare che
   la conversione da array semplice a rappresentazione bit-packed non abbia
   introdotto errori.

### 11.3 Cosa fare se i risultati non coincidono

Se emergono discrepanze, isola il problema restringendo ulteriormente il caso
di test (es. mazzo di 4-5 carte, dove puoi letteralmente seguire la partita a
mano su carta) finché non trovi il rank minimo che produce un comportamento
diverso tra Python e CUDA — a quel punto il debug diventa gestibile.

---

## 12. Preparazione dell'ambiente e compilazione

### 12.1 Verificare la GPU e l'architettura CUDA

```bash
nvidia-smi          # mostra modello GPU, driver, memoria disponibile
nvcc --version       # verifica versione del compilatore CUDA (nvcc)
```

Ogni GPU NVIDIA ha una "compute capability" (es. 8.6 per una RTX 3090, 8.9 per
una RTX 4090) che va specificata al compilatore con il flag `-arch=sm_XX` (es.
`-arch=sm_86`, `-arch=sm_89`). Puoi verificare il valore esatto per la tua
scheda cercando "compute capability" + il nome del modello, oppure con:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 12.2 Compilazione

Assumendo un unico file `kernel.cu` contenente sia il codice host (`main`) sia
il codice device (i `__global__`/`__device__`):

```bash
nvcc -O3 -arch=sm_86 --ptxas-options=-v kernel.cu -o straccia_camicia
```

- `-O3`: massimo livello di ottimizzazione del compilatore host (per il codice
  C++ lato CPU).
- `-arch=sm_86`: genera codice specifico per la compute capability della tua
  GPU (sostituisci con il valore corretto).
- `--ptxas-options=-v`: stampa informazioni sull'uso di registri e local memory
  per ogni kernel (§8.6), fondamentale per verificare che non ci sia spilling.

### 12.3 Taratura sperimentale dei parametri di lancio

I valori `num_blocks = 2048` e `threads_per_block = 256` di §10.3 sono un punto
di partenza ragionevole, ma **vanno tarati sulla GPU specifica** che userai:

1. Esegui un batch di test (es. 10⁷ configurazioni invece di 2·10⁸, per
   iterare velocemente) con diverse combinazioni di `num_blocks` e
   `threads_per_block`.
2. Misura il tempo di esecuzione con `cudaEventRecord`/`cudaEventElapsedTime`
   (o anche solo `std::chrono` intorno alla chiamata a `cudaDeviceSynchronize()`).
3. Scegli la combinazione che massimizza il throughput (configurazioni
   simulate al secondo).

Come riferimento generale: `threads_per_block` è quasi sempre un multiplo di 32
(la dimensione del warp) — 128, 256 o 512 sono scelte tipiche. `num_blocks`
conviene sia un multiplo del numero di SM della GPU (visibile con
`cudaGetDeviceProperties`), per garantire che tutti gli SM restino occupati.

---

## 13. Riepilogo del flusso di lavoro end-to-end

1. **Python (una tantum)**: genera `multinomial_table.bin` (§5.2) e la lista di
   `batch_starts` (il tuo script attuale, eventualmente esteso per salvare i
   rank in un formato binario invece che testuale, più comodo da leggere in
   C++).
2. **Validazione (una tantum, prima del run completo)**: confronto Python vs
   CUDA su un mazzo ridotto (§11), sia per la versione "semplice" sia per
   quella bit-packed.
3. **Compilazione** del programma CUDA (§12.2), con verifica dell'assenza di
   spilling (§8.6).
4. **Taratura** dei parametri di lancio su un batch piccolo (§12.3).
5. **Run completo**: il programma host (§10.4) itera su tutti i ~967.922 batch,
   per ciascuno lancia il kernel persistente (§10.2), e accumula su
   `hits.bin` i rank di tutte le configurazioni "potenzialmente infinite"
   trovate (partite che superano 5000 turni).
6. **Fase 2 (successiva, non coperta in questo documento)**: analisi più
   approfondita dei soli rank salvati in `hits.bin` — ad esempio con
   rilevamento di cicli esatto (§9) per confermare quali sono davvero cicli
   infiniti e quali erano solo partite molto lunghe ma finite.

---

## 14. Domande aperte da chiarire prima del run completo

Per evitare di scoprire un bug sistemico dopo settimane di calcolo, conviene
fissare esplicitamente questi punti prima di lanciare il run completo (già
segnalati anche nei paragrafi sopra, riassunti qui per comodità):

1. **Divisione del mazzo in due metà** (§6.1): posizionale a blocchi di 20
   (`deck[0:20]` / `deck[20:40]`) — confermi che è questa la regola, o è
   un'altra convenzione (es. basata su chi distribuisce alternando)?
2. **Ordine FIFO di reinserimento del mazzetto vinto** (§6.6): confermi che la
   prima carta giocata nel turno/sequenza sia la prima a essere reinserita
   (e quindi la prima a essere ri-pescata in futuro)?
3. **Chi gioca per primo** all'inizio della partita: assunto `leader = 0`
   (il giocatore della prima metà del mazzo) — è corretto?
4. **Dimensione del buffer `MAX_HITS`**: non abbiamo ancora una stima di quante
   configurazioni "potenzialmente infinite" ci si aspetta per batch da 2·10⁸.
   Conviene partire con un valore prudente (es. 65536) e affidarsi al controllo
   `h_count > MAX_HITS` per capire se va aumentato.

Una volta confermati questi punti, il codice descritto in questo README è
pronto per essere assemblato in un singolo file `kernel.cu` e validato secondo
la procedura di §11.
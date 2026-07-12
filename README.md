# Stracciacamicia infinita

Programmi in C scritti con l'obiettivo di trovare configurazioni iniziali dei mazzi per cui si ha una partita infinita, ovvero si ottiene una configurazione già vista in precedenza nella stessa partita (ciclo).

## Introduzione
Il gioco è la variante italiana del più noto all'estero "Beggar My Neighbour", giocato con carte da ramino. Anche per questo gioco si presenta la stessa potenziale sfida, che è stata già affrontata da molti, individuando alcune configurazioni che portano a partite infinite.

## Configurazioni possibili
Sfruttando la formula delle permutazioni con 40 carte, 12 carte vincenti e 28 carte generiche.
40!/(4!4!4!28!)
Esprimiamo le configurazioni come una sequenza di 40 numeri da una cifra, che poi verranno suddivise nei due mazzetti iniziali da 20 carte per i due giocatori.

### Configurazioni non banali
Configurazione banale: tutte le carte vincenti sono nel mazzetto di un giocatore o dell'altro
2*(20!)/(4!4!4!8!)

Il totale delle configurazioni non banali sarà la differenza tra il numero precedente e questo, anche nell'ottica globale non è che ci sia chissà quale risparmio di tempo e spazio (le configurazioni banali sono 5 ordini di grandezza in meno rispetto al totale).
Tuttavia, dato che alcune di esse si trovano all'inizio e alla fine delle permutazioni, tanto vale escluderle fin da subito (lasciando comunque quelle presenti in mezzo alle altre, si fa prima a simularle che a escluderle, sarebbe,come si suol dire, più la salsa che l'arrosto).

### Generazione delle permutazioni
Possiamo considerare il mazzo come una sequenza ordinata, dove la prima configurazione è
0000000000000000000000000000111122223333
Mentre l'ultima è 
3333222211110000000000000000000000000000

### Configurazioni di partenza e di arrivo
Alcune partite terminano sicuramente, per vale la pena escluderle. In particolare, sono facilmente identificabili le configurazioni dove non sono presenti carte vincenti in una delle due metà del mazzo.

Prima configurazione non banale
0000000000000000000033332222111100000000 (banale)
diventa
0000000000000000000100000000011122223333 (non banale)

Ultima configurazione non banale
3333222211100000000010000000000000000000 (non banale)
diventa
3333222211100000000100000000000000000000 (banale)

## Programmazione

## Architettura del programma

### Generazione batch di configurazioni
Creazione di un file contenente un certo numero di configurazioni da testare, generate in ordine.

Dimensione del batch?
Circa 2*10^14 configurazioni, quindi scegliendo di creare 2*10^6 batch, si avrebbero 2*10^8 configurazioni per batch, e supponendo di usare 1 Byte a carattere, e che ciascuna configurazione sia composta da 40 caratteri, ogni file peserebbe circa 4GB.
Questo permetterebbe al file di essere scaricato "agevolmente" e caricato in ram da quasi tutti i dispositivi (pc, smartphone, raspberry, ecc), rendendo percorribile un approccio distribuito al bruteforce.

Velocità di generazione delle configurazioni su i7-6700K circa 6,5*10^5 configurazioni/secondo.
Supponendo di generare configurazioni a pieno regime si avrebbero 6,5*3600*24*10^5 = 56160000000 = 5,61*10^10 configurazioni/giorno.
Su hardware migliore è lecito supporre di avere velocità sensibilmente migliori.

### Caricamento delle configurazioni e simulazione
In parallelo usando pthread
Focus sull'efficienza (array statici, uso più memoria ma più veloci)

### Velocità media di simulazione
Numero medio di partite al secondo

## Utilizzo dei programmi

### Generazione dei batch

### Simulazione
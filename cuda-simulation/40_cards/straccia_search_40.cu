// straccia_search_40.cu
//
// PROGRAMMA DI PRODUZIONE: ricerca esaustiva di partite "potenzialmente
// infinite" (>= max_turns turni) su TUTTO lo spazio delle configurazioni del
// mazzo reale da 40 carte (composizione 28:4:4:4, ~1,9358 x 10^14
// configurazioni distinte).
//
// Usa:
//   - unranking + simulate in versione BIT-PACKED (README §8), validati
//     esaustivamente contro la versione ad array su mazzi da 8, 10 e 20
//     carte (vedi compare_bitpacked_hits.py)
//   - kernel persistente con coda di lavoro atomica (README §10.2), un batch
//     alla volta (batch_size default 2*10^8, come nello script Python
//     originale di generazione dei batch)
//   - CHECKPOINT su file: dopo che un batch e' stato completato con successo
//     E i suoi hit sono stati scritti su disco, il programma salva il
//     prossimo batch_start da processare. Se il programma viene interrotto
//     (es. disconnessione di sessione Colab) PRIMA che un batch sia stato
//     completato, al riavvio quel batch viene semplicemente rielaborato da
//     capo: nessuna perdita di dati, nessun hit duplicato.
//   - BUDGET DI TEMPO: il programma si ferma da solo dopo circa
//     time_budget_seconds, salva il checkpoint, e termina puliito. Rilancia
//     lo stesso comando per riprendere esattamente da dove si era fermato.
//
// Uso:
//   nvcc -O3 -arch=sm_XX straccia_search_40.cu -o straccia_search_40
//   ./straccia_search_40 multinomial_table.bin \
//       [batch_size=200000000] [max_turns=5000] [max_hits_per_batch=100000] \
//       [time_budget_seconds=36000] [checkpoint_file=checkpoint.txt] \
//       [hits_file=hits_40.bin]
//
// IMPORTANTE su Colab: checkpoint_file e hits_file vanno puntati a un
// percorso su Google Drive montato (es. /content/drive/MyDrive/straccia/...),
// altrimenti si perdono quando la sessione Colab si resetta.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>

#define DIM0 29
#define DIM1 5
#define DIM2 5
#define DIM3 5
#define TABLE_SIZE (DIM0 * DIM1 * DIM2 * DIM3)
#define MAX_DECK 40
#define HALF_SIZE 20

__constant__ uint64_t d_table[TABLE_SIZE];
__constant__ int d_counts0[4] = {28, 4, 4, 4};
__constant__ int d_max_turns;

__device__ unsigned long long g_next_rank;
__device__ unsigned int g_hit_count;

__device__ __forceinline__ uint64_t tbl(int c0, int c1, int c2, int c3) {
    int idx = ((c0 * DIM1 + c1) * DIM2 + c2) * DIM3 + c3;
    return d_table[idx];
}

// Unranking (README §5.5): dato un rank, ricostruisce la sequenza di 40
// simboli corrispondente, usando la tabella multinomiale precalcolata.
__device__ void unrank40(uint64_t rank, uint8_t out[40]) {
    int c[4] = {d_counts0[0], d_counts0[1], d_counts0[2], d_counts0[3]};
    for (int pos = 0; pos < 40; pos++) {
        for (int sym = 0; sym < 4; sym++) {
            if (c[sym] == 0) continue;
            c[sym]--;
            uint64_t perms = tbl(c[0], c[1], c[2], c[3]);
            if (rank < perms) { out[pos] = (uint8_t)sym; break; }
            rank -= perms;
            c[sym]++;
        }
    }
}

// --- Coda bit-packed a 40 slot (README §8.5) ---
// Ogni carta occupa 2 bit; w0 copre gli slot 0-31, w1 gli slot 32-39. Shift e
// mask su variabili scalari restano in registri per l'intera vita del
// thread, evitando gli accessi a local memory che si avrebbero con un array
// indicizzato dinamicamente (README §8.2).
struct Queue40 { uint64_t w0, w1; int head; int size; };

__device__ __forceinline__ int q_get_slot(const Queue40 &q, int slot) {
    if (slot < 32) return (int)((q.w0 >> (slot * 2)) & 3ULL);
    else           return (int)((q.w1 >> ((slot - 32) * 2)) & 3ULL);
}
__device__ __forceinline__ void q_set_slot(Queue40 &q, int slot, int val) {
    uint64_t mask = 3ULL, v = (uint64_t)val & 3ULL;
    if (slot < 32) { int sh = slot * 2;        q.w0 = (q.w0 & ~(mask << sh)) | (v << sh); }
    else           { int sh = (slot - 32) * 2; q.w1 = (q.w1 & ~(mask << sh)) | (v << sh); }
}
__device__ __forceinline__ int q_pop_front(Queue40 &q) {
    int v = q_get_slot(q, q.head);
    q.head++; if (q.head == 40) q.head = 0;
    q.size--;
    return v;
}
__device__ __forceinline__ void q_push_back(Queue40 &q, int v) {
    int slot = q.head + q.size;
    if (slot >= 40) slot -= 40;
    q_set_slot(q, slot, v);
    q.size++;
}

// Logica di gioco (README §7, regole confermate: divisione a blocchi
// posizionali, reinserimento del mazzetto FIFO, leader iniziale = giocatore
// della prima meta' del mazzo).
__device__ int simulate(const uint8_t deck[40], int max_turns) {
    Queue40 handA{0, 0, 0, HALF_SIZE}, handB{0, 0, 0, HALF_SIZE}, pile{0, 0, 0, 0};
    for (int i = 0; i < HALF_SIZE; i++) {
        q_set_slot(handA, i, deck[i]);
        q_set_slot(handB, i, deck[HALF_SIZE + i]);
    }

    Queue40* hand[2] = {&handA, &handB};
    int leader = 0, turn = 0;

    while (hand[0]->size > 0 && hand[1]->size > 0 && turn < max_turns) {
        int attacker = leader, defender = 1 - leader;
        int v = q_pop_front(*hand[attacker]);
        q_push_back(pile, v);
        turn++;
        if (hand[defender]->size == 0) break;
        if (v == 0) { leader = defender; continue; }

        int pending = v;
        while (pending > 0) {
            if (hand[defender]->size == 0) break;
            int rv = q_pop_front(*hand[defender]);
            q_push_back(pile, rv);
            turn++; pending--;
            if (rv != 0) { int t = attacker; attacker = defender; defender = t; pending = rv; }
        }
        if (hand[defender]->size == 0) { leader = attacker; break; }

        while (pile.size > 0) q_push_back(*hand[attacker], q_pop_front(pile));
        leader = attacker;
    }
    return turn;
}

__global__ void init_globals() {
    g_next_rank = 0;
    g_hit_count = 0;
}

// Kernel persistente (README §10.2): ogni thread resta in loop pescando il
// prossimo rank da processare da un contatore atomico condiviso, invece di
// ricevere un rank fisso -- bilancia il carico anche se alcune partite durano
// molto piu' a lungo di altre.
__global__ void search_kernel(uint64_t batch_start, uint64_t batch_size,
                               uint64_t *hit_ranks, uint32_t *hit_turns,
                               unsigned int max_hits) {
    while (true) {
        uint64_t offset = atomicAdd(&g_next_rank, 1ULL);
        if (offset >= batch_size) return;

        uint64_t rank = batch_start + offset;
        uint8_t deck[MAX_DECK];
        unrank40(rank, deck);

        int turns = simulate(deck, d_max_turns);

        if (turns >= d_max_turns) {
            unsigned int idx = atomicAdd(&g_hit_count, 1);
            if (idx < max_hits) {
                hit_ranks[idx] = rank;
                hit_turns[idx] = (uint32_t)turns;
            }
        }
    }
}

static std::vector<char> read_file(const char *path) {
    std::ifstream fh(path, std::ios::binary | std::ios::ate);
    if (!fh) { fprintf(stderr, "Impossibile aprire %s\n", path); exit(1); }
    size_t size = (size_t)fh.tellg(); fh.seekg(0);
    std::vector<char> buf(size); fh.read(buf.data(), size);
    return buf;
}

#define CUDA_CHECK(call) do { cudaError_t e__ = (call); if (e__ != cudaSuccess) { \
    fprintf(stderr, "Errore CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); exit(1); } } while (0)

static uint64_t read_checkpoint(const char *path) {
    std::ifstream fh(path);
    if (!fh) return 0;
    uint64_t v = 0;
    fh >> v;
    return v;
}

static void write_checkpoint(const char *path, uint64_t next_batch_start) {
    std::ofstream fh(path, std::ios::trunc);
    fh << next_batch_start << "\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Uso: %s multinomial_table.bin [batch_size=200000000] [max_turns=5000] "
            "[max_hits_per_batch=100000] [time_budget_seconds=36000] "
            "[checkpoint_file=checkpoint.txt] [hits_file=hits_40.bin]\n", argv[0]);
        return 1;
    }
    const char *table_path = argv[1];
    uint64_t batch_size = argc > 2 ? strtoull(argv[2], nullptr, 10) : 200000000ULL;
    int max_turns = argc > 3 ? atoi(argv[3]) : 5000;
    unsigned int max_hits = argc > 4 ? (unsigned int)atoll(argv[4]) : 100000;
    long time_budget_seconds = argc > 5 ? atol(argv[5]) : 36000;
    const char *checkpoint_path = argc > 6 ? argv[6] : "checkpoint.txt";
    const char *hits_path = argc > 7 ? argv[7] : "hits_40.bin";

    auto table_buf = read_file(table_path);
    if (table_buf.size() != TABLE_SIZE * sizeof(uint64_t)) {
        fprintf(stderr, "Dimensione tabella inattesa: attesi %zu byte, trovati %zu. "
                        "Stai usando multinomial_table.bin per la composizione 28/4/4/4?\n",
                TABLE_SIZE * sizeof(uint64_t), table_buf.size());
        return 1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_table, table_buf.data(), table_buf.size()));
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_turns, &max_turns, sizeof(int)));

    const uint64_t *table_data = reinterpret_cast<const uint64_t *>(table_buf.data());
    int idx0 = ((28 * DIM1 + 4) * DIM2 + 4) * DIM3 + 4;
    uint64_t total = table_data[idx0];

    uint64_t resume_start = read_checkpoint(checkpoint_path);

    printf("=== Straccia Camicia: ricerca esaustiva mazzo da 40 carte ===\n");
    printf("Totale configurazioni: %llu\n", (unsigned long long)total);
    printf("Batch size: %llu  (~%llu batch totali)\n",
           (unsigned long long)batch_size,
           (unsigned long long)((total + batch_size - 1) / batch_size));
    printf("max_turns=%d  max_hits_per_batch=%u  time_budget=%lds\n",
           max_turns, max_hits, time_budget_seconds);
    printf("Checkpoint: %s (ripresa da batch_start=%llu)\n\n",
           checkpoint_path, (unsigned long long)resume_start);

    if (resume_start >= total) {
        printf("Il checkpoint indica che la ricerca e' gia' completa (resume_start >= totale).\n");
        return 0;
    }

    uint64_t *d_hit_ranks; uint32_t *d_hit_turns;
    CUDA_CHECK(cudaMalloc(&d_hit_ranks, (size_t)max_hits * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hit_turns, (size_t)max_hits * sizeof(uint32_t)));

    std::vector<uint64_t> h_ranks(max_hits);
    std::vector<uint32_t> h_turns(max_hits);

    std::ofstream out(hits_path, std::ios::binary | std::ios::app);
    if (!out) { fprintf(stderr, "Impossibile aprire %s in scrittura\n", hits_path); return 1; }

    auto t_run_start = std::chrono::steady_clock::now();
    uint64_t configs_done_this_run = 0;
    unsigned long long total_hits_this_run = 0;

    int blocks = 2048, threads = 256;  // punto di partenza, tarabile (README §12.3)

    for (uint64_t batch_start = resume_start; batch_start < total; batch_start += batch_size) {
        uint64_t this_batch_size = std::min(batch_size, total - batch_start);

        init_globals<<<1, 1>>>();
        CUDA_CHECK(cudaGetLastError());

        cudaEvent_t bstart, bstop;
        cudaEventCreate(&bstart); cudaEventCreate(&bstop);
        cudaEventRecord(bstart);

        search_kernel<<<blocks, threads>>>(batch_start, this_batch_size, d_hit_ranks, d_hit_turns, max_hits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(bstop); cudaEventSynchronize(bstop);
        float ms = 0; cudaEventElapsedTime(&ms, bstart, bstop);
        cudaEventDestroy(bstart); cudaEventDestroy(bstop);

        unsigned int h_hit_count;
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_hit_count, g_hit_count, sizeof(unsigned int)));

        if (h_hit_count > max_hits) {
            fprintf(stderr,
                "ATTENZIONE: buffer saturo per batch_start=%llu (%u trovati, capacita' %u). "
                "Alcuni hit persi in questo batch: aumenta max_hits_per_batch e rilancia "
                "SOLO questo batch a parte, se vuoi recuperarli tutti.\n",
                (unsigned long long)batch_start, h_hit_count, max_hits);
        }
        unsigned int n_to_copy = h_hit_count < max_hits ? h_hit_count : max_hits;

        if (n_to_copy > 0) {
            CUDA_CHECK(cudaMemcpy(h_ranks.data(), d_hit_ranks, n_to_copy * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_turns.data(), d_hit_turns, n_to_copy * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (unsigned int i = 0; i < n_to_copy; i++) {
                out.write(reinterpret_cast<char *>(&h_ranks[i]), sizeof(uint64_t));
                out.write(reinterpret_cast<char *>(&h_turns[i]), sizeof(uint32_t));
            }
            out.flush();
        }

        // Il checkpoint viene aggiornato SOLO ORA, dopo che il batch e' stato
        // completato con successo e i suoi hit sono gia' su disco. Se il
        // programma si interrompesse prima di questo punto (es. crash,
        // disconnessione Colab), al riavvio questo stesso batch verrebbe
        // semplicemente rielaborato da capo: nessuna perdita, nessun
        // duplicato (i suoi hit non sono mai stati scritti prima di questo
        // punto in nessuna esecuzione parziale).
        uint64_t next_batch_start = batch_start + this_batch_size;
        write_checkpoint(checkpoint_path, next_batch_start);

        configs_done_this_run += this_batch_size;
        total_hits_this_run += n_to_copy;

        double throughput_m = (this_batch_size / 1e6) / (ms / 1000.0);
        double pct_total = 100.0 * next_batch_start / (double)total;

        printf("batch_start=%llu  size=%llu  tempo=%.2fs  throughput=%.1fM/s  hit=%u  progresso=%.6f%%\n",
               (unsigned long long)batch_start, (unsigned long long)this_batch_size,
               ms / 1000.0, throughput_m, n_to_copy, pct_total);

        long elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t_run_start).count();
        if (elapsed >= time_budget_seconds) {
            printf("\nBudget di tempo (%lds) raggiunto. Checkpoint salvato a batch_start=%llu.\n",
                   time_budget_seconds, (unsigned long long)next_batch_start);
            printf("Rilancia lo stesso comando per riprendere da qui.\n");
            break;
        }
    }

    double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t_run_start).count();
    printf("\n=== Riepilogo di questa sessione ===\n");
    printf("Configurazioni elaborate: %llu\n", (unsigned long long)configs_done_this_run);
    printf("Hit trovati in questa sessione: %llu\n", total_hits_this_run);
    printf("Tempo totale: %.1f secondi\n", elapsed_s);
    if (elapsed_s > 0)
        printf("Throughput medio: %.1f milioni di configurazioni/secondo\n",
               (configs_done_this_run / 1e6) / elapsed_s);

    cudaFree(d_hit_ranks);
    cudaFree(d_hit_turns);
    return 0;
}

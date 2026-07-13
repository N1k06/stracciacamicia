// search_full_20_bitpacked.cu
//
// Identico a search_full_20.cu (stessa composizione 14:2:2:2, stessa logica
// di gioco, stesso criterio di hit), ma con la rappresentazione delle mani e
// del mazzetto convertita a BIT-PACKED (§8 del README): ogni carta occupa 2
// bit invece di un byte intero, impacchettati in due interi a 64 bit (w0, w1)
// invece che in un array indicizzato dinamicamente. Questo evita che il
// compilatore metta le code in local memory (§8.2-8.3 del README).
//
// Serve a validare che questa conversione non introduca bug PRIMA di usarla
// nel programma di produzione a 40 carte: deve trovare esattamente gli stessi
// hit (stesso rank, stessi turni) gia' ottenuti con la versione ad array.
//
// Uso:
//   nvcc -O3 -arch=sm_XX search_full_20_bitpacked.cu -o search_full_20_bitpacked
//   ./search_full_20_bitpacked table_20.bin

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

#define MAX_TABLE 4096
#define N_BUCKETS 6

__constant__ uint64_t d_table[MAX_TABLE];
__constant__ int d_counts0[4];
__constant__ int d_dims[4];
__constant__ int d_deck_size;
__constant__ int d_half_size;
__constant__ int d_max_turns;
__constant__ int d_hit_threshold;

__device__ unsigned long long g_next_rank;
__device__ unsigned int g_hit_count;
__device__ unsigned int g_max_turns_found;
__device__ unsigned long long g_histogram[N_BUCKETS];

__device__ __forceinline__ uint64_t tbl(int c0, int c1, int c2, int c3) {
    int idx = ((c0 * d_dims[1] + c1) * d_dims[2] + c2) * d_dims[3] + c3;
    return d_table[idx];
}

__device__ void unrank_deck(uint64_t rank, uint8_t *out) {
    int c[4] = {d_counts0[0], d_counts0[1], d_counts0[2], d_counts0[3]};
    for (int pos = 0; pos < d_deck_size; pos++) {
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

// --- Coda bit-packed a 40 slot (capacita' fissa, compatibile col mazzo reale
// da 40 carte anche se qui il mazzo di test ne usa solo 20). Ogni carta
// occupa 2 bit; w0 copre gli slot 0-31, w1 gli slot 32-39. (README §8.5) ---
struct Queue40 { uint64_t w0, w1; int head; int size; };

__device__ __forceinline__ int q_get_slot(const Queue40 &q, int slot) {
    if (slot < 32) return (int)((q.w0 >> (slot * 2)) & 3ULL);
    else           return (int)((q.w1 >> ((slot - 32) * 2)) & 3ULL);
}
__device__ __forceinline__ void q_set_slot(Queue40 &q, int slot, int val) {
    uint64_t mask = 3ULL, v = (uint64_t)val & 3ULL;
    if (slot < 32) { int sh = slot * 2;      q.w0 = (q.w0 & ~(mask << sh)) | (v << sh); }
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

__device__ int simulate(const uint8_t *deal_a, const uint8_t *deal_b, int half, int max_turns) {
    Queue40 handA{0, 0, 0, half}, handB{0, 0, 0, half}, pile{0, 0, 0, 0};
    for (int i = 0; i < half; i++) {
        q_set_slot(handA, i, deal_a[i]);
        q_set_slot(handB, i, deal_b[i]);
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

__device__ __forceinline__ int bucket_for(int turns, int max_turns) {
    if (turns >= max_turns) return 5;
    if (turns >= 1000) return 4;
    if (turns >= 200) return 3;
    if (turns >= 50) return 2;
    if (turns >= 10) return 1;
    return 0;
}

__global__ void init_globals() {
    g_next_rank = 0;
    g_hit_count = 0;
    g_max_turns_found = 0;
    for (int i = 0; i < N_BUCKETS; i++) g_histogram[i] = 0;
}

__global__ void search_kernel(uint64_t total, uint64_t *hit_ranks, uint32_t *hit_turns, unsigned int max_hits) {
    while (true) {
        uint64_t offset = atomicAdd(&g_next_rank, 1ULL);
        if (offset >= total) return;

        uint8_t deck[40];
        unrank_deck(offset, deck);
        int turns = simulate(deck, deck + d_half_size, d_half_size, d_max_turns);

        atomicMax(&g_max_turns_found, (unsigned int)turns);
        atomicAdd(&g_histogram[bucket_for(turns, d_max_turns)], 1ULL);

        if (turns >= d_hit_threshold) {
            unsigned int idx = atomicAdd(&g_hit_count, 1);
            if (idx < max_hits) {
                hit_ranks[idx] = offset;
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s table_20.bin [max_turns=5000] [hit_threshold=5000] [max_hits=100000] [out=hits_20_bitpacked.bin]\n", argv[0]);
        return 1;
    }
    const char *table_path = argv[1];
    int max_turns = argc > 2 ? atoi(argv[2]) : 5000;
    int hit_threshold = argc > 3 ? atoi(argv[3]) : 5000;
    unsigned int max_hits = argc > 4 ? (unsigned int)atoll(argv[4]) : 100000;
    const char *out_path = argc > 5 ? argv[5] : "hits_20_bitpacked.bin";

    auto table_buf = read_file(table_path);
    int32_t counts[4];
    memcpy(counts, table_buf.data(), sizeof(counts));
    int dims[4] = {counts[0] + 1, counts[1] + 1, counts[2] + 1, counts[3] + 1};
    int table_size = dims[0] * dims[1] * dims[2] * dims[3];
    if (table_size > MAX_TABLE) { fprintf(stderr, "Tabella troppo grande.\n"); return 1; }
    const uint64_t *table_data = reinterpret_cast<const uint64_t *>(table_buf.data() + sizeof(counts));

    int deck_size = counts[0] + counts[1] + counts[2] + counts[3];
    int half_size = deck_size / 2;

    int idx0 = ((counts[0] * dims[1] + counts[1]) * dims[2] + counts[2]) * dims[3] + counts[3];
    uint64_t total = table_data[idx0];

    printf("[VERSIONE BIT-PACKED] Composizione: [%d,%d,%d,%d]  carte=%d  meta'=%d\n",
           counts[0], counts[1], counts[2], counts[3], deck_size, half_size);
    printf("Totale configurazioni da esaminare: %llu\n", (unsigned long long)total);
    printf("max_turns=%d  hit_threshold=%d  max_hits=%u\n\n", max_turns, hit_threshold, max_hits);

    CUDA_CHECK(cudaMemcpyToSymbol(d_table, table_data, table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_counts0, counts, sizeof(counts)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dims, dims, sizeof(dims)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_deck_size, &deck_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_half_size, &half_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_turns, &max_turns, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hit_threshold, &hit_threshold, sizeof(int)));

    uint64_t *d_hit_ranks; uint32_t *d_hit_turns;
    CUDA_CHECK(cudaMalloc(&d_hit_ranks, (size_t)max_hits * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_hit_turns, (size_t)max_hits * sizeof(uint32_t)));

    init_globals<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = 2048, threads = 256;
    search_kernel<<<blocks, threads>>>(total, d_hit_ranks, d_hit_turns, max_hits);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    unsigned int h_hit_count, h_max_turns_found;
    unsigned long long h_histogram[N_BUCKETS];
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_hit_count, g_hit_count, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_max_turns_found, g_max_turns_found, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_histogram, g_histogram, sizeof(h_histogram)));

    printf("=== RISULTATO RICERCA (BIT-PACKED) ===\n");
    printf("Tempo GPU: %.3f secondi\n", ms / 1000.0);
    printf("Throughput: %.2f milioni di configurazioni/secondo\n", (total / 1e6) / (ms / 1000.0));
    printf("Turni massimi trovati: %u\n", h_max_turns_found);
    printf("Configurazioni con turni >= %d (candidate): %u\n\n", hit_threshold, h_hit_count);
    printf("Distribuzione turni:\n");
    printf("  <10:            %llu\n", h_histogram[0]);
    printf("  10-49:          %llu\n", h_histogram[1]);
    printf("  50-199:         %llu\n", h_histogram[2]);
    printf("  200-999:        %llu\n", h_histogram[3]);
    printf("  1000-%d: %llu\n", max_turns - 1, h_histogram[4]);
    printf("  >= %d (tetto):  %llu\n", max_turns, h_histogram[5]);

    unsigned int n_to_write = h_hit_count < max_hits ? h_hit_count : max_hits;
    std::vector<uint64_t> h_ranks(n_to_write);
    std::vector<uint32_t> h_turns(n_to_write);
    CUDA_CHECK(cudaMemcpy(h_ranks.data(), d_hit_ranks, n_to_write * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_turns.data(), d_hit_turns, n_to_write * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::ofstream out(out_path, std::ios::binary);
    int32_t max_turns32 = max_turns;
    uint32_t n32 = n_to_write;
    out.write(reinterpret_cast<char *>(&max_turns32), sizeof(max_turns32));
    out.write(reinterpret_cast<char *>(&n32), sizeof(n32));
    for (unsigned int i = 0; i < n_to_write; i++) {
        out.write(reinterpret_cast<char *>(&h_ranks[i]), sizeof(uint64_t));
        out.write(reinterpret_cast<char *>(&h_turns[i]), sizeof(uint32_t));
    }
    printf("\nScritti %u hit in %s\n", n_to_write, out_path);

    cudaFree(d_hit_ranks);
    cudaFree(d_hit_turns);
    return 0;
}

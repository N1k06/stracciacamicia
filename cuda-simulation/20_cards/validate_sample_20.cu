// validate_sample_20.cu
//
// Confronta l'output della GPU con il riferimento Python su un CAMPIONE di
// rank espliciti (non l'intero spazio, gia' coperto esaustivamente da
// search_full_20.cu). Utile come verifica indipendente e piu' rapida.
//
// Uso:
//   nvcc -O3 -arch=sm_XX validate_sample_20.cu -o validate_sample_20
//   ./validate_sample_20 table_20.bin sample_ranks_20.bin sample_turns_20.bin

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

#define MAX_DECK 40
#define MAX_TABLE 4096

__constant__ uint64_t d_table[MAX_TABLE];
__constant__ int d_counts0[4];
__constant__ int d_dims[4];
__constant__ int d_deck_size;
__constant__ int d_half_size;
__constant__ int d_max_turns;

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

struct Queue { uint8_t data[MAX_DECK]; int head; int size; };

__device__ __forceinline__ int q_pop_front(Queue &q) {
    int v = q.data[q.head];
    q.head++; if (q.head == MAX_DECK) q.head = 0;
    q.size--;
    return v;
}
__device__ __forceinline__ void q_push_back(Queue &q, int v) {
    int slot = q.head + q.size;
    if (slot >= MAX_DECK) slot -= MAX_DECK;
    q.data[slot] = (uint8_t)v;
    q.size++;
}

__device__ int simulate(const uint8_t *deal_a, const uint8_t *deal_b, int half, int max_turns) {
    Queue handA{}, handB{}, pile{};
    handA.head = 0; handA.size = half;
    handB.head = 0; handB.size = half;
    pile.head = 0; pile.size = 0;
    for (int i = 0; i < half; i++) { handA.data[i] = deal_a[i]; handB.data[i] = deal_b[i]; }

    Queue* hand[2] = {&handA, &handB};
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

__global__ void test_kernel(const uint64_t *ranks, int n, uint32_t *out_turns) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint8_t deck[MAX_DECK];
    unrank_deck(ranks[tid], deck);
    out_turns[tid] = (uint32_t)simulate(deck, deck + d_half_size, d_half_size, d_max_turns);
}

static std::vector<char> read_file(const char *path) {
    std::ifstream fh(path, std::ios::binary | std::ios::ate);
    if (!fh) { fprintf(stderr, "Impossibile aprire %s\n", path); exit(1); }
    size_t s = (size_t)fh.tellg(); fh.seekg(0);
    std::vector<char> buf(s); fh.read(buf.data(), s);
    return buf;
}

#define CUDA_CHECK(call) do { cudaError_t e__ = (call); if (e__ != cudaSuccess) { \
    fprintf(stderr, "Errore CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s table_20.bin sample_ranks_20.bin sample_turns_20.bin\n", argv[0]);
        return 1;
    }

    auto table_buf = read_file(argv[1]);
    int32_t counts[4]; memcpy(counts, table_buf.data(), sizeof(counts));
    int dims[4] = {counts[0] + 1, counts[1] + 1, counts[2] + 1, counts[3] + 1};
    int table_size = dims[0] * dims[1] * dims[2] * dims[3];
    if (table_size > MAX_TABLE) { fprintf(stderr, "Tabella troppo grande.\n"); return 1; }
    const uint64_t *table_data = reinterpret_cast<const uint64_t *>(table_buf.data() + sizeof(counts));
    int deck_size = counts[0] + counts[1] + counts[2] + counts[3];
    int half_size = deck_size / 2;

    auto ranks_buf = read_file(argv[2]);
    int n = (int)(ranks_buf.size() / sizeof(uint64_t));
    std::vector<uint64_t> h_ranks(n);
    memcpy(h_ranks.data(), ranks_buf.data(), ranks_buf.size());

    auto turns_buf = read_file(argv[3]);
    uint32_t ref_max_turns, ref_n;
    memcpy(&ref_max_turns, turns_buf.data(), 4);
    memcpy(&ref_n, turns_buf.data() + 4, 4);
    if ((int)ref_n != n) {
        fprintf(stderr, "Mismatch dimensioni: %d rank ma %u turni di riferimento\n", n, ref_n);
        return 1;
    }
    const uint32_t *h_ref_turns = reinterpret_cast<const uint32_t *>(turns_buf.data() + 8);

    printf("Composizione: [%d,%d,%d,%d]  carte=%d  meta'=%d\n", counts[0], counts[1], counts[2], counts[3], deck_size, half_size);
    printf("Rank campione: %d  max_turns riferimento=%u\n", n, ref_max_turns);

    int max_turns = (int)ref_max_turns;
    CUDA_CHECK(cudaMemcpyToSymbol(d_table, table_data, table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_counts0, counts, sizeof(counts)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dims, dims, sizeof(dims)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_deck_size, &deck_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_half_size, &half_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_max_turns, &max_turns, sizeof(int)));

    uint64_t *d_ranks; uint32_t *d_out_turns;
    CUDA_CHECK(cudaMalloc(&d_ranks, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_turns, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (n + threads - 1) / threads;
    test_kernel<<<blocks, threads>>>(d_ranks, n, d_out_turns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_out_turns(n);
    CUDA_CHECK(cudaMemcpy(h_out_turns.data(), d_out_turns, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    const int MAX_PRINT = 15;
    for (int i = 0; i < n; i++) {
        if (h_out_turns[i] != h_ref_turns[i]) {
            mismatches++;
            if (mismatches <= MAX_PRINT)
                fprintf(stderr, "[MISMATCH] rank=%llu GPU=%u Python=%u\n",
                        (unsigned long long)h_ranks[i], h_out_turns[i], h_ref_turns[i]);
        }
    }

    printf("\n=== RISULTATO VALIDAZIONE CAMPIONE ===\n");
    printf("Rank testati: %d\n", n);
    printf("Corrispondenze: %d OK, %d MISMATCH\n", n - mismatches, mismatches);
    printf("\n%s\n", mismatches == 0 ? "TUTTI I TEST SUPERATI." : "ATTENZIONE: divergenze trovate.");

    cudaFree(d_ranks);
    cudaFree(d_out_turns);
    return mismatches == 0 ? 0 : 1;
}

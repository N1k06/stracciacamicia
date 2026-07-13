// test_unrank.cu
//
// Verifica che l'algoritmo di unranking (e la sua inversa, il ranking) girino
// correttamente sulla GPU, per la composizione fissa 28/4/4/4 (mazzo di
// Straccia Camicia). Due controlli indipendenti:
//
//   1. Confronto diretto: la sequenza calcolata dalla GPU per un dato rank deve
//      coincidere esattamente con quella calcolata in Python (verita' di terra),
//      per lo stesso identico rank.
//   2. Round-trip: partendo dal rank, si calcola la sequenza (unrank), poi da
//      quella sequenza si ricalcola il rank (rank, l'operazione inversa). Deve
//      tornare il rank di partenza. Fatto interamente sulla GPU.
//
// Uso:
//   nvcc -O3 -arch=sm_XX --ptxas-options=-v test_unrank.cu -o test_unrank
//   ./test_unrank multinomial_table.bin test_ranks.bin reference_sequences.bin
//
// (sostituire sm_XX con la compute capability della GPU, es. sm_75 per una T4,
// sm_80/sm_86 per A100/altre Ampere, sm_89 per L4/Ada -- su Colab va bene anche
// omettere -arch e lasciare che nvcc scelga un default ragionevole per un primo test)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

#define DIM0 29
#define DIM1 5
#define DIM2 5
#define DIM3 5
#define TABLE_SIZE (DIM0 * DIM1 * DIM2 * DIM3)

// Tabella dei coefficienti multinomiali, precalcolata in Python e caricata qui
// una volta sola prima di ogni lancio di kernel. Vive in __constant__ memory:
// e' di sola lettura per i kernel, con una cache dedicata molto efficiente
// quando tutti i thread di un warp leggono lo stesso indirizzo nello stesso
// momento (il nostro caso, dato che tutti i thread consultano la stessa tabella).
__constant__ uint64_t d_table[TABLE_SIZE];

// Composizione base del mazzo: 28 carte "0", 4 carte "1", 4 carte "2", 4 carte "3".
__constant__ int d_counts0[4] = {28, 4, 4, 4};

__device__ __forceinline__ uint64_t tbl(int c0, int c1, int c2, int c3) {
    int idx = ((c0 * DIM1 + c1) * DIM2 + c2) * DIM3 + c3;
    return d_table[idx];
}

// Unranking: dato un rank, ricostruisce la sequenza di 40 simboli corrispondente.
__device__ void unrank40(uint64_t rank, uint8_t out[40]) {
    int c[4] = {d_counts0[0], d_counts0[1], d_counts0[2], d_counts0[3]};
    for (int pos = 0; pos < 40; pos++) {
        for (int sym = 0; sym < 4; sym++) {
            if (c[sym] == 0) continue;
            c[sym]--;
            uint64_t perms = tbl(c[0], c[1], c[2], c[3]);
            if (rank < perms) {
                out[pos] = (uint8_t)sym;
                break;
            }
            rank -= perms;
            c[sym]++;
        }
    }
}

// Ranking (operazione inversa): dalla sequenza di 40 simboli calcola il rank.
__device__ uint64_t rank40(const uint8_t seq[40]) {
    int c[4] = {d_counts0[0], d_counts0[1], d_counts0[2], d_counts0[3]};
    uint64_t rank = 0;
    for (int pos = 0; pos < 40; pos++) {
        int sym_actual = seq[pos];
        // Sommiamo la dimensione di ogni blocco "che precede" il simbolo reale
        // (tutti i simboli piu' piccoli che si sarebbero potuti piazzare qui).
        for (int sym = 0; sym < sym_actual; sym++) {
            if (c[sym] == 0) continue;
            c[sym]--;
            rank += tbl(c[0], c[1], c[2], c[3]);
            c[sym]++;
        }
        c[sym_actual]--;
    }
    return rank;
}

__global__ void test_kernel(const uint64_t *ranks, int n,
                             uint8_t *out_seqs, uint64_t *out_ranks_recomputed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint64_t rank = ranks[tid];
    uint8_t seq[40];
    unrank40(rank, seq);

    for (int i = 0; i < 40; i++)
        out_seqs[tid * 40 + i] = seq[i];

    out_ranks_recomputed[tid] = rank40(seq);
}

static std::vector<char> read_file(const char *path, size_t expected_size = 0) {
    std::ifstream fh(path, std::ios::binary | std::ios::ate);
    if (!fh) {
        fprintf(stderr, "Impossibile aprire il file: %s\n", path);
        exit(1);
    }
    size_t size = (size_t)fh.tellg();
    fh.seekg(0);
    std::vector<char> buf(size);
    fh.read(buf.data(), size);
    if (expected_size != 0 && size != expected_size) {
        fprintf(stderr, "Dimensione inattesa per %s: attesi %zu byte, trovati %zu\n",
                path, expected_size, size);
        exit(1);
    }
    return buf;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "Errore CUDA a %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s multinomial_table.bin test_ranks.bin reference_sequences.bin\n", argv[0]);
        return 1;
    }

    // --- caricamento tabella multinomiale generata da Python ---
    auto table_buf = read_file(argv[1], TABLE_SIZE * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpyToSymbol(d_table, table_buf.data(), table_buf.size()));

    // --- caricamento rank di test ---
    auto ranks_buf = read_file(argv[2]);
    int n = (int)(ranks_buf.size() / sizeof(uint64_t));
    std::vector<uint64_t> h_ranks(n);
    memcpy(h_ranks.data(), ranks_buf.data(), ranks_buf.size());
    printf("Caricati %d rank di test.\n", n);

    // --- caricamento sequenze di riferimento (calcolate in Python) ---
    auto ref_buf = read_file(argv[3], (size_t)n * 40);
    const uint8_t *h_ref = reinterpret_cast<const uint8_t *>(ref_buf.data());

    // --- allocazione buffer device ---
    uint64_t *d_ranks, *d_ranks_recomputed;
    uint8_t *d_seqs;
    CUDA_CHECK(cudaMalloc(&d_ranks, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ranks_recomputed, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_seqs, (size_t)n * 40));

    CUDA_CHECK(cudaMemcpy(d_ranks, h_ranks.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    test_kernel<<<blocks, threads>>>(d_ranks, n, d_seqs, d_ranks_recomputed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_seqs((size_t)n * 40);
    std::vector<uint64_t> h_ranks_recomputed(n);
    CUDA_CHECK(cudaMemcpy(h_seqs.data(), d_seqs, (size_t)n * 40, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ranks_recomputed.data(), d_ranks_recomputed, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // --- confronto 1: sequenza GPU vs sequenza di riferimento Python ---
    // --- confronto 2: round-trip rank -> unrank -> rank, calcolato sulla GPU ---
    int mismatches_seq = 0;
    int mismatches_roundtrip = 0;
    const int MAX_PRINT = 10;

    for (int i = 0; i < n; i++) {
        bool seq_ok = true;
        for (int j = 0; j < 40; j++) {
            if (h_seqs[i * 40 + j] != h_ref[i * 40 + j]) { seq_ok = false; break; }
        }
        bool roundtrip_ok = (h_ranks_recomputed[i] == h_ranks[i]);

        if (!seq_ok) {
            mismatches_seq++;
            if (mismatches_seq <= MAX_PRINT) {
                fprintf(stderr, "[SEQ MISMATCH] rank=%llu\n  GPU: ", (unsigned long long)h_ranks[i]);
                for (int j = 0; j < 40; j++) fprintf(stderr, "%d", h_seqs[i * 40 + j]);
                fprintf(stderr, "\n  REF: ");
                for (int j = 0; j < 40; j++) fprintf(stderr, "%d", h_ref[i * 40 + j]);
                fprintf(stderr, "\n");
            }
        }
        if (!roundtrip_ok) {
            mismatches_roundtrip++;
            if (mismatches_roundtrip <= MAX_PRINT) {
                fprintf(stderr, "[ROUNDTRIP MISMATCH] rank originale=%llu, ricalcolato=%llu\n",
                        (unsigned long long)h_ranks[i], (unsigned long long)h_ranks_recomputed[i]);
            }
        }
    }

    printf("\n=== RISULTATO TEST ===\n");
    printf("Rank testati: %d\n", n);
    printf("Sequenze GPU vs riferimento Python: %d OK, %d MISMATCH\n", n - mismatches_seq, mismatches_seq);
    printf("Round-trip rank->unrank->rank su GPU: %d OK, %d MISMATCH\n", n - mismatches_roundtrip, mismatches_roundtrip);

    bool all_ok = (mismatches_seq == 0 && mismatches_roundtrip == 0);
    printf("\n%s\n", all_ok ? "TUTTI I TEST SUPERATI." : "ATTENZIONE: test falliti, vedi dettagli sopra (stderr).");

    cudaFree(d_ranks);
    cudaFree(d_ranks_recomputed);
    cudaFree(d_seqs);

    return all_ok ? 0 : 1;
}

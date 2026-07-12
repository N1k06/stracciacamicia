from math import factorial
from collections import Counter

def multinomial(counts):
    """Calcola il numero di permutazioni con ripetizioni per un dato multinsieme"""
    total = sum(counts.values())
    result = factorial(total)
    for v in counts.values():
        result //= factorial(v)
    return result

def unrank_multiset(rank, multiset):
    """Restituisce la permutazione n-esima (unranking) del multinsieme"""
    counts = Counter(multiset)
    result = []
    length = len(multiset)

    for _ in range(length):
        for char in sorted(counts):  # ordinamento per stabilità
            if counts[char] == 0:
                continue
            counts[char] -= 1
            perms = multinomial(counts)
            if rank < perms:
                result.append(char)
                break
            else:
                rank -= perms
                counts[char] += 1
    return ''.join(result)

def main():
    base_str = '0'*28 + '1'*4 + '2'*4 + '3'*4
    total_perms = multinomial(Counter(base_str))
    step = 200_000_000

    print(f"Totale permutazioni distinte: {total_perms}")
    print(f"Stampo ogni {step}-esima permutazione:\n")

    rank = 0
    while rank < total_perms:
        perm = unrank_multiset(rank, base_str)
        print(f"{rank}: {perm}")
        rank += step

if __name__ == "__main__":
    main()

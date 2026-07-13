"""
compare_bitpacked_hits.py

Confronta l'output di search_full_20_bitpacked.cu con i 16 candidati GIA'
TROVATI E VALIDATI dalla versione ad array (il contenuto esatto di
hits_validation_report.txt che hai gia' generato ed ispezionato). Se la
conversione a bit-packed e' corretta, deve trovare esattamente lo stesso
insieme di rank, con esattamente gli stessi turni -- nessuno in piu', nessuno
in meno, nessun valore diverso.

Uso:
    python3 compare_bitpacked_hits.py hits_20_bitpacked.bin
"""

import struct
import sys

# Candidati noti, presi verbatim da hits_validation_report.txt (versione ad
# array, gia' validata contro il riferimento Python turno per turno).
KNOWN_HITS = {
    246414: 5000,
    477920: 5000,
    917330: 5003,
    670912: 5000,
    1120477: 5004,
    1257979: 5005,
    1397712: 5002,
    1199338: 5001,
    1849144: 5000,
    1613646: 5003,
    2597170: 5004,
    2568257: 5000,
    2779940: 5005,
    3305725: 5004,
    3371832: 5003,
    3330878: 5001,
}


def main():
    if len(sys.argv) != 2:
        print("Uso: python3 compare_bitpacked_hits.py hits_20_bitpacked.bin")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as fh:
        max_turns, n = struct.unpack("<iI", fh.read(8))
        found = {}
        for _ in range(n):
            rank, turns = struct.unpack("<QI", fh.read(12))
            found[rank] = turns

    print(f"Candidati attesi (versione ad array, gia' validata): {len(KNOWN_HITS)}")
    print(f"Candidati trovati dalla versione bit-packed: {len(found)}")
    print(f"max_turns usato nella ricerca bit-packed: {max_turns}\n")

    known_ranks = set(KNOWN_HITS)
    found_ranks = set(found)

    missing = known_ranks - found_ranks       # trovati prima, non trovati ora
    extra = found_ranks - known_ranks          # trovati ora, non presenti prima
    common = known_ranks & found_ranks

    turn_mismatches = [
        (r, KNOWN_HITS[r], found[r]) for r in common if KNOWN_HITS[r] != found[r]
    ]

    ok = True

    if missing:
        ok = False
        print(f"MANCANTI ({len(missing)}): rank presenti prima ma non trovati dalla versione bit-packed:")
        for r in sorted(missing):
            print(f"  rank={r}  (turni attesi: {KNOWN_HITS[r]})")

    if extra:
        ok = False
        print(f"\nIN PIU' ({len(extra)}): rank trovati ora ma assenti nella versione ad array:")
        for r in sorted(extra):
            print(f"  rank={r}  (turni bit-packed: {found[r]})")

    if turn_mismatches:
        ok = False
        print(f"\nTURNI DIVERSI ({len(turn_mismatches)}): stesso rank, conteggio turni diverso:")
        for r, t_known, t_found in turn_mismatches:
            print(f"  rank={r}  array={t_known}  bit-packed={t_found}")

    if ok:
        print(f"\nTUTTI I {len(KNOWN_HITS)} CANDIDATI COINCIDONO ESATTAMENTE tra le due versioni.")
        print("La conversione a bit-packed e' validata: stesso insieme di rank, stessi turni.")
    else:
        print("\nATTENZIONE: la versione bit-packed NON coincide con la versione ad array gia' validata.")
        print("Non procedere al mazzo da 40 carte finche' questa discrepanza non e' risolta.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Assembly based on Debruijn graph

Simple assembly program based debruijn graph.
This work is part of an academic course gave by the Dr. Ghozlane of the Pasteur Institute.

## Dependencies

```
pip install networkx pytest pylint pytest-cov
```

## Basic Usage

```
usage:
    python debruijn.py -i=FILE -k=INT -o=FILE

option:
    -i File with reads, FASTA format
    -k Size of the kmer
    -o Output file with contigs, FASTA format
```

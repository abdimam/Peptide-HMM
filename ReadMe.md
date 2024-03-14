# pHMM Decoder
The pHMM Decoder is a Python tool that decodes a Profile Hidden Markov Model (pHMM) with given sequences, following the conventions of HMMER but implemented in Python. It currently supports the Forward and Viterbi algorithms.

## Functionality
In addition to the standard HMMER functionality, the pHMM Decoder offers an extra feature: the ability to model inversion uncertainties. This feature, which is only compatible with the Viterbi algorithm, provides enhanced modeling capabilities.

## Usage
To use the pHMM Decoder, follow these steps:

### Initialization:
Initialize the Decoder class with a FASTA file containing sequences and a HMM format file.

Example initialization
from pHMMDecoder import Decoder
decoder = Decoder("sequences.fa", "model.hmm")

### Decoding:
Call the decode() function, specifying the algorithm and inversion mode if desired.
result = decoder.decode(alg_base="viterbi", inverse_mode=True)

## Example
Here's an example of how to use the pHMM Decoder:

from pHMMDecoder import Decoder

decoder = Decoder("sequences.fa", "model.hmm")

result = decoder.decode(alg_base="viterbi", inverse_mode=True)

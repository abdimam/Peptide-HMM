# pHMM Decoder
The pHMM Decoder is a Python tool that decodes a Profile Hidden Markov Model (pHMM) with given sequences, following the conventions of [HMMER](http://eddylab.org/software/hmmer/Userguide.pdf) but implemented in Python. It currently supports the Forward and Viterbi algorithms. The implementation DOES NOT include the peformance enchancing filters.

## Functionality
In addition to the standard HMMER functionality, the pHMM Decoder offers an extra feature: the ability to model inversion uncertainties. This feature, which is only compatible with the Viterbi algorithm, provides enhanced modeling capabilities. It should be noted that pHMM Decoder can also report the score from Viterbi algorithm to the user, which HMMER does not.

The standard HMMER functionality includes calculating a bit score (log2-odds) of the sequence being homologous to the given pHMM. Viterbi will in addition to giving such a score also give an alignment of the sequence against the pHMM.

## Usage
To use the pHMM Decoder, follow these steps:

### Initialization:
Initialize the Decoder class with a FASTA file containing sequences and a HMM format file.

Example initialization
```
from pHMMDecoder import Decoder
decoder = Decoder("sequences.fa", "model.hmm")
```
### Decoding:
Call the inverse() function, specifying the algorithm and inversion mode if desired.
```
result = decoder.inverse(alg_base="viterbi", inverse_mode=True)
```
## Example
Here's an example of how to use the pHMM Decoder:
```
from pHMMDecoder import Decoder

decoder = Decoder("sequences.fa", "model.hmm")

result = decoder.inverse(alg_base="viterbi", inverse_mode=True)
```
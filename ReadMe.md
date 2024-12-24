# pHMM Decoder
The pHMM Decoder is a Python tool designed to decode a Profile Hidden Markov Model (pHMM) using a set of peptide sequences provided in FASTA format. It follows the conventions outlined in the HMMER User Guide but is implemented entirely in Python. This tool attempts to match peptide sequences to the pHMM and reconstruct the corresponding protein's amino acid sequence. . Note that the implementation does not include performance-enhancing filters.

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
result = decoder.inverse(alg_base="viterbi", inverse_mode=False, consensus_alignment=True, mspeptide_alignment=True)
```
## Example
Here's an example of how to use the pHMM Decoder:
```
from pHMMDecoder import Decoder

decoder = Decoder("sequences.fa", "model.hmm")

result = decoder.inverse(alg_base="viterbi", inverse_mode=False, consensus_alignment=True, mspeptide_alignment=True)
```

# STS via Jaccard Similarity of Tsetlin Machine Optimized Topic Sets

## What's in this codebase?

- bin/tmjc-sts
  - convert-glove -o <out-file> -f <glove-file> -k <num_groups> [ ...other opts ]
    - produces two files froma  pre-trained glove dataset:
        - Word to cluster ID
        - Word to bitmap
  - train-tm -o <out-file> -f <bitmap-file> -w <training-file> -i <iterations>
    - train a set of Tsetlin Machines to perform the Jaccard Similarity
      refinement task, saving weights to a file

- lib/tmjc-sts
  - libraries for runtime usage
    - Load a word to bitmap table and TM weights
    - Convert a sentence to it's bitmap representation
    - Jaccard Similarity between bitmaps
    - Memory-optimized approximate nearest neighbor search

## Basic Implementation

- Word embeddings are clustered via K-means++ clustering into K groups via
  cosine similarity (note that the K-means++ variant is used to ensure optimal
  clustering)

- Words are modeled as a bitmap of length K with the bit corresponding to its
  cluster set

- Sentences are modeled as an OR over all of the word-bitmaps therein

- A Tsetlin Machine is trained to learn which bits should be flipped as to
  optimize for the correct Jaccard Similarity between sentence bitmaps.

## Sentence Representation & Training

- Given a sentence represention described above, use one Tsetlin Machine for
  each bit classifying whether or not the bit was set correctly as to align the
  Jaccard Similarity of sentences-as-bitmaps with human-labeled data

- A Tsetlin Machine returning 0 means "do not change this bit," whereas 1 means
  "flip the bit"

- Initialize all Tsetlin Machines with random internal states

- Calculate the initial total loss by iterating through each training set pair
  and taking the overall sum of the differences between the Jaccard Similarities
  of the bitmap representations and the human-labeled similarity.

- Calculate the next total loss by iterating through each training pair as above
  but pass the bitmap representations through the TMs and adjust their bits
  accordingly before taking the Jaccard similarities and comparing to the human
  labeled data

- For each TM, if changing its individual decision would have resulted in a
  lower overall loss, reward it, otherwise penalize it.

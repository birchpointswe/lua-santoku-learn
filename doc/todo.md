# Now

- Clustering
    - Implement eval.clustering_accuracy, similar to encoding_similarity, which
      sweeps for best margin for optimal F1 but using same cluster membership as
      the target.
        - TP: similar and same cluster
        - FP: dissimilar but same cluster
        - TN: dissimilar but diff cluster or both noise
        - FN: similar but diff cluster or both noise

- ANN search
    - After spectral, demo search on train/test codes

# Next

- tk_inv_t
    - Strictly sparse, no templating
    - Based on tk_iuset_t
    - Uses inverted index with jaccard similarity

- tk_graph_t
    - Accept tk_inv_t instead of sentences/nodes
    - Replace roaring with tk_iuset_t

- tk_xxmap_t:
    - Templatize over khash/kbtree
    - Replace use of roaring with this
    - Proper Lua/C API like tk_xvec_t
    - Replace all tk_malloc and manual kh_x initializations with vec/map
      templates across libraries
    - Automatically add parent ephemeron and pop on create

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- TBHSS
    - Rename
    - Integrate latest TM updates
    - Explore classify
    - Explore encode
    - Index & search

# Later

- tk_ann_t
    - Precompute probes

- Clustering
    - Move implementation to C with proper Lua/C APIs

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)
    - Allow dense bit operations on arbitrary-length input (non-multiples of 64,
      see the filter pattern from initial TM implementation)

- tk_rvec_t
    - Add Lua API
    - Generalize template to allow push/peekbase to work with composite types

- tk_tokenizer_t
    - Proper Lua/C API and tk naming
    - Use booleanizer under the hood
    - Include text statistics as continuous/thresholded values

- tk_graph_t
    - Proper Lua/C API and tk naming
    - Support querying for changes made to seed list over time (removed during
      dedupe, added by phase)
    - Support KFN (furthest neighbors) in addition to KNN for densification of
      negatives (random zero-overlap nodes first, then furthest in feature
      space)

- Clustering
    - K-means, k-medoids, hierarchical?

- Double-check auto-vectorization and paralellization across the board
    - Especially corex, given added branching/etc logic added

- Parallelize!

# Consider

- vec/tpl.h
      Allow tk_vec_noundef to skip undefs

- tk_roaring_t
    - Separate library that uses the tk_xxmap_t API to wrap roaring bitmaps

- TCH
    - Add learnability to optimization?

- Corex
    - Should anchor multiply feature MI instead of hard-boosting it?
    - Allow explicit anchor feature list instead of last n_hidden, supporting
      multiple assigned to same latent, etc.

# Eventually

- Multi-layer classifiers and encoders

- Absorbing (speeds up training over time as clauses are dropped)

- Convolutional
    - Tried this, see branch

- Coalesced (reduces memory footprint for multi-output configurations)
    - Tried this, didn't really help

- Weighted (reduces memory footprint)
- Integer-Weighted (see weighted)

- Indexed (improves learning and classification speed)
    - Triples memory usage

- Titanic dataset

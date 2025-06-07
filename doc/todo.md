# Now

- tk_graph_t
    - Accept tk_inv_t instead of sentences/nodes
    - Replace roaring with tk_iuset_t

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t
    - Automatically add parent ephemeron and pop on create

- tk_inv_t
    - Use heap for candidates

- tk_ann_t
    - Use heap here like in inv

- Chores
    - Standardize API between tk_ann_t and tk_inv_t
    - Table input to entropy_stats and optimize_retrieval
    - Reorganize optimize_retrieval as per optimize_clustering
    - Parallelize
    - Sanitize

# Next

- Chores
    - Documentation

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- tk_ann_t
    - After spectral, demo search on train/test codes

- TBHSS
    - Rename
    - Integrate latest TM updates
    - Explore classify
    - Explore encode
    - Index & search

# Later

- Spectral
    - Weighted laplacian (ground truths, transitives, knn, kfn, etc.)

- tk_ann_t
    - Precompute probes

- tk_ann_t
    - Allow user-provided list of hash bits and enable easy mi, chi2 or
      corex-based feature bit-selection.
      - mi, chi2, and corex could be run over all pairs using 1/pos 0/neg as the
        binary label and the XOR of the nodes' features as the set of features

- Clustering
    - Move implementation to C with proper Lua/C APIs

- General
    - replace most tk_malloc/realloc/numa with tk_xvec operations across
      libraries

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)
    - Allow dense bit operations on arbitrary-length input (non-multiples of 64,
      see the filter pattern from initial TM implementation)

- tk_r/pvec_t
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

- Clustering
    - K-means, k-medoids, hierarchical?

- Double-check auto-vectorization across the board
    - Especially corex, given added branching/etc logic added

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

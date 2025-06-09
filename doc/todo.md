# Now

- Chores
    - Sanitize
    - Refactor other tests
    - Commit

- Eval
    - Table input to entropy_stats and optimize_retrieval
    - Reorganize optimize_retrieval as per optimize_clustering

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- tk_inv/ann_t
    - Persist/load to string and disk
    - Demo search on train/test codes (for ann) and raw features (for inv)

- TBHSS
    - Rename
    - Integrate latest TM updates
    - Explore classify
    - Explore encode
    - Index & search

# Next

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- tk_xvec_t
    - Align Lua/C APIs (currently out of sync for pvec, rvec, etc)

- tk_booleanizer_t
    - When both double and string observations found, split into two different
      features, one for continuous and the other for categorical

- tk_tokenizer_t
    - Proper Lua/C API and tk naming
    - Use booleanizer under the hood
    - Include text statistics as continuous/thresholded values

- Chores
    - Parallelize
    - Vectorize
    - Document

# Later

- tk_tsetlin_t
    - Regression

- tk_graph_t
    - Proper Lua/C API and tk naming
    - Support querying for changes made to seed list over time (removed during
      dedupe, added by phase)

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
    - Replace most tk_malloc/realloc/numa with tk_xvec operations across
      libraries

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)
    - Allow dense bit operations on arbitrary-length input (non-multiples of 64,
      see the filter pattern from initial TM implementation)

- tk_xvec_t
    - Generalize template to allow push/peekbase to work with composite types

- Clustering
    - K-means, k-medoids, hierarchical?

# Consider

- tk_dsu_t
    - Split from graph for other uses?
    - Full Lua/C API?

- vec/tpl.h
      Allow tk_vec_noundef to skip undefs?

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

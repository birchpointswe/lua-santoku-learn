# Now

- Clustering visualization
- All tests working without leaks/errors

- tk_inv/ann/hbi_t
    - Implement inv neighbors
    - DRY up ANN and HBI
    - Persist/load to/from disk/string (note that this will likely first require compaction to ensure contiguous
      vectors)
    - Final TODOs

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- Chore
    - Set num threads for blas in spectral, itq, and anywhere else so that THREADS is respected
    - Sanitize everything

- TBHSS
    - Get working with new TM code
    - Explore classify
    - Explore encode
    - Confirm accuracy on mnist/imdb is as expected in both classify and encode mode
    - Index & search

# Next

- tk_ann_t
    - Allow user to pass dataset in, which then uses top_entropy to select bits
      internally

- tk_xvec_t
    - Align Lua/C APIs (currently out of sync for pvec, rvec, etc)
    - Optional lua gc management via tk_xvec_create(NULL, ...), allowing later opt-in via tk_xvec_register(L, xv)

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- TBHSS
    - Move hyperparameter search/exploration code into TM library
    - Restrict TBHSS code to cover cli usage, pre-processing, etc.

- Graph
    - Support passing labels to graph to ensure edges respect classes (e.g. dont positively connect between diff classes
      or negative with same, support multi-labels as cvec and single-labels as ivec)
    - Move multiclass-to-graph initial setup using star centers/negatives into graph.create such that seed pos/neg pairs
      become optional.
        - E.g. graph.create({ index, labels, trans_hops/pos/neg, etc })

- Clustering
    - Huge opportunities for performance improvement

- Misc
    - Generalize the patterns found in tests into:
        - santoku.tsetlin.encoder: wraps all supported encoder use-cases: graph or labeled data input, generating a
          codebook, clustering, learning hash functions, ann search, persist/load, etc, integrates booleanizer,
        tokenizer, etc, to reduce user-required pre-processing
        - santoku.tsetlin.classifier: similar to encoder: labeled data input
        - santoku.tsetlin.explore: TM hyperparameter exploration
             - 1+ epoch restarts (few) doing a grid search or random hill climbing to find best starting point, followed
               by longer training runs on the top contenders

# Later

- QQP

- Eval/optimize
    - Split optimize_retrieval into separate module
        - optimize.retrieval: finds best margin for retrieval
        - optimize.clustering: finds best margin for DSU clustering
    - When scanning margins and multiple are tied for best, pick the median
    - Parallelize AUC rank accumulation over pairs

- Chores
    - Double-check & comment auto-vectorization opportunities

- TCH, ITQ
    - Parallelize

- tk_graph_t
    - Parallelize init & seed phase (slowest phase of entire pipeline)
    - Parallelize transitive expansion
    - Restrict trans via trans_pos/neg_k/eps

- Graph
    - Avoid precomputing hoods?

- tk_inv/ann/hbi_t
    - Parallelize add

- INV, ANN, HBI
    - Try to get INV down to 30s on 40k mnist (if possible)

- Chore
    - Generalized interface over INV, ANN, and HBI

- tk_booleanizer_t
    - Mergable booleanizers, externally paralellized + merged
    - encode_sparse/dense for ivec/cvec
    - When both double and string observations found, split into two different features, one for continuous and the
      other for categorical

- tk_tokenizer_t
    - Mergable tokenizers parallelized independently
    - Proper Lua/C API and tk naming
    - Use booleanizer under the hood
    - Include text statistics as continuous/thresholded values

- Chore
    - Document all public APIs

- TBHSS/etc
    - Log stats to disk(?) render with ggplot(?)

- Encoder
    - Curriculum learning, bit pruning, etc.

- tk_graph_t
    - Proper Lua/C API and tk naming
    - Support querying for changes made to seed list over time (removed during dedupe, added by phase)

- tk_ann_t
    - Precompute probes

- tk_ann_t
    - Allow user-provided list of hash bits and enable easy mi, chi2 or corex-based feature bit-selection.
      - mi, chi2, and corex could be run over all pairs using 1/pos 0/neg as the binary label and the XOR of the nodes'
        features as the set of features

- General
    - Replace most tk_malloc/realloc/numa with tk_xvec operations across libraries

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)
    - Allow dense bit operations on arbitrary-length input (non-multiples of 64, see the filter pattern from initial TM
      implementation)

- Chore
    - Proper Lua/C APIs across the board

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

# Consider

- Misc
    - Sparse/dense linear SVM for sanity checking all tasks

- Spectral
    - Weighted laplacian (ground truths, transitives, knn, kfn, etc.)

- Clustering
    - K-means, k-medoids, hierarchical, dbscan?

- tk_dsu_t
    - Split from graph for other uses?
    - Full Lua/C API?

- vec/tpl.h Allow tk_vec_noundef to skip undefs?

- Corex
    - Should anchor multiply feature MI instead of hard-boosting it?
    - Allow explicit anchor feature list instead of last n_hidden, supporting multiple assigned to same latent, etc.

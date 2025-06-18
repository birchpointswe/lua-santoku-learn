# Now

- Clustering via: DSU + Hamming Tree
    - Replace LSH with hamming tree
      Replace DBSCAN with DSU + Hamming tree
    - Write in C, scan all margins, parallelized by margin
    - Re-use DSU from graph
    - For each code (node) in your dataset:
      - Initialize a DSU/Union-Find parent array, so every node starts as its own cluster.
    - For a given Hamming radius (hard-code max of 3 if d <= 64, 2 if greater):
      - For each code (node):
        - Use the LSH ANN to query for all neighbors within the Hamming radius (excluding self).
        - For each neighbor found:
          - Use DSU to union the current node and its neighbor (i.e., merge their clusters).
    - To explore all possible radii (from 0 up to the code length, e.g., 0 to 8 or 64):
      - For each radius:
        - Repeat the neighbor-finding and union steps as above.
        - Optionally, after processing, extract cluster labels by finding DSU roots.
        - Record cluster statistics if desired (e.g., number of clusters, sizes, etc).

- Graph
    - Avoid precomputing hoods?
    - Support passing labels to graph to ensure edges respect classes (e.g. dont positively connect between diff classes
      or negative with same, support multi-labels as cvec and single-labels as ivec)
    - Move multiclass-to-graph initial setup using star centers/negatives into graph as graph.create({ index, labels,
      trans_hops/pos/neg, etc }), where seed pos/neg pairs become optional.

- Eval
    - Parallelize AUC rank accumulation over pairs
    - When scanning margins and multiple are tied for best, pick the median

- tk_graph_t
    - Parallelize init & seed phase (slowest phase of entire pipeline)
    - Parallelize transitive expansion
    - Restrict trans via trans_pos/neg_k/eps

- tk_inv/ann_t
    - Implement neighbors for inv
    - Refactor/DRY both implementations
    - ANN currently leaves a lot of performance on the table
    - Parallelize add
    - Persist/load to/from disk/string (note that this will likely first require compaction to ensure contiguous
      vectors)

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- Misc
    - Sparse/dense linear SVM for sanity checking all tasks

- Chore
    - Set num threads for blas in spectral, itq, and anywhere else so that THREADS is respected

- TBHSS
    - Get working with new TM code
    - Explore classify
    - Explore encode
    - Confirm accuracy on mnist/imdb is as expected in both classify and encode mode
    - Index & search

# Next

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- tk_xvec_t
    - Align Lua/C APIs (currently out of sync for pvec, rvec, etc)

- Chores
    - Double-check & comment auto-vectorization opportunities

- TBHSS
    - Move hyperparameter search/exploration code into TM library
    - Restrict TBHSS code to cover cli usage, pre-processing, etc.

- Misc
    - Generalize the patterns found in tests into:
        - santoku.tsetlin.encoder: wraps all supported encoder use-cases: graph or labeled data input, generating a
          codebook, clustering, learning hash functions, ann search, persist/load, etc, integrates booleanizer,
        tokenizer, etc, to reduce user-required pre-processing
        - santoku.tsetlin.classifier: similar to encoder: labeled data input
        - santoku.tsetlin.explore: TM hyperparameter exploration
             - 1+ epoch restarts (few) doing a grid search or random hill climbing to find best starting point, followed
               by longer training runs on the top contenders

- TCH, ITQ
    - Parallelize

- tk_inv/ann_t
    - Parallel add?
    - Shrink/compaction

# Later

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

- tk_tsetlin_t
    - Regression

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

# Consider

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

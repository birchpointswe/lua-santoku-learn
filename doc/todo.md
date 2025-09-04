# Now

- Finalize L-STH
    - AUC-based bit selection (greedy selection of highest AUC-gain bits)
    - Always store UIDs in neighborhoods. Set up indices if needed after.
    - Ensure docs up to date (various changes across matrix/tsetlin)
    - Revise l-sth.md
    - QQP

- ann/hbi: implement mutualize/etc
    - Note: Partially completed for ann
    - Include mutualize lua api

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- tk_cvec_t
    - Ensure cvec used everywhere (no strings!)
    - Implement cvec to/from string

- Lua GC for threadpool
- Pcalls for callbacks, or make sure everything is connected to the lua GC

# Next

- Avoid direct malloc, use newuserdata or cvec APIs

- Rename tch
- Rename to santoku-learn or similar
- Profile hot paths and vectorization opportunities

- Move conf.h items to other libraries
    - Interleaved allocation (santoku-threads)

- Versioning or other safety measures around persist/load

- Feature selection via analysis of clause weights, with option to prune
  unused literals, returning pruned for subsequent filtering of
  tokenizer/booleanizer, giving faster inference.

- Graph
    - Re-evaluate negative edge handling in graph (see 2025-08-10/simplify
      branch and prior history)
    - Will need this for handling signed (e.g. SNLI) input pairs in a logically
      sound way

- Rounding out functionality
    - Support PCA for dimensionality reduction (likely followed by ITQ)
    - Support sparse/dense linear SVM for codebook/classifier learning

- Clustering
    - Return n_clusters, n_core, n_border, n_noise

- Graph
    - Return positive and negative weight totals when creating adjacency

- Chore
    - Error checks on dimensions to prevent segfaults
    - Sanitize everything

- TCH
    - Parallelize

- Bitsel
    - Parallelize

# Later

- TBHSS
    - Reboot as a cli interface to this ML framework

- Templated Trie

- SNLI (will likely require some new graph features to handle the negatives)

- tk_graph_t
    - Parallelize init & seed phase (slowest phase of entire pipeline)
    - Parallelize transitive expansion
    - Restrict trans via trans_pos/neg_k/eps

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

- Misc
    - Generalize the patterns found in tests into:
        - santoku.tsetlin.encoder: wraps all supported encoder use-cases: graph or labeled data input, generating a
          codebook, clustering, learning hash functions, ann search, persist/load, etc, integrates booleanizer,
        tokenizer, etc, to reduce user-required pre-processing
        - santoku.tsetlin.classifier: similar to encoder: labeled data input
        - santoku.tsetlin.explore: TM hyperparameter exploration
             - 1+ epoch restarts (few) doing a grid search or random hill climbing to find best starting point, followed
               by longer training runs on the top contenders

- Eval/optimize
    - Split optimize_retrieval into separate module
        - optimize.retrieval: finds best margin for retrieval
        - optimize.clustering: finds best margin for DSU clustering
    - When scanning margins and multiple are tied for best, pick the median
    - Parallelize AUC rank accumulation over pairs

- Chores
    - Auto-vectorize
    - Document all public APIs

- tk_graph_t
    - Proper Lua/C API and tk naming
    - Support querying for changes made to seed list over time (removed during dedupe, added by phase)

- tk_hbi/ann_t
    - Precompute probes

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)

- Chore
    - Proper Lua/C APIs across the board

# Eventually

- tk_ann_t
    - Allow user-provided list of hash bits and enable easy mi, chi2 or corex-based feature bit-selection.
        - mi, chi2, and corex could be run over all pairs using 1/pos 0/neg as
          the binary label and the XOR of the nodes' features as the set of
          features

- Multi-layer classifiers and encoders

- Convolutional
    - Tried this, see branch

- Titanic dataset?

# Consider

- ProNe
    - Potential alternative to spectral
    - Also explore various non-neural alternatives to GNNs

- tk_dsu_t
    - Full Lua/C API?

- tk_ann_t
    - Guided hash bit selection (instead of random), using passed-in dataset to
      select bits by entropy or some other metric (is this actually useful?)

- Corex
    - Should anchor multiply feature MI instead of hard-boosting it?
    - Allow explicit anchor feature list instead of last n_hidden, supporting
      multiple assigned to same latent, etc.

- Generative model, next token predictor
    - Predict visible features from spectral codes
    - Spectral embeddings from graph of ngrams as nodes, probabilities as
      weights?

- Tsetlin
    - Bayesian optimization

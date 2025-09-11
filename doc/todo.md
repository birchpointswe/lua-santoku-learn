# Now

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- tk_cvec_t
    - Ensure cvec used everywhere (no strings!)
    - Implement cvec to/from string

- Lua GC for threadpool
- Connect all allocs to Lua
- Double-check lua callback error handling

# Next

- Supplementary documentation similar to l-sth.md for the general classification
  pipeline using tsetlin machines for binary, multiclass, and multi-output
  scenarios
    - Covering booleanizer, tokenizer, supporting external embeddings via
      booleanizer, feature selection via learned clauses, feature selection via
      chi2, mi, etc (look at test cases for all features)

- Rename tch to flipper
- Rename library and project to santoku-learn

- l-sth
    - IMDB & QQP
    - Encoding landmark codes as bit-frequencies instead of (or in addition to)
      concatenated codes for triangulation based on landmark bit statistics of
      neighbors.
    - Coreset sampling for landmark index (filter both features and samples for
      optimal triangulation)

- l-sth
    - Explore chi2/etc on landmark features
    - Select bits for encoder-learnability

- graph
    - Re-evaluate negative edge handling in graph (see 2025-08-10/simplify
      branch and prior history)
    - Will need this for handling signed (e.g. SNLI) input pairs in a logically
      sound way

- tsetlin
    - Feature selection via analysis of clause weights, with option to prune
      unused literals, returning pruned for subsequent filtering of
      tokenizer/booleanizer, giving faster inference.

- Additional capabilities
    - Sparse/dense PCA for dimensionality reduction (can be followed by ITQ)
    - Sparse/dense linear SVM for codebook/classifier learning

- Chores
    - Move conf.h helpers into other santoku libraries (str, hash, interleaved alloc)
    - Error checks on dimensions to prevent segfaults everywhere
    - Sanitize all tests and downstream projects for memory leaks
    - Persist/load versioning or other safety measures
    - Profile hot paths and vectorization opportunities

# Later

- tbhss
    - Reboot as a cli interface to this framework (toku learn {OPTIONS})

- tk_ctrie_t/tk_itrie_t/etc
    - templated trie

- l-sth
    - SNLI (will likely require some new graph features to handle the negatives)

- tk_graph_t
    - speed up init & seed phase (slowest phase of entire pipeline)

- tk_booleanizer_t
    - mergable booleanizers, externally paralellized + merged
    - encode_sparse/dense for ivec/cvec
    - when both double and string observations found, split into two different features, one for continuous and the
      other for categorical

- tk_tokenizer_t
    - mergable tokenizers parallelized independently
    - proper Lua/C API and tk naming
    - use booleanizer under the hood
    - include text statistics as continuous/thresholded values

- High-level APIs
    - santoku.learn.encoder
        - l-sth pipeline, covering all supported variations and defaulting to
          l-sth.md reference implementation: graph or labeled data input,
          generating a codebook, clustering, learning hash functions, ann search,
          persist/load, etc, integrates booleanizer, tokenizer, etc, to reduce
          user-required pre-processing, feature selection at all phases, etc.
    - santoku.learn.classifier
        - similar to encoder, but for classification pipelines
    - santoku.learn.explore
        - generalized hyperparameter exploration

- evaluator
    - when scanning margins and multiple are tied for best, pick the median

- tk_graph_t
    - Support querying for changes made to seed list over time (removed during dedupe, added by phase)

- tk_hbi/ann_t
    - Precompute probes

# Eventually

- graph
    - Consider removing neighborhoods calls from the graph module, instead of
      requiring the user to pass in the seed edges, potentially even with weights.
      The provided index, would then only be used for weight lookup in the fallback
      phase.

- hti: hamming tree index

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

- clustering
    - Return n_clusters, n_core, n_border, n_noise

- graph
    - Return positive and negative weight totals when creating adjacency

- tk_dsu_t
    - Full Lua/C API?

- tk_ann_t
    - Guided hash bit selection (instead of random), using passed-in dataset to
      select bits by entropy or some other metric (is this actually useful?)

- corex
    - should anchor multiply feature MI instead of hard-boosting it?
    - allow explicit anchor feature list instead of last n_hidden, supporting
      multiple assigned to same latent, etc.

- generative model, next token predictor
    - predict visible features from spectral codes
    - spectral embeddings from graph of ngrams as nodes, probabilities as
      weights?

- tsetlin
    - Bayesian optimization

- tm optimizer
    - Early stopping with no improvement

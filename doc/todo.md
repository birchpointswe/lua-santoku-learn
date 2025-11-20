# Now

- Parallelize booleanizer and tokenizer
- Need a true hierarchical test case for HLTH (newsgroups?)
- Abstract the landmark/out-of-sample phase behind a simple tm + index wrapper
- Regression, autoencoder, convolutional
- Explore shared libaries, optimistic dynmaic linking? Is that a thing?

# Next

- Rename library and project to santoku-learn

- Supplementary documentation similar to l-sth.md for the general classification
  pipeline using tsetlin machines for binary, multiclass, and multi-output
  scenarios
    - Covering booleanizer, tokenizer, supporting external embeddings via
      booleanizer, feature selection via learned clauses, feature selection via
      chi2, mi, etc (look at test cases for all features)

# Later

- Generalize tm.optimize exploration to an explore module that covers various
  optimizations: exhaustive, binary search, and the parameter sampling mechanism
  in tm.optimize, with integrated evaluation like the other optimize routines so
  we don't create evaluator thread pools every epoch

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

- tsetlin
    - Feature selection via analysis of clause weights, with option to prune
      unused literals, returning pruned for subsequent filtering of
      tokenizer/booleanizer, giving faster inference.

- Additional capabilities
    - Sparse/dense PCA for dimensionality reduction (can be followed by ITQ)
    - Sparse/dense linear SVM for codebook/classifier learning

- Chores
    - Error checks on dimensions to prevent segfaults everywhere
    - Persist/load versioning or other safety measures

- tbhss
    - Reboot as a cli interface to this framework (toku learn {OPTIONS})

- tk_ctrie_t/tk_itrie_t/etc
    - templated trie

- l-sth
    - SNLI (use flip_at 0.5 with edges below 0.5 for repulsion)

- tk_graph_t
    - speed up init & seed phase (slowest phase of entire pipeline)

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
    - When scanning margins and multiple are tied for best, pick the median

- tk_graph_t
    - Support querying for changes made to seed list over time (removed during dedupe, added by phase)

- ann/hbi
    - Further parallelize adding and removing from indices where possible
    - Precompute hash probes in ann/hbi
    - Consider bloom filter to avoid empty buckets
    - Explore/consider batch probe multiple buckets

- tokenizer
    - store known tokens contiguously, no need for separate mallocs/etc for each
      token. Use a single cvec as backend, storing pointers into that cvec
      insted of separately alloc'd strings. Ensure tokens are null terminated.

- Separate all lua api functions explicitly via _lua variant (_lua variants must
  respect the expected stack semantics strictly)

- Consistently follow the x/x_lua pattern for naming conventions

# Eventually

- tk_inv_t
    - Consider making decay a query-time parameter

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

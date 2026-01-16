# Now

- Support lambda for centroid clustering optimization instead of on/off. When 0,
  off, when 1.0, default, used as an alpha to loosen/tighten the lower bound
  optimization

- Support passing in an index instead of codes to clustering, which allows
  clustering based on tk_inv_t, tk_ann_t, or tk_hbi_t using a new
  tk_inv/ann/hbi_distances (batch distance) API
    - Centroid optimization disabled in this case

- Batch distance API for ann/inv indices (distance_many or similar) to avoid
  O(n*k) individual :distance() calls in diagnostics and weighted encoding

- Extend tests
    - imdb encoder
    - qqp encoder/classifier
    - snli encoder/classifier

- Supplementary documentation similar to sth/hlth/nystrom for classification
  pipelines

# Next

- Parallelize booleanizer and tokenizer

- Rename library and project to santoku-learn

- Additional capabilities
    - Sparse/dense PCA for dimensionality reduction (can be followed by ITQ)
    - Sparse/dense linear SVM for codebook/classifier learning

- Regression, autoencoder

- tsetlin
    - Interpretation of learned clauses, e.g. emitting clauses weighted by
      importance, confidence, memorization strength, etc.
        - Should show me top clauses or top features by class or overall
        - Overall top features/literals is similar to bits_top_xxx
        - By class top features/literals is similar to bits_top_chi2_ind
        - Overall & by class top clauses, where each "clause" is represented
          like "143 23 !345 41 !5"
        - We should demonstrate mapping these clauses back to known/intelligible
          names/etc (pixel coordinates for mnist, tokens for text, etc)
    - Prune unused literals, returning pruned for subsequent filtering of
      tokenizer/booleanizer, giving faster inference
        - Demonstrate this

- _ind variants for:
    - All bits_top functions
    - Corex top features

- Chores
    - Error checks on dimensions to prevent segfaults everywhere
    - Persist/load versioning or other safety measures

- tk_ctrie_t/tk_itrie_t/etc
    - templated trie

- tk_graph_t
    - speed up init & seed phase (slowest phase of entire pipeline)

- High-level APIs (tbhss 2.0)
    - santoku.learn.encoder
      santoku.learn.classifier
        - Generalizations of encoders and classifiers provided via a high-level
          API that takes source data, runs the entire optimization pipeline, and
          returns fully packaged and persistable runtime encoder/classifier
          constructs. A user should be able to plug in data in a variety of
          formats, and we auto parse, booleanize, etc according to their
          configurations.
    - santoku.learn.explore
        - Generalization over the existing random search with center-based
          tightening

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

- tk_ann_t
    - Allow user-provided list of hash bits and enable easy mi, chi2 or
      corex-based feature bit-selection.
        - mi, chi2, and corex could be run over all pairs using 1/pos 0/neg as
          the binary label and the XOR of the nodes' features as the set of
          features

# Consider

- Convolutional
    - Tried this, see branch

- Multi-layer classifiers and encoders
- Titanic dataset

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

- Explore shared libaries, optimistic dynmaic linking? Is that a thing?

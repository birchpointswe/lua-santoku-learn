# Current

- Run end to end and update defaults
- Make heterogeneous (keep spot checks the same; print label/doc nodes together)
- Create label only index
- Refactor to avoid graph.xxx entirely, extending the index apis to directly
  produce what is needed for eval/etc.
- Evaluate and spot check train/dev/test asymmetric label rankings
- Train regressor
- Evaluate and spot check train/dev/test asymmetric label selected sets
- Rewrite docs in favor of nystrom approach
- Save branch
- Purge everything not in current/active code usage, incl:
    - corex, tch, cknn, sigma, elbows, prone, knn_xxx=true helpers simhash,
      top_coherence, top_lift, optimize elbows, prone, graph module
    - essentially all unused code, saving behind a save branch

- Clean up todo.md

# Now

- Add EUR-Lex-4k test
- Can score_elbow/etc should use a bipartite multi-label/etc structure? Or just
  known binary connections? Can we do that for clustering?

- Need to write up the philosophy: using prone/spectral based on observable knn
  by reweighted with label info as a dimensionality reduction bridge, thus
  allowing classifiers/learners to recover the representation from the
  observable space.

- Check: do we need include_bits? IB=3 functionaly equivalent to state_bits=5?

- Document/define how we do thousand-label multi-label classification via
  encoder/knn search (likely need to bring back elbow method search likely)

- Add simple hamming distance cutoff to elbow optimization

- Adapt all existing encoder tests to support prone vs spectral, as well as
  explored adjacencies vs expected-as-training adjacencies

- Explore expected+cknn filtering for spectral instead of separate training
  adjacency

- Simplify spectral adjacency sampling by simply pre-creating a single big
  bridge=none knn adjacency, and then pass it to graph.adjacency as
  seed/bipartite, relying on existing cknn/etc logic. Double-check on whether
  this is actually equivalent to re-creating it every time.

- Revise/rewrite documentation for sth
- Supplementary documentation similar to sth for classification pipelines

- Support passing in an index instead of codes to clustering, which allows
  clustering based on tk_inv_t, tk_ann_t, or tk_hbi_t using a new
  tk_inv/ann/hbi_distances (batch distance) API
    - Centroid optimization disabled in this case

- Batch distance API for ann/inv indices (distance_many or similar) to avoid
  O(n*k) individual :distance() calls in diagnostics and weighted encoding

- Parallelize booleanizer and tokenizer

- Rename library and project to santoku-learn

- Additional capabilities
    - Sparse/dense PCA for dimensionality reduction (can be followed by ITQ)
    - Sparse/dense linear SVM for codebook/classifier learning

- Regression TM, supporting single and multi-output

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

# Next

- Extend tests
    - AmaonCat-13K encoder
    - snli encoder/classifier
    - qqp encoder/classifier

- Add options to add unsupervised spectral as additional feature engineering for
  classification and encoder targets. For encoder, raw features and nystrom, and
  for classifier, encoder and nystrom. Also include an option to allow raw
  features along with encoder and or nystrom features for final classifiers.

- Autoencoder

- Chores
    - Error checks on dimensions to prevent segfaults everywhere
    - Persist/load versioning or other safety measures

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

# Current

- Run end to end and update defaults
- Make heterogeneous (keep spot checks the same; print label/doc nodes together)
- Create label only index
- Asym neighborhoods lookup for train-spectral/train-predicted/dev/test against
  spectral label codes
- Spot check found label neighbors
- Fit regression predicting k from neighbors lists
- Evaluate regression-selected tags against train/dev/test
- Rewrite docs in favor of new approach
- Clean up todo.md

- TM curriculum leaning/re-allocation of capacity during training

# Now

- Allow clustering on tk_inv_t (centroid lower-bound possible?)
- Batch distance API for ann/inv indices (distance_many or similar) to avoid
  O(n*k) individual :distance() calls in diagnostics and weighted encoding
- Parallelize booleanizer and tokenizer
- Rename library and project to santoku-learn

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

# Next

- Consider explicitly NOT exposing any :destroy() functionality to lua, instead
  accomplishing lua-side explicit cleanup via =nil and collectgarbage. Then, see
  what additional cleanup c-side this enables.

- Extend tests
    - AmaonCat-13K encoder
    - snli encoder/classifier
    - qqp encoder/classifier

- Explore higher-level architectures:
    - Autoencoder
    - Triplet-loss trained encoder/regressor
    - Stacked encoders

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

- tokenizer
    - store known tokens contiguously, no need for separate mallocs/etc for each
      token. Use a single cvec as backend, storing pointers into that cvec
      insted of separately alloc'd strings. Ensure tokens are null terminated.

- Separate all lua api functions explicitly via x/x_lua variant (_lua variants
  must respect the expected stack semantics strictly); use it consistently

- tk_dsu_t
    - Full Lua/C API?

# Consider

- Titanic dataset

- Convolutional
    - Tried this, see branch

- Generative model, next token predictor
    - predict visible features from spectral codes
    - spectral embeddings from graph of ngrams as nodes, probabilities as
      weights?

- Optimize
    - Bayesian optimization

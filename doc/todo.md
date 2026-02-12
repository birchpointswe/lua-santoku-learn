# Active

- EUR-Lex XMLC pipeline
    - Train top-k regressor from predicted neighborhoods
    - Evaluate regression-selected tags against train/dev/test
    - Separate search/final datasets with different dimensionalities (search on
      top dims by threshold count, final train on full set, dynamic quantizer
      after)

- Apply absorb/pruning logic after training/finalizing to remove unused
  literals.

- Prune unused clauses and restructor/defrag for faster inference

# Next

- Embedding classification experiments (prove information retention by training
  classifiers on reduced/binary/projected representations)
    - Finish newsgroups_embedding
    - imdb_embedding
    - mnist_embedding

# Backlog

- Rename package to santoku-learn (README done, package-level pending)
- Batch distance API for ann/inv (avoid O(n*k) individual :distance() calls in
  diagnostics and weighted encoding)
- Parallelize booleanizer and tokenizer
- Allow clustering on tk_inv_t
- Clause interpretation (top features/clauses by class or overall, map back to
  intelligible names)
- Error checks on dimensions to prevent segfaults
- High-level APIs (santoku.learn.encoder, santoku.learn.classifier)

# Consider

- Additional datasets (AmazonCat-13K, SNLI, QQP)
- Convolutional input (see branch)
- Bayesian optimization

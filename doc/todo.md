# Active

- EUR-Lex XMLC pipeline
    - Evaluate with train/dev/test splits
    - Test-time inference: tokenize -> BNS select -> TM regress -> ridge
      encode -> threshold -> label IDs

- Apply absorb/pruning logic after training/finalizing to remove unused
  literals.

- Prune unused clauses and restructure/defrag for faster inference

# Next

- Embedding classification experiments (prove information retention by
  training classifiers on reduced/binary/projected representations)
    - Finish newsgroups_embedding
    - imdb_embedding
    - mnist_embedding

# Backlog

- Rename package to santoku-learn (README done, package-level pending)
- Batch distance API for ann/inv (avoid O(n*k) individual :distance()
  calls in diagnostics and weighted encoding)
- Parallelize booleanizer and tokenizer
- Allow clustering on tk_inv_t
- Clause interpretation (top features/clauses by class or overall, map
  back to intelligible names)
- Error checks on dimensions to prevent segfaults
- High-level APIs (santoku.learn.encoder, santoku.learn.classifier)

# Consider

- Additional datasets (AmazonCat-13K, SNLI, QQP)
- Convolutional input (see branch)

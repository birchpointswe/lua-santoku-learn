# Now

- Fix openmp sanitizer issues
- Replace roaring with kbtree/khash
- Matrix as extension to kvec
- ANN via multi-probe LSH
- Add learnability to tch optimization

# Consider

- Corex anchor should multiply feature MI instead of hard boost?
- Corex allow explicit anchor feature list instead of last n_hidden, handle
  multiple assigned to same latent, etc.

- Better booleanizer:
    - If threshold levels is 1, pick the middle number, if 2, use the 1/3 and
      2/3s break points, and so on. Consider selecting levels dynamically instead
      of by fixed increments. Mini clustering algorithm?

- Flip/interleave input bits should be done in TM library, not required by the
  user. It's too easy to forget the complement bits.

- Should classes in Lua always be 1 indexed? Currently they're 0 indexed.
  Indexing from 1 makes sense for multi-class, but gets odd when doing 0/1
  classification and having to increment to 1/2.

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

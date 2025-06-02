# Now

- Refactor booleanizer/thresholding
    - santoku.booleanize
        - Focused on converting a set of feature observations across many
          samples to a fixed-width bit-matrix representation
        - Accepts either dense matrix of floats and converts via thresholding or
          a table of tables of feature observations and converts via
          thresholding or multi-hot encoding
        - Returns the encoded input and a function for converting new samples
        - Takes either a matrix of floats or lua table of tables and produces:
            - A bit-matrix representing the input
            - An encoder function for converting new samples
        - User-specified `n_thresholds` for continuous features
            - `n_thresholds = 1`: median thresholding
            - `n_thresholds > 1`: percentile binning
        - For table input:
            - Optional `feature_types` table can be provided: `{ feature_name = "categorical" | "continuous" }`
            - Otherwise, features are automatically classified:
                - Numeric → continuous
                - Strings, booleans → categorical
                - Mixed types → categorical
            - Manual type annotations take precedence
            - Continuous features are binarized as above
            - Categorical features are one-hot or multi-hot encoded
    - santoku.refine
        - Focused on embedding-level bit revision/optimization
        - Takes a bitmatrix and refines it in some way, producing another bit
          matrix
        - Currently only supports tch greedy flipping
        - Could expose an auc/entropy/etc-based pruning/selection process
        - Takes a corpus of bitmaps or floats and produces a new corpus of
          bitmaps

- Bitmap clustering (k-medoids or dbscan)
- Multi-probe LSH ANN
- Graph.pairs should show modifications (seed, removed, added by phase)
- Support KF(urthest)N in addition to KNN (random zero-overlap nodes first, then
  furthest in feature space)

# Next

- Automatically handle clause numbers, feature numbers, etc. of arbitrary
  numbers (i.e. support for non-multiples of 64)

- Expose tk_bits_t API to Lua
    - Consider using tk_cvec_t

# Later

- Double-check auto-vectorization and paralellization across the board
    - Especially corex, given added branching/etc logic added
- Parallelize threshold tch, median
- Parallelize graph adj_init, to_bits, render_pairs,
- ANN via multi-probe LSH
- Hamming DBSCAN & K-Medoids

# Consider

- Add learnability to tch optimization
- Replace roaring with kbtree/khash

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

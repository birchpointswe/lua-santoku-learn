# Now

- MNIST cluster/encoder
    - Fix segfault in mnist clustering
    - Gracefully handle n_components > 1 case (seeing 2 here)
    - Support passing labels to graph to ensure edges respect classes (e.g. dont
      positively connect between diff classes or negative with same)

- Misc
    - Sparse/dense linear SVM for sanity checking all tasks

- Chore
    - Set num threads for blas in spectral, itq, and anywhere else so that
      THREADS is respected
    - When scanning margins, what should we do when multiple are best? Pick
      smallest, closest, or largest margin? Center of largest gap?

- tk_inv/ann_t
    - Nearest neighbor search
    - Persist/load to/from disk/string

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- TBHSS
    - Get working with new TM code
    - Explore classify
    - Explore encode
    - Index & search

- MNIST clustering
    - Does this even make sense? Can't we achieve the same goal by just
      statically assigning codes to each class? Does clustering add anything?

# Later

- tk_xvec_t
    - Align Lua/C APIs (currently out of sync for pvec, rvec, etc)

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- TBHSS
    - Move hyperparameter search/exploration code into TM library
    - Restrict TBHSS code to cover cli usage, pre-processing, etc.

- Misc
    - Move multiclass to pairwise transform to C
    - Generalize the patterns found in tests into:
        - santoku.tsetlin.encoder: wraps all supported encoder use-cases: graph
          or labeled data input, generating a codebook, clustering, learning
          hash functions, ann search, persist/load, etc, integrates booleanizer,
          tokenizer, etc, to reduce user-required pre-processing
        - santoku.tsetlin.classifier: similar to encoder: labeled data input
        - santoku.tsetlin.explore: TM hyperparameter exploration
             - 1+ epoch restarts (few) doing a grid search or random hill
               climbing to find best starting point, followed by longer training
               runs on the top contenders

- Chore
    - End to end code review
    - Log stats to disk(?) render with ggplot(?)

- Eval
    - Table input to entropy_stats
    - Reorganize optimize_retrieval as per optimize_clustering

- tk_graph_t
    - Parallelize transitive expansion

- TCH, ITQ
    - Parallelize

- tk_inv/ann_t
    - Parallel add?
    - Shrink/compaction

- tk_booleanizer_t
    - Mergable booleanizers, externally paralellized + merged
    - encode_sparse/dense for ivec/cvec
    - When both double and string observations found, split into two different
      features, one for continuous and the other for categorical

- tk_tokenizer_t
    - Mergable tokenizers parallelized independently
    - Proper Lua/C API and tk naming
    - Use booleanizer under the hood
    - Include text statistics as continuous/thresholded values

- Chores
    - Parallelize
    - Vectorize
    - Document

# Later

- Curriculum learning, bit pruning, etc. for encoder

- tk_tsetlin_t
    - Regression

- tk_graph_t
    - Proper Lua/C API and tk naming
    - Support querying for changes made to seed list over time (removed during
      dedupe, added by phase)

- Spectral
    - Weighted laplacian (ground truths, transitives, knn, kfn, etc.)

- tk_ann_t
    - Precompute probes

- tk_ann_t
    - Allow user-provided list of hash bits and enable easy mi, chi2 or
      corex-based feature bit-selection.
      - mi, chi2, and corex could be run over all pairs using 1/pos 0/neg as the
        binary label and the XOR of the nodes' features as the set of features

- Clustering
    - Move implementation to C with proper Lua/C APIs

- General
    - Replace most tk_malloc/realloc/numa with tk_xvec operations across
      libraries

- tk_cvec_t
    - Move dense bit operations here (hamming, xor, etc, etc)
    - Allow dense bit operations on arbitrary-length input (non-multiples of 64,
      see the filter pattern from initial TM implementation)

- tk_xvec_t
    - Generalize template to allow push/peekbase to work with composite types

- Clustering
    - K-means, k-medoids, hierarchical?

# Consider

- tk_dsu_t
    - Split from graph for other uses?
    - Full Lua/C API?

- vec/tpl.h
      Allow tk_vec_noundef to skip undefs?

- Corex
    - Should anchor multiply feature MI instead of hard-boosting it?
    - Allow explicit anchor feature list instead of last n_hidden, supporting
      multiple assigned to same latent, etc.

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

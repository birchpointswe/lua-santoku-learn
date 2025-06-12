# Now

- Supervised discrete hashing:
    - Phase 0: Form a fully connected graph and adjacency lists from
      ground-truth similar/dissimilar pairs and performing graph enrichment
      (KNN, Kruskal MST, transitive positives/negatives).
    - Phase 1: Initialize your projection matrix P by running your
      signed-normalized graph Laplacian through PRIMME to get the top K
      eigenvectors or by filling P with random normals and orthonormalizing its
      columns.
    - Phase 2: Hand-code two sparse×dense routines. One streams each sample’s
      nonzero feature indices into P to produce a dense N×K output. The other
      streams that N×K dense matrix back through the same sparsity pattern to form
      a D×K right-hand side. Parallelize both over samples with OpenMP.
    - Phase 3: Solve for P in the ridge system without ever forming XᵀX. Use Conjugate
      Gradient on each column (or a block variant) where the “A·v” call is implemented
      as a sparse×dense multiply into X and back into Xᵀ, plus adding λ·v.
      Precondition each solve with Jacobi: compute diag(A) as feature-nonzero-counts+λ
      once, invert those scalars, and scale the CG residual coordinate-wise by them. λ
      and μ are hyperparams on a positive log scale (λ from 1e-4 to 1e3 and μ from
      1e-2 to 1e4).
    - Phase 4: Compute V=X·P with your sparse×dense kernel. For each bit-column,
      scan V and greedily flip signs in B (initialized to the sign of V) to
      reduce reconstruction error plus a Laplacian smoothness penalty. Repeat
      flips until no further gain or you hit an iteration limit.
    - Phase 5: After each ridge-solve and flip cycle, evaluate the combined
      objective on B and V and stop when improvements fall below a small
      threshold.
    - Phase 6: Deploy by hashing new sparse samples through P: iterate their
      nonzeros, accumulate into a K-vector, then take the sign of each entry as
      your K-bit code. Optionally, train K sparse binary classifiers on your final
      B if you need non-linear bit boundaries.
    --
    - At each SDH iteration compute V=X·P, threshold to B' for both training and
      testing samples, running encoding_accuracy and optimize_retrieval to guage
      performance. Optionally, move on to the K-TMs/classifiers stage if the
      linear encoder using P is insufficient. When using the separate classifier
      stage, μ can be dialed up and λ down to prefer laplacian over
      reconstruction.
    --
    - For graphs with a single node type, X is the expected format for a
      binarized sparse feature matrix (e.g. list of set bits). For graphs with
      multiple node types, each with different feature spaces, we represent all
      nodes in X using a globally concatenated binary feature spaces (e.g. if each
      of our 10 node types have 500 unique features, our combined X would have
      5000 columns). Nodes without any features are still considered as part of X
      and considered as part of the total N throughout SDH.

- tk_inv/ann_t
    - Nearest neighbor search
    - Persist/load to/from disk/string

- Chore
    - Sanity check older version to confirm encoder accuracy is as expected
    - Set num threads for blas in spectral, itq, and anywhere else so that
      THREADS is respected

- Eval
    - Table input to entropy_stats and optimize_retrieval
    - Reorganize optimize_retrieval as per optimize_clustering

# Next

- tk_xvec_t
    - Align Lua/C APIs (currently out of sync for pvec, rvec, etc)

- tk_cvec_t
    - Use for Corex output, TM input/output, ANN input/output, TCH input/output
    - Standardize bits APIs between cvec and ivec
        - Rename to tk_ivec/tk_cvec_bits_xxx
        - Convert between ivec/cvec with tk_ivec_bits_cvec and tk_cvec_bits_ivec
        - Special ivec_bits_xxx methods
            - top_chi2/mi, score chi2/mi, filter, extend, cvec
        - Special cvec_bitx_xxx methods
            - top_entropy, score_entropy, flip_interleave, filter, extend, ivec

- tk_xxmap/set_t:
    - Templatize over khash/kbtree
    - Proper Lua/C API like tk_xvec_t

- TBHSS
    - Rename
    - Integrate latest TM updates
    - Explore classify
    - Explore encode
    - Index & search
    - Move hyperparameter search/exploration code into TM library
    - Restrict TBHSS code to cover cli usage, pre-processing, etc.

- tk_graph_t
    - Parallelize transitive expansion

- TCH, ITQ
    - Parallelize

- tk_inv/ann_t
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

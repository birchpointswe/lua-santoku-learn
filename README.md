# Santoku Tsetlin

Machine learning library with Tsetlin machines, clustering, indexing, and embedding components.

## API Reference

### santoku.corex

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `corex.create` | `table: {visible, hidden, lam?, spa?, tmin?, ttc?, anchor?, tile_s?, tile_v?, threads?}` | `userdata: corex_t` | Create CorEx model with visible/hidden dimensions |
| `corex.load` | `string|userdata: data, boolean?: from_string, number?: threads` | `userdata: corex_t` | Load CorEx model from file path or serialized data |
| `corex:train` | `table: {corpus, samples, iterations, each?}` | `nil` | Train CorEx model on corpus data |
| `corex:compress` | `userdata: set_bits, number?: n_samples` | `userdata: ivec_t` | Compress data using trained model |
| `corex:top_visible` | `number: top_k` | `userdata: ivec_t` | Extract top-k visible features |
| `corex:visible` | `-` | `number` | Return number of visible dimensions |
| `corex:hidden` | `-` | `number` | Return number of hidden dimensions |
| `corex:persist` | `string?: filepath` | `string?` | Save model to file or return serialized data |
| `corex:destroy` | `-` | `nil` | Free model memory |

### santoku.tsetlin

Main module providing Tsetlin machine classification and encoding.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tsetlin.create` | `string: type, table: config` | `userdata: tsetlin_t` | Create Tsetlin machine ("classifier" or "encoder") |
| `tsetlin.load` | `string|userdata: data, boolean?: from_string, boolean?: read_state` | `userdata: tsetlin_t` | Load Tsetlin machine from file or serialized data |
| `tsetlin:train` | `table: training_data` | `nil` | Train Tsetlin machine on provided data |
| `tsetlin:predict` | `string: patterns, number: n_samples` | `string` | Predict on bit-encoded patterns |
| `tsetlin:persist` | `string?: filepath, boolean?: persist_state` | `string?` | Save model to file or return serialized data |
| `tsetlin:type` | `-` | `string` | Return machine type ("classifier" or "encoder") |
| `tsetlin:destroy` | `-` | `nil` | Free machine memory |

### santoku.tsetlin.ann

Approximate nearest neighbor index with binary encodings.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `ann.create` | `number: features, number?: threads` | `userdata: ann_t` | Create ANN index with binary feature count and thread pool |
| `ann.load` | `string\|userdata: data, boolean?: from_string, number?: threads` | `userdata: ann_t` | Load ANN index from file path or serialized data |
| `ann:add` | `string: data, number\|userdata: base_id\|ids, number?: n_nodes` | `nil` | Add binary vectors with node IDs |
| `ann:remove` | `number: id` | `nil` | Remove node by ID |
| `ann:get` | `number: id` | `string?` | Get binary vector by ID |
| `ann:neighborhoods` | `number?: k, number?: eps, number?: min, boolean?: mutual` | `userdata: hoods_t, userdata: ivec_t` | Find k-nearest neighborhoods within Hamming distance eps |
| `ann:neighbors` | `number\|string: id\|vector, number?: knn, number?: eps, userdata?: out` | `userdata: pvec_t` | Find neighbors for ID or vector within distance eps |
| `ann:size` | - | `number` | Return number of indexed nodes |
| `ann:threads` | - | `number` | Return thread count |
| `ann:features` | - | `number` | Return feature count |
| `ann:persist` | `string\|boolean: filepath\|to_string` | `string?` | Save to file or return serialized data |
| `ann:destroy` | - | `nil` | Free index memory |
| `ann:shrink` | - | `nil` | Reduce memory usage |
| `ann:ids` | - | `userdata: ivec_t` | Return all node IDs |

### santoku.tsetlin.inv

Inverted index for sparse binary features.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `inv.create` | `table: {features}` | `userdata: inv_t` | Create inverted index with feature count |
| `inv.load` | `string\|userdata: data, boolean?: from_string` | `userdata: inv_t` | Load from file or serialized data |
| `inv:add` | `userdata: features, userdata: ids` | `nil` | Add sparse feature vectors with node IDs |
| `inv:remove` | `number: id` | `nil` | Remove node by ID |
| `inv:get` | `number: id` | `table?` | Get feature vector by ID |
| `inv:neighborhoods` | `number?: k, number?: eps, number?: min, string?: cmp, number?: alpha, number?: beta, boolean?: mutual` | `userdata: hoods_t, userdata: ivec_t` | Find neighborhoods using similarity metrics |
| `inv:neighbors` | `number\|userdata: id\|vector, number?: knn, number?: eps, string?: cmp, number?: alpha, number?: beta, userdata?: out` | `userdata: rvec_t` | Find neighbors with similarity scores |
| `inv:distance` | `number: id1, number: id2, string?: cmp, number?: alpha, number?: beta` | `number` | Calculate distance between nodes |
| `inv:similarity` | `userdata: vec1, userdata: vec2, string?: cmp, number?: alpha, number?: beta` | `number` | Calculate similarity between vectors |
| `inv:size` | - | `number` | Return number of nodes |
| `inv:features` | - | `number` | Return feature count |
| `inv:persist` | `string\|boolean: filepath\|to_string` | `string?` | Save to file or serialized data |
| `inv:destroy` | - | `nil` | Free index memory |
| `inv:ids` | - | `userdata: ivec_t` | Return all node IDs |

### santoku.tsetlin.hbi

Hamming ball index for binary similarity search.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `hbi.create` | `number: features, number?: threads` | `userdata: hbi_t` | Create HBI with feature count and threads |
| `hbi.load` | `string\|userdata: data, boolean?: from_string, number?: threads` | `userdata: hbi_t` | Load from file or serialized data |
| `hbi:add` | `string: data, number\|userdata: base_id\|ids, number?: n_nodes` | `nil` | Add binary data with node IDs |
| `hbi:remove` | `number: id` | `nil` | Remove node by ID |
| `hbi:get` | `number: id` | `string?` | Get binary data by ID |
| `hbi:neighborhoods` | `number?: k, number?: eps, number?: min, boolean?: mutual` | `userdata: hoods_t, userdata: ivec_t` | Find neighborhoods within Hamming radius |
| `hbi:neighbors` | `number\|string: id\|vector, number?: knn, number?: eps, userdata?: out` | `userdata: pvec_t` | Find neighbors within Hamming distance |
| `hbi:size` | - | `number` | Return indexed node count |
| `hbi:threads` | - | `number` | Return thread count |
| `hbi:features` | - | `number` | Return feature count |
| `hbi:persist` | `string\|boolean: filepath\|to_string` | `string?` | Save to file or serialized data |
| `hbi:destroy` | - | `nil` | Free memory |
| `hbi:shrink` | - | `nil` | Reduce memory usage |
| `hbi:ids` | - | `userdata: ivec_t` | Return node IDs |

### santoku.tsetlin.dataset

Dataset utilities for machine learning tasks.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `dataset.read_snli_pairs` | `string: filepath, number?: max, boolean?: single_split` | `table: dataset` | Read SNLI sentence pairs for entailment tasks |
| `dataset.split_snli_pairs` | `table: dataset, number?: ratio` | `table: train, table?: test` | Split SNLI dataset into train/test sets |
| `dataset.read_binary_mnist` | `string: filepath, number: n_features, number?: max, number?: class_max` | `table: dataset` | Read binary MNIST data with feature and sample limits |
| `dataset.split_binary_mnist` | `table: dataset, number: ratio` | `table: train, table?: test` | Split binary MNIST into train/test sets |
| `dataset.random_pairs` | `userdata: ids, number?: edges_per_node` | `userdata: pvec_t` | Generate random node pairs for graph construction |
| `dataset.anchor_pairs` | `userdata: ids, number?: n_anchors` | `userdata: pvec_t` | Generate anchor-based node pairs |
| `dataset.classes_index` | `userdata: ids, userdata: classes` | `userdata: inv_t` | Create class-based inverted index |
| `dataset.multiclass_pairs` | `userdata: ids, userdata: labels, number?: n_anchors_pos, number?: n_anchors_neg, userdata?: index, number?: eps_pos, number?: eps_neg` | `userdata: pvec_t, userdata: pvec_t` | Generate positive and negative pairs for classification |
| `dataset.read_imdb` | `string: directory, number?: max` | `table: dataset` | Read IMDB movie reviews dataset |
| `dataset.split_imdb` | `table: dataset, number: ratio` | `table: train, table: test` | Split IMDB dataset |
| `dataset.read_glove` | `string: filepath, number?: max` | `table: embeddings` | Read GloVe word embeddings |

### santoku.tsetlin.evaluator

Model evaluation and metrics computation.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `evaluator.class_accuracy` | `userdata: predicted, userdata: expected, number: n_samples, number: n_classes, number?: threads` | `table: metrics` | Calculate per-class precision, recall, F1 scores |
| `evaluator.encoding_accuracy` | `userdata|string: predicted, userdata|string: expected, number: n_samples, number: n_hidden, number?: threads` | `table: metrics` | Calculate bit-wise encoding accuracy and statistics |
| `evaluator.clustering_accuracy` | `userdata: assignments, userdata: ids, userdata: pos, userdata: neg, number?: threads` | `table: accuracy` | Evaluate clustering with positive/negative pairs |
| `evaluator.optimize_retrieval` | `table: {codes, ids?, pos, neg, n_dims, threads?, each?}` | `table: results` | Optimize retrieval threshold for best accuracy |
| `evaluator.optimize_clustering` | `table: {index, ids?, pos, neg, min_pts?, assign_noise?, min_margin, max_margin, threads?, each?}` | `table: results, userdata: ids, userdata: assignments, number: n_clusters` | Optimize clustering parameters |
| `evaluator.entropy_stats` | `userdata\|string: codes, number: n_samples, number: n_hidden, number?: threads` | `table: entropy` | Calculate per-bit entropy statistics |
| `evaluator.auc` | `userdata: ids, string\|userdata: codes, userdata: pos, userdata: neg, number: n_hidden, string?: mask, number?: threads` | `number` | Calculate AUC for similarity ranking |

### santoku.tsetlin.graph

Graph construction and adjacency computation.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `graph.create` | `table: {edges?, index, cmp?, cmp_alpha?, cmp_beta?, weight_eps?, flip_at?, neg_scale?, sigma_k?, sigma_scale?, knn?, knn_min?, knn_cache?, knn_eps?, bridge?, threads?, each?}` | `userdata: graph_t` | Create graph from index with various weighting schemes |
| `graph:pairs` | - | `userdata: pvec_t, userdata: dvec_t` | Get edge pairs and weights |
| `graph:adjacency` | - | `userdata: ivec_t, userdata: ivec_t, userdata: ivec_t, userdata: dvec_t` | Get CSR adjacency representation: node IDs, offsets, neighbors, weights |

### santoku.tsetlin.itq

Iterative quantization for binary embeddings.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `itq.encode` | `table: {codes, n_dims, iterations?, tolerance?, threads?, each?}` | `userdata: ivec_t` | Apply ITQ to continuous embeddings |
| `itq.sign` | `table: {codes, n_dims}` | `userdata: ivec_t` | Simple sign-based binarization |
| `itq.median` | `table: {codes, n_dims}` | `userdata: ivec_t` | Median-based binarization |

### santoku.tsetlin.spectral

Spectral graph embedding.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `spectral.encode` | `table: {ids, offsets, neighbors, weights, n_hidden, threads?, normalized?, eps_primme?, each?}` | `userdata: ivec_t, userdata: dvec_t` | Compute spectral embeddings from graph adjacency |

### santoku.tsetlin.tch

Greedy coordinate ascent for graph adjacency alignment via bit flipping.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tch.refine` | `table: {codes, ids, offsets, neighbors, weights, n_dims, each?}` | `userdata: ivec_t` | Refine binary codes by minimizing graph energy |

### santoku.tsetlin.booleanizer

Feature booleanization for continuous and categorical data.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `booleanizer.create` | `table?: {n_thresholds?, continuous?, categorical?}` | `userdata: booleanizer_t` | Create booleanizer with optional settings |
| `booleanizer.load` | `string: data, boolean?: from_string` | `userdata: booleanizer_t` | Load booleanizer from file or string |
| `booleanizer:observe` | `feature, value OR dvec, n_dims, id_dim0?` | `integer, integer?` | Observe feature values for training |
| `booleanizer:encode` | `sample_id, feature, value OR dvec, n_dims, id_dim0?, out?` | `userdata: ivec_t` | Encode features to binary representation |
| `booleanizer:finalize` | `-` | `nil` | Finalize booleanizer after observing all data |
| `booleanizer:features` | `-` | `integer` | Return number of features |
| `booleanizer:bits` | `-` | `integer` | Return number of output bits |
| `booleanizer:bit` | `feature` | `integer?` | Get feature ID for given feature |
| `booleanizer:restrict` | `userdata: ivec_t` | `nil` | Restrict to subset of features |
| `booleanizer:persist` | `string|boolean: filepath|to_string` | `string?` | Save to file or return serialized data |
| `booleanizer:destroy` | `-` | `nil` | Free booleanizer memory |

### santoku.tsetlin.tokenizer

Text tokenization with n-grams, skip-grams, and character grams.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tokenizer.create` | `table: {max_vocab?, max_df?, min_df?, max_len, min_len, max_run, ngrams, cgrams_min, cgrams_max, skips, negations, align?}` | `userdata: tokenizer_t` | Create tokenizer with vocabulary and n-gram parameters |
| `tokenizer.load` | `string: data, number?: threads, boolean?: from_string` | `userdata: tokenizer_t` | Load tokenizer from file or string |
| `tokenizer:train` | `table: {corpus}` | `nil` | Train tokenizer on document corpus |
| `tokenizer:tokenize` | `string\|table: text, userdata?: out` | `userdata: ivec_t` | Tokenize text to feature indices |
| `tokenizer:parse` | `string\|table: text` | `table: tokens` | Parse text to human-readable tokens |
| `tokenizer:features` | - | `number` | Return total feature count (aligned) |
| `tokenizer:finalize` | - | `nil` | Finalize vocabulary after training |
| `tokenizer:restrict` | `userdata: top_vocab` | `nil` | Restrict to top vocabulary items |
| `tokenizer:index` | - | `table: vocab` | Return vocabulary index mapping |
| `tokenizer:persist` | `string?: filepath` | `string?` | Save tokenizer or return serialized data |
| `tokenizer:destroy` | - | `nil` | Free tokenizer memory |

## Data Structures

### Similarity Metrics (inv module)

- `jaccard`: Jaccard similarity coefficient
- `overlap`: Overlap coefficient
- `tversky`: Tversky index with alpha/beta parameters
- `dice`: Dice coefficient

### DSU (Disjoint Set Union) Operations

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_dsu_init` | `dsu: tk_dsu_t*, ids: tk_ivec_t*` | `nil` | Initialize DSU with ID vector |
| `tk_dsu_free` | `dsu: tk_dsu_t*` | `nil` | Free DSU memory |
| `tk_dsu_find` | `dsu: tk_dsu_t*, uid: int64_t` | `int64_t` | Find root component of node |
| `tk_dsu_findx` | `dsu: tk_dsu_t*, uidx: int64_t` | `int64_t` | Find root by index |
| `tk_dsu_union` | `dsu: tk_dsu_t*, x: int64_t, y: int64_t` | `nil` | Union two components |
| `tk_dsu_unionx` | `dsu: tk_dsu_t*, xidx: int64_t, yidx: int64_t` | `nil` | Union by indices |
| `tk_dsu_components` | `dsu: tk_dsu_t*` | `int64_t` | Get component count |

### Clustering Operations (cluster.h)

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_cluster_dsu` | `hbi: tk_hbi_t*, ann: tk_ann_t*, rtmp: tk_rvec_t*, ptmp: tk_pvec_t*, margin: uint64_t, min_pts: uint64_t, assign_noise: bool, ids: tk_ivec_t*, assignments: tk_ivec_t*, ididx: tk_iumap_t*, n_clusters: uint64_t*` | `nil` | Perform DSU-based clustering with density constraints |

### ITQ Operations (itq.h)

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_itq_encode` | `L: lua_State*, codes: tk_dvec_t*, n_dims: uint64_t, max_iterations: uint64_t, tolerance: double, i_each: int, n_threads: uint` | `nil` | Perform iterative quantization encoding |
| `tk_itq_sign` | `out: tk_ivec_t*, X: double*, N: uint64_t, K: uint64_t` | `nil` | Extract positive elements from matrix |
| `tk_itq_median` | `out: tk_ivec_t*, X: double*, N: uint64_t, K: uint64_t` | `nil` | Extract above-median elements |
| `tk_itq_center` | `M: double*, N: size_t, K: size_t` | `nil` | Center matrix columns |

### TCH Operations (tch.h)

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_tch_refine` | `L: lua_State*, codes: tk_ivec_t*, uids: tk_ivec_t*, adj_offset: tk_ivec_t*, adj_data: tk_ivec_t*, adj_weights: tk_dvec_t*, n_dims: uint64_t, i_each: int` | `nil` | Refine binary codes via coordinate ascent on graph adjacency |

## Constants

### HBI Constants (hbi.h)

- `TK_HBI_BITS`: 32 - Maximum bits for hash codes
- `TK_HBI_MT`: "tk_hbi_t" - Metatable name
- `TK_HBI_EPH`: "tk_hbi_eph" - Ephemeron table name

### Graph Constants (graph.h)

- `TK_GRAPH_MT`: "tk_graph_t" - Metatable name
- `TK_GRAPH_EPH`: "tk_graph_eph" - Ephemeron table name

### Tokenizer Constants (tokenizer.c)

- `MT_TOKENIZER`: "santoku_tokenizer" - Metatable name


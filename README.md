# Santoku Tsetlin

Machine learning library with Tsetlin machines, clustering, indexing, and embedding components.

## API Reference

### `santoku.tsetlin`

High-level Tsetlin machine interface for classification and encoding.

#### Classifier API

##### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `tsetlin.classifier` | `table: {features, classes, clauses, clause_tolerance, clause_maximum, specificity, negative?, target?, state?, threads?}` | `classifier` | Create Tsetlin classifier |
| `tsetlin.optimize_classifier` | `table: {features, classes, negative?, clauses?, clause_tolerance?, clause_maximum?, target?, specificity?, search_rounds?, search_trials?, search_iterations?, search_patience?, search_dev?, search_metric, each?, final_iterations?, samples?, problems?, solutions?, threads?}` | `classifier` | Optimize and train classifier |

##### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `classifier:train` | `table: {samples, problems, solutions, iterations, each?}` | `-` | Train classifier on labeled data. Note: `problems` can be passed as either a string or tk_cvec_t |

#### Encoder API

##### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `tsetlin.encoder` | `table: {visible, hidden, clauses, clause_tolerance, clause_maximum, specificity, target?, state?, threads?}` | `encoder` | Create Tsetlin encoder |
| `tsetlin.optimize_encoder` | `table: {visible, hidden, clauses?, clause_tolerance?, clause_maximum?, target?, specificity?, search_rounds?, search_trials?, search_iterations?, search_patience?, search_dev?, search_metric, each?, final_iterations?, sentences?, codes?, samples?, threads?}` | `encoder` | Optimize and train encoder |

##### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `encoder:train` | `table: {samples, sentences, codes, iterations, each?}` | `-` | Train encoder on binary patterns. Note: Both `sentences` and `codes` can be passed as either strings or tk_cvec_t objects |

#### Common Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `machine:predict` | `string: patterns, number: n_samples` | `string` | Make predictions |
| `machine:persist` | `string?: filepath, boolean?: persist_state` | `string?` | Save model |
| `machine:type` | `-` | `string` | Return "classifier" or "encoder" |
| `machine:destroy` | `-` | `-` | Free memory |

#### Common Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `tsetlin.load` | `string/data: path/serialized, boolean?: from_string, boolean?: read_state` | `machine` | Load saved machine |

#### Module Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `tsetlin.align` | `8` | Bit alignment boundary for features |

### `santoku.tsetlin.ann`

Approximate nearest neighbor index with binary encodings.

#### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `ann.create` | `table: {expected_size, features, bucket_target?: 30, probe_radius?: 2, threads?}` | `tk_ann_t` | Create ANN index with LSH parameters |
| `ann.load` | `string/tk_ann_t: data, boolean?: from_string, number?: threads` | `tk_ann_t` | Load ANN index from file |

#### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `ann:add` | `string: data, number/tk_ivec_t: base_id/ids, number?: n_nodes` | `-` | Add binary vectors |
| `ann:remove` | `number: id` | `-` | Remove node by ID |
| `ann:get` | `number: id` | `string?` | Get binary vector by ID |
| `ann:neighborhoods` | `number?: k, number?: eps, number?: min, boolean?: mutual` | `tk_ann_hoods_t, tk_ivec_t` | Find k-nearest neighborhoods |
| `ann:neighbors` | `number/string: id/vector, number?: knn, number?: eps, tk_pvec_t?: out` | `tk_pvec_t` | Find neighbors within distance |
| `ann:size` | `-` | `number` | Get indexed node count |
| `ann:threads` | `-` | `number` | Get thread count |
| `ann:features` | `-` | `number` | Get feature count |
| `ann:persist` | `string/boolean: filepath/to_string` | `string?` | Save index |
| `ann:destroy` | `-` | `-` | Free memory |
| `ann:shrink` | `-` | `-` | Reduce memory usage |
| `ann:ids` | `-` | `tk_ivec_t` | Get all node IDs |

### `santoku.tsetlin.inv`

Inverted index for sparse binary features.

#### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `inv.create` | `table: {features, weights?: tk_dvec_t, ranks?: tk_ivec_t, n_ranks?, rank_decay_window?, rank_decay_sigma?, rank_decay_floor?, threads?}` | `tk_inv_t` | Create inverted index with ranking |
| `inv.load` | `string/tk_inv_t: data, boolean?: from_string, number?: threads` | `tk_inv_t` | Load from file |

#### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `inv:add` | `tk_ivec_t: features, tk_ivec_t: ids` | `-` | Add sparse feature vectors |
| `inv:remove` | `number: id` | `-` | Remove node by ID |
| `inv:get` | `number: id` | `table?` | Get feature vector |
| `inv:neighborhoods` | `number?: k, number?: eps, number?: min, string?: cmp, number?: alpha, number?: beta, boolean?: mutual` | `tk_inv_hoods_t, tk_ivec_t` | Find neighborhoods |
| `inv:neighbors` | `number/tk_ivec_t: id/vector, number?: knn, number?: eps, string?: cmp, number?: alpha, number?: beta, tk_rvec_t?: out` | `tk_rvec_t` | Find neighbors |
| `inv:distance` | `number: id1, number: id2, string?: cmp, number?: alpha, number?: beta` | `number` | Calculate distance |
| `inv:similarity` | `tk_ivec_t: vec1, tk_ivec_t: vec2, string?: cmp, number?: alpha, number?: beta` | `number` | Calculate similarity |
| `inv:size` | `-` | `number` | Get node count |
| `inv:features` | `-` | `number` | Get feature count |
| `inv:persist` | `string/boolean: filepath/to_string` | `string?` | Save index |
| `inv:destroy` | `-` | `-` | Free memory |
| `inv:ids` | `-` | `tk_ivec_t` | Get all node IDs |

### `santoku.tsetlin.hbi`

Hamming ball index for binary similarity search.

#### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `hbi.create` | `table: {features, threads?}` | `tk_hbi_t` | Create HBI index |
| `hbi.load` | `string/tk_hbi_t: data, boolean?: from_string, number?: threads` | `tk_hbi_t` | Load from file |

#### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `hbi:add` | `string/tk_cvec_t: data, number/tk_ivec_t: base_id/ids, number?: n_nodes` | `-` | Add binary data. Note: `data` can be passed as either a string or tk_cvec_t |
| `hbi:remove` | `number/tk_ivec_t: id/ids` | `-` | Remove node(s) by ID |
| `hbi:keep` | `number/tk_ivec_t: id/ids` | `-` | Keep only specified node(s), removing all others |
| `hbi:get` | `number/tk_ivec_t: id/ids, tk_cvec_t?: out, number?: dest_sample, number?: dest_stride` | `tk_cvec_t` | Get binary data. Supports single ID or multiple IDs with optional packed output control |
| `hbi:neighborhoods` | `number?: k, number?: eps, number?: min, boolean?: mutual` | `tk_hbi_hoods_t, tk_ivec_t` | Find neighborhoods for all nodes |
| `hbi:neighborhoods_by_ids` | `tk_ivec_t: ids, number?: k, number?: eps, number?: min, boolean?: mutual` | `tk_hbi_hoods_t, tk_ivec_t` | Find neighborhoods for specific node IDs |
| `hbi:neighborhoods_by_vecs` | `tk_cvec_t: vectors, number?: k, number?: eps, number?: min` | `tk_hbi_hoods_t, tk_ivec_t` | Find neighborhoods for query vectors |
| `hbi:neighbors` | `number/string/tk_cvec_t: id/vector, number?: knn, number?: eps, tk_pvec_t?: out` | `tk_pvec_t` | Find neighbors. Note: vector can be string or tk_cvec_t |
| `hbi:similarity` | `number: uid0, number: uid1` | `number` | Calculate similarity between two nodes (0 to 1) |
| `hbi:distance` | `number: uid0, number: uid1` | `number` | Calculate normalized Hamming distance between two nodes (0 to 1) |
| `hbi:size` | `-` | `number` | Get node count |
| `hbi:threads` | `-` | `number` | Get thread count |
| `hbi:features` | `-` | `number` | Get feature count |
| `hbi:persist` | `string/boolean: filepath/to_string` | `string?` | Save index |
| `hbi:destroy` | `-` | `-` | Free memory |
| `hbi:shrink` | `-` | `-` | Reduce memory |
| `hbi:ids` | `-` | `tk_ivec_t` | Get all node IDs |

### `santoku.tsetlin.dataset`

Dataset utilities for machine learning tasks.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `dataset.read_binary_mnist` | `string: filepath, number: n_features, number?: max` | `table: dataset` | Read binary MNIST data with feature and sample limits |
| `dataset.split_binary_mnist` | `table: dataset, number: ratio` | `table: train, table?: test` | Split binary MNIST into train/test sets |
| `dataset.read_imdb` | `string: directory, number?: max` | `table: dataset` | Read IMDB movie reviews dataset |
| `dataset.split_imdb` | `table: dataset, number: ratio` | `table: train, table: test` | Split IMDB dataset |
| `dataset.random_pairs` | `tk_ivec_t: ids, number?: edges_per_node` | `tk_pvec_t` | Generate random node pairs for graph construction |
| `dataset.anchor_pairs` | `tk_ivec_t: ids, number?: n_anchors` | `tk_pvec_t` | Generate anchor-based node pairs |
| `dataset.classes_index` | `tk_ivec_t: ids, tk_ivec_t: classes` | `tk_inv_t` | Create class-based inverted index |
| `dataset.multiclass_pairs` | `tk_ivec_t: ids, tk_ivec_t: labels, number?: n_anchors_pos, number?: n_anchors_neg, tk_inv_t?: index, number?: eps_pos, number?: eps_neg` | `tk_pvec_t, tk_pvec_t` | Generate positive and negative pairs for classification |
| `dataset.star_hoods` | `tk_ivec_t: ids, hoods` | `tk_pvec_t` | Convert neighborhoods to edge pairs (star topology) |

### `santoku.tsetlin.evaluator`

Model evaluation and metrics computation.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `evaluator.class_accuracy` | `tk_ivec_t: predicted, tk_ivec_t: expected, number: n_samples, number: n_classes, number?: threads` | `table: metrics` | Calculate per-class precision, recall, F1 scores |
| `evaluator.encoding_accuracy` | `tk_ivec_t/string: predicted, tk_ivec_t/string: expected, number: n_samples, number: n_hidden, number?: threads` | `table: metrics` | Calculate bit-wise encoding accuracy and statistics |
| `evaluator.clustering_accuracy` | `tk_ivec_t: assignments, tk_ivec_t: ids, tk_pvec_t: pos, tk_pvec_t: neg, number?: threads` | `table: accuracy` | Evaluate clustering with positive/negative pairs |
| `evaluator.optimize_retrieval` | `table: {codes, ids?, pos, neg, n_dims, threads?, each?}` | `table: results` | Optimize retrieval threshold for best accuracy |
| `evaluator.optimize_clustering` | `table: {index, ids?, pos, neg, min_pts?, assign_noise?, min_margin, max_margin, threads?, each?}` | `table: results, tk_ivec_t: ids, tk_ivec_t: assignments, number: n_clusters` | Optimize clustering parameters |
| `evaluator.entropy_stats` | `tk_ivec_t/string: codes, number: n_samples, number: n_hidden, number?: threads` | `table: entropy` | Calculate per-bit entropy statistics |
| `evaluator.auc` | `tk_ivec_t: ids, string/tk_ivec_t: codes, tk_pvec_t: pos, tk_pvec_t: neg, number: n_hidden, string?: mask, number?: threads` | `number` | Calculate AUC for similarity ranking |

### `santoku.tsetlin.graph`

Graph construction and adjacency computation.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `graph.create` | `table: {edges?, index, cmp?, cmp_alpha?, cmp_beta?, weight_eps?, flip_at?, neg_scale?, sigma_k?, sigma_scale?, knn?, knn_min?, knn_cache?, knn_eps?, bridge?, threads?, each?}` | `tk_graph_t` | Create graph from index with various weighting schemes |
| `graph:pairs` | `-` | `tk_pvec_t, tk_dvec_t` | Get edge pairs and weights |
| `graph:adjacency` | `-` | `tk_ivec_t, tk_ivec_t, tk_ivec_t, tk_dvec_t` | Get CSR adjacency representation: node IDs, offsets, neighbors, weights |

### `santoku.tsetlin.itq`

Iterative quantization for binary embeddings.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `itq.encode` | `table: {codes, n_dims, iterations?, tolerance?, threads?, each?}` | `tk_ivec_t` | Apply ITQ to continuous embeddings |
| `itq.sign` | `table: {codes, n_dims}` | `tk_ivec_t` | Simple sign-based binarization |
| `itq.median` | `table: {codes, n_dims}` | `tk_ivec_t` | Median-based binarization |

### `santoku.tsetlin.spectral`

Spectral graph embedding.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `spectral.encode` | `table: {ids, offsets, neighbors, weights, n_hidden, threads?, normalized?, eps_primme?, each?}` | `tk_ivec_t, tk_dvec_t` | Compute spectral embeddings from graph adjacency |

### `santoku.tsetlin.tch`

Greedy coordinate ascent for graph adjacency alignment via bit flipping.

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tch.refine` | `table: {codes, ids, offsets, neighbors, weights, scale?, n_dims, threads?, each?}` | `tk_ivec_t` | Refine binary codes by minimizing graph energy. The optional `scale` parameter provides degree normalization factors for weighted adjacency |

### `santoku.tsetlin.booleanizer`

Feature booleanization for continuous and categorical data.

#### Module Functions

| Function | Arguments | Returns | Description |
|----------|-----------|---------|-------------|
| `booleanizer.create` | `table?: {n_thresholds?, continuous?, categorical?}` | `tk_booleanizer_t` | Create booleanizer |
| `booleanizer.load` | `string: data, boolean?: from_string` | `tk_booleanizer_t` | Load from file |

#### Instance Methods

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `booleanizer:observe` | `feature, value OR dvec, n_dims, id_dim0?` | `integer, integer?` | Observe feature values |
| `booleanizer:encode` | `sample_id, feature, value OR dvec, n_dims, id_dim0?, out?` | `tk_ivec_t` | Encode to binary |
| `booleanizer:finalize` | `-` | `-` | Finalize after training |
| `booleanizer:features` | `-` | `integer` | Get feature count |
| `booleanizer:bits` | `-` | `integer` | Get output bit count |
| `booleanizer:bit` | `feature` | `integer?` | Get feature ID |
| `booleanizer:restrict` | `tk_ivec_t` | `-` | Restrict features |
| `booleanizer:persist` | `string/boolean: filepath/to_string` | `string?` | Save booleanizer |
| `booleanizer:destroy` | `-` | `-` | Free memory |

### `santoku/tsetlin/dsu.h`

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_dsu_init` | `dsu: tk_dsu_t*, ids: tk_ivec_t*` | `-` | Initialize DSU with ID vector |
| `tk_dsu_free` | `dsu: tk_dsu_t*` | `-` | Free DSU memory |
| `tk_dsu_find` | `dsu: tk_dsu_t*, uid: int64_t` | `int64_t` | Find root component of node |
| `tk_dsu_findx` | `dsu: tk_dsu_t*, uidx: int64_t` | `int64_t` | Find root by index |
| `tk_dsu_union` | `dsu: tk_dsu_t*, x: int64_t, y: int64_t` | `-` | Union two components |
| `tk_dsu_unionx` | `dsu: tk_dsu_t*, xidx: int64_t, yidx: int64_t` | `-` | Union by indices |
| `tk_dsu_components` | `dsu: tk_dsu_t*` | `int64_t` | Get component count |

### `santoku/tsetlin/cluster.h`

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_cluster_dsu` | `hbi: tk_hbi_t*, ann: tk_ann_t*, rtmp: tk_rvec_t*, ptmp: tk_pvec_t*, margin: uint64_t, min_pts: uint64_t, assign_noise: bool, ids: tk_ivec_t*, assignments: tk_ivec_t*, ididx: tk_iumap_t*, n_clusters: uint64_t*` | `-` | Perform DSU-based clustering with density constraints |

### `santoku/tsetlin/itq.h`

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_itq_encode` | `L: lua_State*, codes: tk_dvec_t*, n_dims: uint64_t, max_iterations: uint64_t, tolerance: double, i_each: int, n_threads: uint` | `-` | Perform iterative quantization encoding |
| `tk_itq_sign` | `out: tk_ivec_t*, X: double*, N: uint64_t, K: uint64_t` | `-` | Extract positive elements from matrix |
| `tk_itq_median` | `out: tk_ivec_t*, X: double*, N: uint64_t, K: uint64_t` | `-` | Extract above-median elements |
| `tk_itq_center` | `M: double*, N: size_t, K: size_t` | `-` | Center matrix columns |

### `santoku/tsetlin/tch.h`

| Function | Arguments | Returns | Description |
|----------|------------|---------|-------------|
| `tk_tch_refine` | `L: lua_State*, codes: tk_ivec_t*, uids: tk_ivec_t*, adj_offset: tk_ivec_t*, adj_data: tk_ivec_t*, adj_weights: tk_dvec_t*, n_dims: uint64_t, i_each: int` | `-` | Refine binary codes via coordinate ascent on graph adjacency |

## License

MIT License

Copyright 2025 Birch Point SWE

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

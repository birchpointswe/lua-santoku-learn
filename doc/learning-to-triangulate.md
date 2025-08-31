# L-STH: Landmark-Augmented Self-Taught Hashing

L-STH improves upon the out-of-sample extension solution suggested by STH by leveraging the in-sample codes of the
nearest neighbors of the novel data point as features for per-bit classifier learning. This method both improves per-bit
codebook learning accuracy and enables cross-domain hashing use-cases, where the feature space used to construct the
graph is different than the feature space used during out-of-sample extension.

## Motivation & Context

Spectral hashing pipelines can produce highly useful binary codes for in-sample data points with customizable spatial
behavior. The underlying graph used in spectral hashing can be as simple as a k-NN graph using some observable document
features (e.g. text features or semantic embeddings), in which case the hamming distance between spectral embeddings
approximates distance in the original feature space, or as complex as one specifically designed such that the spectral
embeddings have specific spatial relationships (e.g. packing books of the same author as immediate nearest neighbors,
with books of the same year one band out, followed by books of the same genre, etc.). Note that in this complex scenario
the designed embedding space does not use the observable features of documents at all, instead relying on domain
knowledge to form graph edges.

Challenges arise when a new data point is to be projected into this embedding space. Traditional spectral hashing relies
on the uniform distribution assumption in order for its proposed analytical eigenfunctions to work, which is impractical
for real world datasets. Self-taught hashing addresses this by training per-bit classifiers to learn mappings from
observable features to embeddings, but this typically requires the graph be constructed using the same feature space
used to train the classifiers, and while projected codes can still be useful, the per-bit classifiers struggle to
recover the in-sample codes accurately. Anchor graph hashing is another approach to this problem, which replaces direct
per-bit classifiers with a derivation from distances between new data points and static in-sample anchor/landmark points
to spectral coordinates, and shares the practical requirement that the feature space used to compute distances from new
points to landmark points must be the same as the feature space used to construct the graph.

In the case of a customized embedding space derived from a graph encoding domain knowledge of entity relationships
instead of observable features, neither traditional spectral hashing's out-of-sample extension via analytical
eigenfunctions, self-taught hashing's per-bit classifiers, nor anchor graph hashing's derivation method can project new
data points into the embedding space sufficiently well due primarily due to the mismatch between the domain knowledge
used to construct the graph and the potentially unrelated observable features of the documents.

## Proposed Solution: Learned Triangulation from K-Nearest Landmarks

A generalized spectral hashing system is constructed where Feature Space A (domain features, e.g. tags/metadata) defines
graph topology optimized for domain-specific similarity, while Feature Space B (observed text/semantic features)
provides triangulation landmarks for out-of-sample extension. This intentional separation enables semantic search that
retrieves based on latent structural relationships from Space A while using Space B neighborhoods as landmarks for
triangulation learning to locate new points within that structure.

In other words, self-taught hashing is extended by leveraging the in-sample codes of neighbors to the new data point in
the readily available Feature Space B as the input features for the per-bit classifiers, optionally also including
Feature Space B as additional features. This transforms the difficult problem of learning to project from Feature Space
B into Feature Space A into a triangulation problem, where landmarks are sourced via Feature Space B and then
triangulation of a new data point within Feature Space A from the landmark codes is learned.

**Baseline Spectral Pipeline**

1. Build mutual k-NN graph from Feature Space A similarities

2. Perform eigen decomposition on the graph laplacian to obtain continuous embeddings

3. Apply iterative quantization or sign/median thresholding to obtain binary codes

4. Optionally, run a greedy ascent via bit flipping to further align bit selection with local adjacencies

This produces high-quality codes for in-sample data, but provides no mechanism for out-of-sample extension since new
points cannot be added to the fixed spectral decomposition.

**Out of Sample Extension**

To support out-of-sample extension, per-bit classifiers must be trained on in-sample data that map concatenated landmark
codes to code bits.

To train the classifiers:

1. Determine which in-sample codes are considered landmarks, for example, either choosing to use all of them or a subset
   with maximal coverage

2. Compute a k-nearest landmarks list for every landmark given their associated spectral codes

3. Produce tuples of (landmark code, feature vector) for each of these neighbor lists, where feature vector is the
   concatenated list of codes corresponding to the k-nearest landmarks, padded with zeros as needed

4. Train a multi-output classifier that learns a mapping from these feature vectors to the landmark code itself

At inference, for a new data point:

1. Find k-nearest landmarks in Feature Space B

2. Construct a feature vector from the concatenated list of codes corresponding to the k-nearest landmarks, padded with
   zeros as needed

3. Run the classifiers to predict the new data point's Space A code from the feature vector

### Scaling Considerations

The introduction of an additional landmark-lookup step at inference can add latency. This can be mitigated by both
aggressively reducing the Feature Space B dimensionality via feature selection or by reducing the number of indexed
landmarks.

## Novelty and Impact

**Theoretical**: Treats cross-domain out-of-sample extension as a learned triangulation problem rather than fixed
geometric operation or a direct per-bit learning problem, providing an alternative to STH and AGH that provides a
generalized solution the out-of-sample extension problem, allowing different feature spaces to be used used for graph
construction and extension.

**Practical**: Enables scalable spectral hashing deployment in situations where it is necessary to project into a
domain-specific embedding space from observable features different than those used in graph construction (e.g. designing
an embedding based on labeled data and then projecting into it from observed features)

## Future Exploration

- Optimal landmark selection
- Optimal feature selection for Feature Space B

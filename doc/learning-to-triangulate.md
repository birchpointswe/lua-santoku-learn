## Learning-to-Triangulate Spectral Hash Codes for Cross-Domain Extension

### Related Work and Keywords

This work intersects several research areas in the hashing and retrieval
literature:

**Cross-Modal Hashing**: Methods that learn hash codes across different
modalities (text, image, audio), particularly relevant for our cross-domain
Space A/Space B setup.

**Anchor Graph Hashing**: Uses representative anchor points for scalable
hashing, providing theoretical foundations for neighbor-based triangulation
approaches.

**Out-of-Sample Extension**: Techniques for generating hash codes for new data
without retraining, a core challenge we address through learned triangulation.

**Domain Adaptation Hashing**: Methods that adapt hash functions across
different domains or feature distributions, relevant for bridging Space A
structure with Space B neighborhoods.

**Neighbor-Augmented Learning**: Approaches that use k-nearest neighbor
information as additional features for improved prediction, closely related to
our triangulation learning paradigm.

**Spectral Hashing**: Graph-based methods that use eigenvalue decomposition for
hash code generation, forming our baseline pipeline.

**Consensus Hashing**: Techniques that combine multiple hash codes or hash
functions through learned aggregation, related to our learned triangulation of
neighbor codes.

### Problem Setup

We construct a spectral hashing system where Feature Space A (tags/metadata)
defines graph topology optimized for domain-specific similarity, while Feature
Space B (text/semantic content) provides triangulation landmarks for
out-of-sample extension. This intentional separation enables semantic search
that retrieves based on latent structural relationships from Space A while using
Space B neighborhoods as landmarks for triangulation learning to locate new
points within that structure.

### Baseline Spectral Pipeline

**In-Sample Code Generation:**
1. **Graph Construction**: Build mutual k-NN graph from Feature Space A similarities
2. **Spectral Decomposition**: Compute bottom eigenvectors of graph Laplacian
3. **Binarization**: Apply iterative quantization or sign thresholding to obtain binary codes
4. **Optional Optimization**: Greedy bit flipping to maximize adjacency preservation

This produces high-quality codes for in-sample data, but provides no mechanism
for out-of-sample extension since new points cannot be added to the fixed
spectral decomposition.

### Core Challenge: Out-of-Sample Extension

**The fundamental problem**: How to generate Space A-derived codes for new data
points without recomputing the entire spectral decomposition?

**Direct learning limitations**: Training classifiers to predict Space A codes
directly from Space B features is often intractable and insufficiently accurate.

### Proposed Solution: Learning-to-Triangulate

**Core Insight**: Use Space B neighborhoods to triangulate Space A codes through
learned optimal combination rather than naive aggregation.

**Method Overview:**
1. **Neighbor Retrieval**: For out-of-sample point, find k nearest neighbors in Space B
2. **Space A Code Retrieval**: Extract the pre-computed Space A codes for these k neighbors
3. **Learned Triangulation**: Train mapping from neighborhood Space A codes → inferred Space A code for novel point
4. **Optional Enhancement**: Include Space B features as additional signal when resources allow

**Learning-to-Triangulate Architecture:**

**Input Features:**
- **Primary**: Neighborhood Space A codes (concatenated: k × 128 features OR
  aggregated: 128 features)
- **Optional**: Space B features of query point (d additional features)

**Output**: Triangulated 128-bit Space A code for the novel point

**Training**: Learn mapping using in-sample data where Space B neighborhoods and
corresponding Space A codes are known

### Research Contributions

**1. Learned Triangulation Functions**
Replace hand-crafted aggregation (majority vote, distance weighting) with
learned optimal combination that adapts to local neighborhood patterns and
cross-domain context.

**2. Cross-Domain Out-of-Sample Extension**
Enable spectral hash extension without spectral recomputation by learning the
mapping between Space B neighborhoods and Space A spectral positions.

### Evaluation Framework

**Triangulation Learning Quality:**
- Average Hamming distance between triangulated codes and ground-truth Space A codes
- Per-bit error analysis: min/max/std bit error rates across code positions

**Training Data Performance:**
- AUC for retrieval tasks using triangulated codes
- Clustering quality metrics on triangulated training codes
- Best margin exhaustive scan: optimal Hamming radius for balanced
  precision/recall

**Out-of-Sample Extension Performance:**
- AUC for retrieval tasks using triangulated codes on unseen data
- Clustering quality on triangulated unseen codes
- Best margin analysis for unseen code performance

**Note**: All retrieval and clustering evaluation uses the original/sampled
pairwise data that constructed the initial Space A graph, testing how well the
triangulated codes capture the desired spatial characteristics in the projected
code space.

### Novelty and Impact

**Theoretical**: Treats cross-domain out-of-sample extension as a learned
triangulation problem rather than fixed geometric operation, providing
principled alternative to spectral recomputation.

**Practical**: Enables scalable spectral hashing deployment where recomputing
spectral decomposition for new data is computationally prohibitive.

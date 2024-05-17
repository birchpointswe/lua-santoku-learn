# Now

- Pass arguments as a table
- Consider pushing train & evaluate loops to C
- Shuffle every epoch
- Destroy, persist, load segfaults

- Implement regressor
- Add recurrent classifier, regressor

- Throw an error if length of input bitmaps isn't correct (2 x features /
  sizeof(unsigned int))
- Classes in Lua must always be 1 indexed to avoid confusion. In C they're
  decremented.

# Eventually

- Multi-granular (eliminates hyper-parameter S(ensitivity))
- Coalesced (reduces memory footprint for multi-output configurations)
- Weighted (reduces memory footprint)
- Integer-Weighted (see weighted))
- Indexed (improves learning and classification speed)
    - Triples memory usage

- Titanic dataset
- Mini batches

# Old, Reconsider

- User-specified hamming vs jaccard similarity metric for encoder,
  recurrent_encoder, autoencoder (requires different feedback logic)

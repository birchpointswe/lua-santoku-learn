# Now

- User-specified hamming vs jaccard similarity metric for encoder,
  recurrent_encoder, autoencoder (requires different feedback logic)

- Push training epoch loop into C so thread pool can be reused

- Implement regressor
- Add recurrent classifier, regressor

- Stacked auto-encoders, configurable layers
- Regressor
- Threadpool-based classifier, all classifiers share the same threadpool in the
  autoencoder

- Evaluate without malloc/etc consider using caller-passed tables or other
  data structures to store evaluate results

- Titanic dataset

- Throw an error if length of input bitmaps isn't correct (2 x features /
  sizeof(unsigned int))
- Classes in Lua must always be 1 indexed to avoid confusion. In C they're
  decremented.
- Persist model weights

# Eventually

- Multi-granular (eliminates hyper-parameter S(ensitivity))
- Coalesced (reduces memory footprint for multi-output configurations)
- Weighted (reduces memory footprint)
- Integer-Weighted (see weighted))
- Indexed (improves learning and classification speed)
    - Triples memory usage

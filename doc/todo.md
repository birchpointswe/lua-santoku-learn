# Now

- NEON, AVX, AVX-512
- Experiment with data layout for better cache performance

- Titanic dataset

- Throw an error if length of input bitmaps isn't correct (2 x features /
  sizeof(unsigned int))
- Classes in Lua must always be 1 indexed to avoid confusion. In C they're
  decremented.
- Persist model weights

# Eventually

- Multi-granular (eliminates hyper-parameter S(ensitivity))
- Indexed (improves learning and classification speed)
- Weighted (reduces memory footprint)
- Coalesced (reduces memory footprint for multi-output configurations)

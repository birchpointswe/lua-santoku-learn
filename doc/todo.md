# Now

- NEON, AVX, AVX-512
- Faster initialize random streams, initialize drop clause
- Experiment with data layout for better cache performance

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

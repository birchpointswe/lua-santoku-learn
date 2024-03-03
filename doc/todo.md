# Now

- Throw an error if length of input bitmaps isn't correct (2 x features /
  sizeof(unsigned int))
- Classes in Lua must always be 1 indexed to avoid confusion. In C they're
  decremented.

- Persist model weights
- Allow batch updates passed as a row-major bitmatrix
- NEON, AVX, AVX-512

- Tsetlin Machine advancements
  - Drop clause (increases accuracy and learning speed)
  - Indexed (improves learning and classification speed)
  - Weighted (reduces memory footprint)
  - Multi-granular (eliminates hyper-parameter S(ensitivity))
  - Coalesced (reduces memory footprint for multi-output configurations)

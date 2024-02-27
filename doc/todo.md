# Now

- Classes in Lua must always be 1 indexed to avoid confusion. In C they're
  decremented. '

- Allow batch updates passed as a row-major bitmatrix
- Leverage SIMD where possible

- Tsetlin Machine advancements
  - Drop clause (increases accuracy and learning speed)
  - Indexed (improves learning and classification speed)
  - Weighted (reduces memory footprint)
  - Multi-granular (eliminates hyper-parameter S(ensitivity))
  - Coalesced (reduces memory footprint for multi-output configurations)

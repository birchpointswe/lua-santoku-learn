# Now

- Standard test dataset for recurrent encoder
- Configurable number of threads. No pthreads at all when 0
- Parallelize classes during predict/update single, parallelize examples during
  training

- Pass arguments as a table
- Consider pushing train & evaluate loops to C
- Shuffle every epoch
- Re-use threads across epochs

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

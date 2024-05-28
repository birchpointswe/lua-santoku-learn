# Now

- Try `__thread` on `mcg_state` with random seed for each thread (if they all
  start the same, accuracy worsens dramatically)
- Add a "loss alpha" concept to classifier

- Use scaled loss_alpha based on max observed loss while training
- Release recent changes
- Fast booleanize matrix function (input: embedding matrix, output: thresholds
  and #bits)
- If threshold levels is 1, pick the middle number, if 2, use the 1/3 and 2/3s
  break points, and so on
- Need predict_many (one use case is rapidly encoding bitmaps in tbhss).
  Consider merging train/update and requiring an "n" param. Consider changing
  predict API to always accept a matrix and an "n" param instead of single.
- Error handling: features == 0
- Optimially space threshold levels: insetad of splitting evenly, split by
  value?

- Flipping input bits should be done in TM library, not required by the user.
  It's too easy to forget the complement bits.
- Experiment with dynamic margin for encoder, based on relative distances of N
  and P
- Configurable number of threads. No pthreads at all when 0
- Parallelize classes during predict/update single, parallelize examples during
  training

- Pass arguments as a table
- Consider pushing train & evaluate loops to C
- Shuffle every epoch
- Re-use threads across epochs

- Implement regressor

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

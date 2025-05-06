#include <assert.h>
#include <errno.h>
#include <lauxlib.h>
#include <limits.h>
#include <lua.h>
#include <math.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

#define _STR(x) #x
#define STR(x) _STR(x)

typedef uint64_t tk_bits_t;
#define BITS 64
#define BITS_DIV(x) ((x) >> 6)
#define BITS_MOD(x) ((x) & 63)
#define BYTES (BITS / CHAR_BIT)
#define BYTES_DIV(x) ((x) / BYTES)
const tk_bits_t ALL_MASK = ((tk_bits_t)0xFFFFFFFF << 32) | ((tk_bits_t)0xFFFFFFFF);
const tk_bits_t POS_MASK = ((tk_bits_t)0x55555555 << 32) | ((tk_bits_t)0x55555555);
const tk_bits_t NEG_MASK = ((tk_bits_t)0xAAAAAAAA << 32) | ((tk_bits_t)0xAAAAAAAA);
static inline unsigned int popcount (tk_bits_t x) {
  return (unsigned int) __builtin_popcountll(x);
}

// typedef __uint128_t tk_bits_t;
// #define BITS 128
// #define BITS_DIV(x) ((x) >> 7)
// #define BITS_MOD(x) ((x) & 127)
// const tk_bits_t ALL_MASK = ((tk_bits_t)0xFFFFFFFF << 96) |
//                            ((tk_bits_t)0xFFFFFFFF << 64) |
//                            ((tk_bits_t)0xFFFFFFFF << 32) |
//                            ((tk_bits_t)0xFFFFFFFF);
// const tk_bits_t POS_MASK = ((tk_bits_t)0x55555555 << 96) |
//                            ((tk_bits_t)0x55555555 << 64) |
//                            ((tk_bits_t)0x55555555 << 32) |
//                            ((tk_bits_t)0x55555555);
// const tk_bits_t NEG_MASK = ((tk_bits_t)0xAAAAAAAA << 96) |
//                            ((tk_bits_t)0xAAAAAAAA << 64) |
//                            ((tk_bits_t)0xAAAAAAAA << 32) |
//                            ((tk_bits_t)0xAAAAAAAA);
// static inline unsigned int popcount (tk_bits_t x) {
//   return
//     (unsigned int) __builtin_popcountll((uint64_t)(x >> 64)) +
//     (unsigned int) __builtin_popcountll((uint64_t)(x));
// }

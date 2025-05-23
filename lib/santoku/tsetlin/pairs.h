#ifndef TK_PAIRS_H
#define TK_PAIRS_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/roaring.h>

static inline void tm_pairs_init (
  lua_State *L,
  tm_pairs_t *pairs,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t *n_pos,
  uint64_t *n_neg
) {
  int kha;
  khint_t khi;

  uint64_t n_pos_old = *n_pos;
  uint64_t n_pos_new = 0;

  for (uint64_t i = 0; i < n_pos_old; i ++) {
    tm_pair_t p = pos[i];
    if (p.v < p.u)
      pos[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(pairs, khi) = true;
    n_pos_new ++;
  }

  uint64_t n_neg_old = *n_neg;
  uint64_t n_neg_new = 0;

  for (uint64_t i = 0; i < n_neg_old; i ++) {
    tm_pair_t p = neg[i];
    if (p.v < p.u)
      neg[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(pairs, khi) = false;
    n_neg_new ++;
  }

  *n_pos = n_pos_new;
  *n_neg = n_neg_new;

}

static inline void tm_adj_init (
  tm_pairs_t *pairs,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences
) {
  khint_t khi;

  // Init
  for (uint64_t s = 0; s < n_sentences; s ++) {
    adj_pos[s] = roaring64_bitmap_create();
    adj_neg[s] = roaring64_bitmap_create();
  }

  // Populate adj lists
  for (khi = kh_begin(pairs); khi < kh_end(pairs); khi ++) {
    if (!kh_exist(pairs, khi))
      continue;
    tm_pair_t p = kh_key(pairs, khi);
    bool l = kh_value(pairs, khi);
    if (l) {
      roaring64_bitmap_add(adj_pos[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(adj_pos[p.v], (uint64_t) p.u);
    } else {
      roaring64_bitmap_add(adj_neg[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(adj_neg[p.v], (uint64_t) p.u);
    }
  }
}

#endif

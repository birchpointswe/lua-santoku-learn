#ifndef TK_DSU_H
#define TK_DSU_H

KHASH_INIT(tk_dsu_members, int64_t, tk_iuset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_dsu_members) tk_dsu_members_t;

typedef struct {
  tk_iumap_t *parent;
  tk_iumap_t *rank;
  int64_t components;
} tk_dsu_t;

static inline void tk_dsu_free (tk_dsu_t *dsu)
{
  if (dsu->parent)
    tk_iumap_destroy(dsu->parent);
  if (dsu->rank)
    tk_iumap_destroy(dsu->rank);
}

static inline int64_t tk_dsu_components (tk_dsu_t *dsu)
{
  return dsu->components;
}

// TODO: Replace recursion with manual stack
static inline int64_t tk_dsu_find (
  tk_dsu_t *dsu,
  int64_t x
) {
  khint_t khi = tk_iumap_get(dsu->parent, x);
  assert(khi != tk_iumap_end(dsu->parent));
  int64_t p = tk_iumap_value(dsu->parent, khi);
  if (p != x) {
    p = tk_dsu_find(dsu, p);
    tk_iumap_value(dsu->parent, khi) = p;
  }
  return p;
}

static inline void tk_dsu_union (
  tk_dsu_t *dsu,
  int64_t x,
  int64_t y
) {
  int64_t xr = tk_dsu_find(dsu, x);
  int64_t yr = tk_dsu_find(dsu, y);
  if (xr == yr)
    return;
  khint_t xkhi = tk_iumap_get(dsu->rank, xr);
  assert(xkhi != tk_iumap_end(dsu->rank));
  khint_t ykhi = tk_iumap_get(dsu->rank, yr);
  assert(ykhi != tk_iumap_end(dsu->rank));
  dsu->components --;
  int64_t xrank = tk_iumap_value(dsu->rank, xkhi);
  int64_t yrank = tk_iumap_value(dsu->rank, ykhi);
  if (xrank < yrank) {
    int64_t tmp = xr; xr = yr; yr = tmp;
    int64_t tr = xrank; xrank = yrank; yrank = tr;
    khint_t tk = xkhi; xkhi = ykhi; ykhi = tk;
  }
  khint_t khi = tk_iumap_get(dsu->parent, yr);
  tk_iumap_value(dsu->parent, khi) = xr;
  if (xrank == yrank)
    tk_iumap_value(dsu->rank, xkhi) = xrank + 1;
}

static inline void tk_dsu_init (
  lua_State *L,
  tk_dsu_t *dsu,
  tk_ivec_t *ids
) {
  dsu->parent = tk_iumap_create();
  dsu->rank = tk_iumap_create();
  dsu->components = (int64_t) ids->n;
  int kha;
  khint_t khi;
  for (uint64_t i = 0; i < ids->n; i ++) {
    khi = tk_iumap_put(dsu->parent, ids->a[i], &kha);
    tk_iumap_value(dsu->parent, khi) = ids->a[i];
    khi = tk_iumap_put(dsu->rank, ids->a[i], &kha);
    tk_iumap_value(dsu->rank, khi) = 0;
  }
}

#endif

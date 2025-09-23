#ifndef TK_DSU_H
#define TK_DSU_H

typedef struct {
  int64_t *ids; // NOTE: currently owned by caller! TODO: Use ephemeron
  tk_iumap_t *ididx; // NOTE: rebuilt here, but likely can be passed in from caller
  int64_t *parent;
  int64_t *rank;
  int64_t components;
} tk_dsu_t;

static inline void tk_dsu_free (tk_dsu_t *dsu)
{
  tk_iumap_destroy(dsu->ididx);
  if (dsu->parent) {
    free(dsu->parent);
    dsu->parent = NULL;
  }
  if (dsu->rank) {
    free(dsu->rank);
    dsu->rank = NULL;
  }
}

static inline int64_t tk_dsu_components (tk_dsu_t *dsu)
{
  return dsu->components;
}

// TODO: Replace recursion with manual stack
// TODO: Expose separate strictly index-based version
static inline int64_t tk_dsu_findx (
  tk_dsu_t *dsu,
  int64_t uidx
) {
  int64_t pidx = dsu->parent[uidx];
  if (uidx != pidx) {
    pidx = tk_dsu_findx(dsu, pidx);
    dsu->parent[uidx] = pidx;
  }
  return pidx;
}

static inline int64_t tk_dsu_find (
  tk_dsu_t *dsu,
  int64_t uid
) {
  khint_t khi = tk_iumap_get(dsu->ididx, uid);
  assert(khi != tk_iumap_end(dsu->ididx));
  int64_t uidx = tk_iumap_val(dsu->ididx, khi);
  return dsu->ids[tk_dsu_findx(dsu, uidx)];
}

static inline void tk_dsu_unionx (
  tk_dsu_t *dsu,
  int64_t xidx,
  int64_t yidx
) {
  int64_t xridx = tk_dsu_findx(dsu, xidx);
  int64_t yridx = tk_dsu_findx(dsu, yidx);
  if (xridx == yridx)
    return;
  dsu->components --;
  int64_t xrank = dsu->rank[xridx];
  int64_t yrank = dsu->rank[yridx];
  if (xrank < yrank) {
    int64_t tmp = xridx; xridx = yridx; yridx = tmp;
    int64_t tr = xrank; xrank = yrank; yrank = tr;
  }
  dsu->parent[yridx] = xridx;
  if (xrank == yrank)
    dsu->rank[xridx] = xrank + 1;
}

static inline void tk_dsu_union (
  tk_dsu_t *dsu,
  int64_t x,
  int64_t y
) {
  khint_t xkhi = tk_iumap_get(dsu->ididx, x);
  assert(xkhi != tk_iumap_end(dsu->ididx));
  int64_t xidx = tk_iumap_val(dsu->ididx, xkhi);
  khint_t ykhi = tk_iumap_get(dsu->ididx, y);
  assert(ykhi != tk_iumap_end(dsu->ididx));
  int64_t yidx = tk_iumap_val(dsu->ididx, ykhi);
  tk_dsu_unionx(dsu, xidx, yidx);
}

static inline void tk_dsu_init (
  tk_dsu_t *dsu,
  tk_ivec_t *ids
) {
  dsu->ids = ids->a; // TODO: add ids as ephemeron to DSU
  dsu->ididx = tk_iumap_from_ivec(0, ids);
  dsu->parent = malloc(ids->n * sizeof(int64_t)); // TODO: ivec, not malloc
  dsu->rank = malloc(ids->n * sizeof(int64_t)); // TODO: ivec, not malloc
  dsu->components = (int64_t) ids->n;
  for (uint64_t i = 0; i < ids->n; i ++) {
    dsu->parent[i] = (int64_t) i;
    dsu->rank[i] = 0;
  }
}

#endif

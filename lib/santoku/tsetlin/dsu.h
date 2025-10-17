#ifndef TK_DSU_H
#define TK_DSU_H

#define TK_DSU_EPH "tk_dsu_eph"

typedef struct {
  tk_ivec_t *ids;
  tk_iumap_t *ididx;
  tk_ivec_t *parent;
  tk_ivec_t *rank;
  int64_t components;
  bool is_userdata;
} tk_dsu_t;

static inline void tk_dsu_destroy (tk_dsu_t *dsu)
{
  if (!dsu)
    return;
  bool was_userdata = dsu->is_userdata;
  if (dsu->ididx) {
    tk_iumap_destroy(dsu->ididx);
    dsu->ididx = NULL;
  }
  if (dsu->parent) {
    tk_ivec_destroy(dsu->parent);
    dsu->parent = NULL;
  }
  if (dsu->rank) {
    tk_ivec_destroy(dsu->rank);
    dsu->rank = NULL;
  }
  if (!was_userdata) {
    free(dsu);
  }
}

static inline int64_t tk_dsu_components (tk_dsu_t *dsu)
{
  return dsu->components;
}

static inline int64_t tk_dsu_findx (
  tk_dsu_t *dsu,
  int64_t uidx
) {
  int64_t pidx = dsu->parent->a[uidx];
  if (uidx != pidx) {
    pidx = tk_dsu_findx(dsu, pidx);
    dsu->parent->a[uidx] = pidx;
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
  return dsu->ids->a[tk_dsu_findx(dsu, uidx)];
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
  int64_t xrank = dsu->rank->a[xridx];
  int64_t yrank = dsu->rank->a[yridx];
  if (xrank < yrank) {
    int64_t tmp = xridx; xridx = yridx; yridx = tmp;
    int64_t tr = xrank; xrank = yrank; yrank = tr;
  }
  dsu->parent->a[yridx] = xridx;
  if (xrank == yrank)
    dsu->rank->a[xridx] = xrank + 1;
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

static inline void tk_dsu_add_ids (
  lua_State *L,
  tk_dsu_t *dsu
) {
  // Check if ids vector has grown
  uint64_t old_n = dsu->parent->n;
  uint64_t new_n = dsu->ids->n;

  if (new_n <= old_n)
    return;  // No new IDs

  // Extend parent and rank arrays
  tk_ivec_ensure(dsu->parent, new_n);
  tk_ivec_ensure(dsu->rank, new_n);

  // Add new IDs to the index and initialize their parent/rank
  for (uint64_t i = old_n; i < new_n; i++) {
    int64_t uid = dsu->ids->a[i];
    int kha;
    khint_t khi = tk_iumap_put(dsu->ididx, uid, &kha);
    if (kha) {  // New ID
      tk_iumap_setval(dsu->ididx, khi, (int64_t)i);
      dsu->parent->a[i] = (int64_t)i;  // Parent to itself
      dsu->rank->a[i] = 0;
      dsu->components++;
    }
  }

  // Update the sizes
  dsu->parent->n = new_n;
  dsu->rank->n = new_n;
}

static inline tk_dsu_t *tk_dsu_create (
  lua_State *L,
  tk_ivec_t *ids
) {
  tk_dsu_t *dsu;
  int dsu_idx;
  if (L) {
    dsu = lua_newuserdata(L, sizeof(tk_dsu_t));
    dsu_idx = lua_gettop(L);
    dsu->is_userdata = true;
  } else {
    dsu = malloc(sizeof(tk_dsu_t));
    if (!dsu) return NULL;
    memset(dsu, 0, sizeof(tk_dsu_t));
    dsu->is_userdata = false;
  }
  dsu->ids = ids;
  if (L) {
    dsu->ididx = tk_iumap_from_ivec(L, ids);
    if (!dsu->ididx)
      tk_error(L, "dsu_create: iumap_from_ivec failed", ENOMEM);
    tk_lua_add_ephemeron(L, TK_DSU_EPH, dsu_idx, -1);
    lua_pop(L, 1);
    dsu->parent = tk_ivec_create(L, ids->n, 0, 0);
    tk_lua_add_ephemeron(L, TK_DSU_EPH, dsu_idx, -1);
    lua_pop(L, 1);
    dsu->rank = tk_ivec_create(L, ids->n, 0, 0);
    tk_lua_add_ephemeron(L, TK_DSU_EPH, dsu_idx, -1);
    lua_pop(L, 1);
  } else {
    dsu->ididx = tk_iumap_from_ivec(0, ids);
    if (!dsu->ididx) {
      free(dsu);
      return NULL;
    }
    dsu->parent = tk_ivec_create(0, ids->n, 0, 0);
    if (!dsu->parent) {
      tk_iumap_destroy(dsu->ididx);
      free(dsu);
      return NULL;
    }
    dsu->rank = tk_ivec_create(0, ids->n, 0, 0);
    if (!dsu->rank) {
      tk_ivec_destroy(dsu->parent);
      tk_iumap_destroy(dsu->ididx);
      free(dsu);
      return NULL;
    }
  }
  dsu->components = (int64_t) ids->n;
  for (uint64_t i = 0; i < ids->n; i ++) {
    dsu->parent->a[i] = (int64_t) i;
    dsu->rank->a[i] = 0;
  }
  return dsu;
}

#endif

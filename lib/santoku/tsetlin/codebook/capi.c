#include "lua.h"
#include "lauxlib.h"
#include "lbfgs.h"
#include "../conf.h"
#include <arpack/arpack.h>
#include "roaring.h"
#include "roaring.c"

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

static inline void *tk_lua_checkuserdata (lua_State *L, int i, char *mt)
{
  if (mt == NULL && (lua_islightuserdata(L, i) || lua_isuserdata(L, i)))
    return lua_touserdata(L, i);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
}

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    memset(p, 0, s);
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i ++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
}

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_checklstring (lua_State *L, int i, size_t *lenp, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checklstring(L, i, lenp);
}

static inline const char *tk_lua_fchecklstring (lua_State *L, int i, size_t *lp, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checklstring(L, -1, lp);
  lua_pop(L, 1);
  return s;
}

static inline unsigned int tk_lua_foptunsigned (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline unsigned int tk_lua_optunsigned (lua_State *L, int i, unsigned int def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline uint64_t tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  return (uint64_t) l;
}

static inline uint64_t tk_lua_fcheckunsigned (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return (uint64_t) l;
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

typedef struct {
  tm_pair_t *pos;
  tm_pair_t *neg;
  uint64_t n_pos;
  uint64_t n_neg;
  uint64_t n_sentences;
  uint64_t n_hidden;
  lua_State *L;
  int i_each;
  unsigned int *global_iter;
} tm_ctx_t;

static double lbfgs_evaluate (
  void *instance,
  const double *x,
  double *g,
  const int n,
  const double step
) {
  tm_ctx_t *ctx = (tm_ctx_t *) instance;
  double loss = 0.0;
  for (int idx = 0; idx < n; idx ++)
    g[idx] = 0.0;
  for (unsigned int k = 0; k < ctx->n_pos + ctx->n_neg; k ++) {
    tm_pair_t *pairs = k < ctx->n_pos ? ctx->pos : ctx->neg;
    uint64_t offset = k < ctx->n_pos ? 0 : ctx->n_pos;
    int S = k < ctx->n_pos ? +1 : -1;
    tm_pair_t p = pairs[k - offset];
    double dot = 0.0;
    double *xi = (double *) (x + ((uint64_t) p.u) * ctx->n_hidden);
    double *xj = (double *) (x + ((uint64_t) p.v) * ctx->n_hidden);
    for (unsigned int f = 0; f < ctx->n_hidden; f ++)
      dot += xi[f] * xj[f];
    double target = (double) ctx->n_hidden * (double) S;
    double diff = dot - target;
    loss += diff * diff;
    double c = 2.0 * diff;
    for (unsigned int f = 0; f < ctx->n_hidden; f ++) {
      double xv = xi[f];
      double yv = xj[f];
      g[((uint32_t) p.u) * ctx->n_hidden + f] += c * yv;
      g[((uint32_t) p.v) * ctx->n_hidden + f] += c * xv;
    }
  }
  return loss;
}

static int lbfgs_progress (
  void *instance,
  const double *x,
  const double *g,
  double fx,
  double xnorm,
  double gnorm,
  double step,
  int n,
  int k,
  int ls
) {
  tm_ctx_t *ctx = (tm_ctx_t*) instance;
  double *z = (double *)x;
  for (unsigned int i = 0; i < ctx->n_sentences; i ++) {
    double norm = 0.0;
    for (unsigned int f = 0; f < ctx->n_hidden; f ++)
      norm += z[i * ctx->n_hidden + f] * z[i * ctx->n_hidden + f];
    norm = sqrt(norm) + 1e-8;
    for (unsigned int f = 0; f < ctx->n_hidden; f ++)
      z[i * ctx->n_hidden + f] /= norm;
  }
  (*ctx->global_iter) ++;
  if (ctx->i_each != -1) {
    lua_pushvalue(ctx->L, ctx->i_each);
    lua_pushinteger(ctx->L, *ctx->global_iter);
    lua_pushnumber(ctx->L, fx);
    lua_call(ctx->L, 2, 1);
    bool stop =
      lua_type(ctx->L, -1) == LUA_TBOOLEAN &&
      lua_toboolean(ctx->L, -1) == 0;
    lua_pop(ctx->L, 1);
    if (stop)
      return 1;
  }
  return 0;
}

typedef struct {
  uint64_t n_original;
  uint64_t n_components;
  int64_t *components;
  int64_t *parent;
  int64_t *rank;
  int64_t **members;
  uint64_t *count;
  uint64_t *cap;
} tm_dsu_t;

static inline void tm_dsu_init_components (lua_State *L, tm_dsu_t *dsu)
{
  dsu->components = tk_realloc(L, dsu->components, dsu->n_components * sizeof(int64_t));
  uint64_t pos = 0;
  for(uint64_t i = 0; i < dsu->n_original; i ++)
    if (dsu->members[i] != NULL)
      dsu->components[pos ++] = (int64_t) i;
}

static inline void tm_dsu_init (lua_State *L, tm_dsu_t *dsu, uint64_t n)
{
  dsu->n_original = n;
  dsu->n_components = n;
  dsu->components = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->parent = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->rank = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->members = tk_malloc(L, (size_t) n * sizeof(int64_t *));
  dsu->count = tk_malloc(L, (size_t) n * sizeof(uint64_t));
  dsu->cap = tk_malloc(L, (size_t) n * sizeof(uint64_t));
  for (uint64_t i = 0; i < n; i ++) {
    dsu->components[i] = (int64_t) i;
    dsu->parent[i] = (int64_t) i;
    dsu->rank[i] = 0;
    dsu->members[i] = tk_malloc(L, sizeof(int64_t));
    dsu->members[i][0] = (int64_t) i;
    dsu->count[i] = 1;
    dsu->cap[i] = 1;
  }
}

static inline void tm_dsu_free (tm_dsu_t *dsu)
{
  for (uint64_t i = 0; i < dsu->n_original; i ++)
    if (dsu->members[i])
      free(dsu->members[i]);
  free(dsu->members);
  free(dsu->count);
  free(dsu->cap);
  free(dsu->parent);
  free(dsu->rank);
  free(dsu->components);
}

static inline int64_t tm_dsu_find (tm_dsu_t *dsu, int64_t x)
{
  if (dsu->parent[x] != x)
    dsu->parent[x] = tm_dsu_find(dsu, dsu->parent[x]);
  return dsu->parent[x];
}

static inline void tm_dsu_union (lua_State *L, tm_dsu_t *dsu, int64_t x, int64_t y)
{
  int64_t xr = tm_dsu_find(dsu, x);
  int64_t yr = tm_dsu_find(dsu, y);
  if (xr == yr)
    return;
  if (dsu->rank[xr] == dsu->rank[yr])
    dsu->rank[xr] ++;
  int64_t big = xr, small = yr;
  if (dsu->rank[xr] < dsu->rank[yr]) {
    big = yr;
    small = xr;
  }
  dsu->parent[small] = big;
  uint64_t new_count = dsu->count[big] + dsu->count[small];
  if (dsu->cap[big] < new_count) {
    dsu->cap[big] = new_count;
    dsu->members[big] = tk_realloc(L, dsu->members[big], dsu->cap[big] * sizeof(int64_t));
  }
  memcpy(
    dsu->members[big] + dsu->count[big],
    dsu->members[small],
    dsu->count[small] * sizeof(int64_t));
  dsu->count[big] = new_count;
  free(dsu->members[small]);
  dsu->members[small] = NULL;
  dsu->count[small] = 0;
  dsu->cap[small] = 0;
  uint64_t old_n = dsu->n_components;
  dsu->n_components --;
  for (uint64_t i = 0; i < old_n; i ++) {
    if (dsu->components[i] == small) {
      dsu->components[i] = dsu->components[dsu->n_components];
      break;
    }
  }
}

static inline void tm_canonical_dedup (
  lua_State *L,
  tm_pair_t **pp,
  uint64_t *np
) {
  uint64_t n = *np;
  tm_pair_t *p0 = *pp;
  tm_pair_t *p1 = tk_malloc(L, n * sizeof(tm_pair_t));
  size_t w = 0;
  for (size_t i = 0; i < n; i ++) {
    int64_t u = p0[i].u;
    int64_t v = p0[i].v;
    if (u == v)
      continue;
    if (u > v) {
      int64_t tmp = u;
      u = v; v = tmp;
    }
    p1[w ++] = (tm_pair_t) { u, v };
  }
  ks_introsort(pair_asc, w, p1);
  size_t write = 0;
  for (size_t i = 0; i < w; i++)
    if (i == 0 || !tm_pair_eq(p1[i], p1[write - 1]))
      p1[write ++] = p1[i];
  free(p0);
  *pp = p1;
  *np = write;
}

static inline void tm_index_init (
  lua_State *L,
  roaring64_bitmap_t **index,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_features
) {
  for (uint64_t i = 0; i < n_features; i ++)
    index[i] = roaring64_bitmap_create();
  for (uint64_t i = 0; i < n_set_bits; i ++) {
    int64_t id = set_bits[i];
    int64_t sid = id / (int64_t) n_features;
    int64_t fid = id % (int64_t) n_features;
    roaring64_bitmap_add(index[fid], (uint64_t) sid);
  }
}

static inline void tm_neighbors_init (
  lua_State *L,
  tm_neighbors_t *pos_neighbors,
  tm_neighbor_t *neg_neighbors,
  roaring64_bitmap_t **index,
  roaring64_bitmap_t **sentences,
  uint64_t n_sentences,
  uint64_t n_features,
  uint64_t knn
) {
  roaring64_bitmap_t *candidates = roaring64_bitmap_create();
  for (int64_t s = 0; s < (int64_t) n_sentences; s ++) {
    tm_neighbor_t *worst_neg = neg_neighbors + s;
    *worst_neg = (tm_neighbor_t) { .v = -1, .s = 1.0 };
    roaring64_bitmap_clear(candidates);
    roaring64_iterator_t *it0 = roaring64_iterator_create(sentences[s]);
    while (roaring64_iterator_has_value(it0)) {
      uint64_t f = roaring64_iterator_value(it0);
      roaring64_bitmap_or_inplace(candidates, index[f]);
      roaring64_iterator_advance(it0);
    }
    roaring64_iterator_free(it0);
    kv_init(pos_neighbors[s]);
    roaring64_iterator_t *it1 = roaring64_iterator_create(candidates);
    while (roaring64_iterator_has_value(it1)) {
      int64_t c = (int64_t) roaring64_iterator_value(it1);
      roaring64_iterator_advance(it1);
      if (c == s)
        continue;
      uint64_t dist = roaring64_bitmap_xor_cardinality(sentences[s], sentences[c]);
      double sim = 1.0 - (double) dist / (double) n_features;
      if (sim < worst_neg->s) {
        worst_neg->v = c;
        worst_neg->s = sim;
      }
      if (sim <= 0.0)
        continue;
      tm_neighbor_t cand = { .v = c, .s = sim };
      kv_push(tm_neighbor_t, pos_neighbors[s], cand);
    }
    roaring64_iterator_free(it1);
    if (pos_neighbors[s].n > knn) {
      ks_ksmall(neighbors_desc, pos_neighbors[s].n, pos_neighbors[s].a, knn);
      pos_neighbors[s].n = knn;
    }
    ks_introsort(neighbors_asc, pos_neighbors[s].n, pos_neighbors[s].a);
    kv_resize(tm_neighbor_t, pos_neighbors[s], pos_neighbors[s].n);
  }
  roaring64_bitmap_free(candidates);
}

static inline void tm_populate_sentences (
  roaring64_bitmap_t **sentences,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_features,
  uint64_t n_sentences
) {
  for (unsigned int i = 0; i < n_sentences; i ++)
    sentences[i] = roaring64_bitmap_create();
  for (uint64_t b = 0; b < n_set_bits; b ++) {
    int64_t v = set_bits[b];
    int64_t s = v / (int64_t) n_features;
    int64_t f = v % (int64_t) n_features;
    roaring64_bitmap_add(sentences[s], (uint64_t) f);
  }
}

static inline int tm_codeify (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  lua_pushboolean(L, true);
  tk_lua_callmod(L, 2, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "neg");
  lua_pushboolean(L, true);
  tk_lua_callmod(L, 2, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "sentences");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "codeify", "n_features");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hidden");
  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "codeify", "n_sentences");
  uint64_t knn = tk_lua_foptunsigned(L, 1, 32, "codeify", "knn");
  uint64_t spectral_iterations = tk_lua_fcheckunsigned(L, 1, "codeify", "spectral_iterations");
  uint64_t lbfgs_iterations = tk_lua_fcheckunsigned(L, 1, "codeify", "lbfgs_iterations");

  if (BITS_MOD(n_hidden) != 0)
    tk_lua_verror(L, 3, "codify", "n_hidden", "must be a multiple of " STR(BITS));

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tm_canonical_dedup(L, &pos, &n_pos);
  tm_canonical_dedup(L, &neg, &n_neg);

  int kha;
  khint_t khi;
  tm_pairs_t *pairs = kh_init(pairs);

  for (uint64_t i = 0; i < n_pos; i ++) {
    khi = kh_put(pairs, pairs, pos[i], &kha);
    kh_value(pairs, khi) = true;
  }

  for (uint64_t i = 0; i < n_neg; i ++) {
    khi = kh_get(pairs, pairs, neg[i]);
    if (khi != kh_end(pairs))
      continue;
    khi = kh_put(pairs, pairs, neg[i], &kha);
    kh_value(pairs, khi) = false;
  }

  tm_dsu_t dsu;
  tm_dsu_init(L, &dsu, n_sentences);
  for (uint64_t i = 0; i < n_pos; i ++) {
    tm_pair_t p = pos[i];
    tm_dsu_union(L, &dsu, p.u, p.v);
  }

  if (dsu.n_components > 1) {

    roaring64_bitmap_t **sentences = tk_malloc(L, n_sentences *sizeof(roaring64_bitmap_t *));
    tm_populate_sentences(sentences, set_bits, n_set_bits, n_features, n_sentences);

    roaring64_bitmap_t **index = tk_malloc(L, n_features * sizeof(roaring64_bitmap_t *));
    tm_index_init(L, index, set_bits, n_set_bits, n_features);

    tm_neighbors_t *pos_neighbors = tk_malloc(L, n_sentences * sizeof(tm_neighbors_t));
    tm_neighbor_t *neg_neighbors = tk_malloc(L, n_sentences * sizeof(tm_neighbor_t));

    tm_neighbors_init(L, pos_neighbors, neg_neighbors, index, sentences, n_sentences, n_features, knn);

    uint64_t new_neg_added;
    kvec_t(tm_pair_t) new_pos_pairs;
    kv_init(new_pos_pairs);

    while (dsu.n_components > 1) {

      new_neg_added = 0;
      kv_size(new_pos_pairs) = 0;

      for (unsigned int i = 0; i < dsu.n_components; i ++) {

        int64_t component = dsu.components[i];

        int64_t best_u = -1;
        tm_neighbor_t best_neighbor = (tm_neighbor_t) { .v = -1, .s = -1.0 };

        for (unsigned int j = 0; j < dsu.count[component]; j ++) {
          int64_t next_u = dsu.members[component][j];
          while (pos_neighbors[next_u].n > 0) {
            tm_neighbor_t next_neighbor = pos_neighbors[next_u].a[pos_neighbors[next_u].n - 1];
            if (tm_dsu_find(&dsu, next_neighbor.v) == component) {
              pos_neighbors[next_u].n --;
              continue;
            } else if (next_neighbor.s > best_neighbor.s) {
              best_u = next_u;
              best_neighbor = next_neighbor;
            }
            break;
          }
        }

        if (best_u != -1) {

          tm_pair_t x;
          if (best_u < best_neighbor.v)
            x = ((tm_pair_t) { best_u, best_neighbor.v });
          else
            x = ((tm_pair_t) { best_neighbor.v, best_u });

          khi = kh_put(pairs, pairs, x, &kha);
          if (kha) {
            kv_push(tm_pair_t, new_pos_pairs, x);
            kh_value(pairs, khi) = true;
          }

          // int64_t worst_u = -1;
          // tm_neighbor_t worst_neighbor = (tm_neighbor_t) { .v = -1, .s = 1.0 };

          // for (unsigned int j = 0; j < dsu.count[component]; j ++) {
          //   int64_t next_u = dsu.members[component][j];
          //   tm_neighbor_t next_neighbor = neg_neighbors[next_u];
          //   if (next_neighbor.v < 0)
          //     continue;
          //   if (tm_dsu_find(&dsu, next_neighbor.v) == component)
          //     continue;
          //   if (next_neighbor.s < worst_neighbor.s) {
          //     worst_u = next_u;
          //     worst_neighbor = next_neighbor;
          //   }
          // }

          // if (worst_u != -1) {
          //   tm_pair_t x;
          //   if (worst_u < worst_neighbor.v)
          //     x = ((tm_pair_t) { worst_u, worst_neighbor.v });
          //   else
          //     x = ((tm_pair_t) { worst_neighbor.v, worst_u });
          //   khi = kh_put(pairs, pairs, x, &kha);
          //   if (kha) {
          //     new_neg_added ++;
          //     kh_value(pairs, khi) = false;
          //   }
          // }

        }

      }

      bool done = kv_size(new_pos_pairs) == 0;

      if (done) {

        // // Build adjacency lists for transitives
        // kvec_t(int64_t) pos_adj[n_sentences];
        // kvec_t(int64_t) neg_adj[n_sentences];
        // for (uint64_t i = 0; i < n_sentences; i ++) {
        //   kv_init(pos_adj[i]);
        //   kv_init(neg_adj[i]);
        // }
        // tm_pair_t p;
        // bool l;
        // kh_foreach(pairs, p, l, ({
        //   int64_t u = p.u, v = p.v;
        //   if (l) {
        //     kv_push(int64_t, pos_adj[u], v);
        //     kv_push(int64_t, pos_adj[v], u);
        //   } else {
        //     kv_push(int64_t, neg_adj[u], v);
        //     kv_push(int64_t, neg_adj[v], u);
        //   }
        // }))

        // // Add transitive positives
        // for (int64_t A = 0; A < (int64_t) n_sentences; A ++) {
        //   khint_t eB = pos_adj[A].n;
        //   bool added = false;
        //   for (khint_t iB = 0; iB < eB; iB ++) {
        //     int64_t B = pos_adj[A].a[iB];
        //     khint_t eC = pos_adj[B].n;
        //     for (khint_t iC = 0; iC < eC; iC ++) {
        //       int64_t C = pos_adj[B].a[iC];
        //       if (C == A)
        //         continue;
        //       tm_pair_t x;
        //       if (C < A) x = (tm_pair_t) { C, A };
        //       else x = (tm_pair_t) { A, C };
        //       khi = kh_put(pairs, pairs, x, &kha);
        //       if (!kha)
        //         continue;
        //       kh_value(pairs, khi) = true;
        //       kv_push(int64_t, pos_adj[A], C);
        //       kv_push(int64_t, pos_adj[C], A);
        //       kv_push(tm_pair_t, new_pos_pairs, x);
        //       added = true;
        //       break;
        //     }
        //     if (added)
        //       break;
        //   }
        // }

        // // Add transative negatives
        // for (int64_t A = 0; A < (int64_t) n_sentences; A ++) {
        //   khint_t eB = pos_adj[A].n;
        //   bool added = false;
        //   for (khint_t iB = 0; iB < eB; iB ++) {
        //     int64_t B = pos_adj[A].a[iB];
        //     khint_t eC = neg_adj[B].n;
        //     for (khint_t iC = 0; iC < eC; iC ++) {
        //       int64_t C = neg_adj[B].a[iC];
        //       if (C == A)
        //         continue;
        //       tm_pair_t x;
        //       if (C < A) x = (tm_pair_t) { C, A };
        //       else x = (tm_pair_t) { A, C };
        //       khi = kh_put(pairs, pairs, x, &kha);
        //       if (!kha)
        //         continue;
        //       kh_value(pairs, khi) = false;
        //       kv_push(int64_t, neg_adj[A], C);
        //       kv_push(int64_t, neg_adj[C], A);
        //       new_neg_added ++;
        //       added = true;
        //       break;
        //     }
        //     if (added)
        //       break;
        //   }
        // }

        for (unsigned i = 0; i < dsu.n_components - 1; i ++) {
          int64_t c0 = dsu.components[i];
          int64_t c1 = dsu.components[i + 1];
          int64_t u  = dsu.members[c0][0];
          int64_t v  = dsu.members[c1][0];
          tm_pair_t e = (u < v) ? (tm_pair_t) { u , v } : (tm_pair_t) { v, u };
          if (kh_get(pairs, pairs, e) != kh_end(pairs))
            continue;
          khi = kh_put(pairs, pairs, e, &kha);
          if (kha) {
            kh_val(pairs, khi) = true;
            kv_push(tm_pair_t, new_pos_pairs, e);
          }
        }

      }

      n_pos += kv_size(new_pos_pairs);
      n_neg += new_neg_added;

      for (khint_t k = 0; k < kv_size(new_pos_pairs); k ++)
        tm_dsu_union(L, &dsu, new_pos_pairs.a[k].u, new_pos_pairs.a[k].v);

      if (done)
        break;

    }

    kv_destroy(new_pos_pairs);

    for (int64_t s = 0; s < (int64_t) n_sentences; s ++)
      roaring64_bitmap_free(sentences[s]);

    for (int64_t s = 0; s < (int64_t) n_sentences; s ++)
      kv_destroy(pos_neighbors[s]);

    for (uint64_t f = 0; f < n_features; f ++)
      roaring64_bitmap_free(index[f]);

    free(sentences);
    free(pos_neighbors);
    free(neg_neighbors);
    free(index);

  }

  tm_dsu_free(&dsu);

  // free copy pairs back to pos/neg, update counts, free hash
  uint64_t wp = 0, wn = 0;
  tm_pair_t p;
  bool l;
  pos = tk_realloc(L, pos, n_pos * sizeof(tm_pair_t));
  neg = tk_realloc(L, neg, n_neg * sizeof(tm_pair_t));
  kh_foreach(pairs, p, l, ({
    if (l) pos[wp ++] = p;
    else neg[wn ++] = p;
  }))
  n_pos = wp;
  n_neg = wn;
  kh_destroy(pairs, pairs);
  ks_introsort(pair_asc, n_pos, pos);
  ks_introsort(pair_asc, n_neg, neg);

  printf(">  n_neg=%lu  n_pos=%lu\n", n_neg, n_pos);

  unsigned int global_iter = 0;

  // Build degree vector for Laplacian
  unsigned int *degree = tk_malloc(L, n_sentences * sizeof(unsigned int));
  memset(degree, 0, (size_t) n_sentences * sizeof(unsigned int));
  for (unsigned int k = 0; k < n_pos; k ++) {
    tm_pair_t p = pos[k];
    degree[p.u] ++;
    degree[p.v] ++;
  }
  for (unsigned int k = 0; k < n_neg; k ++) {
    tm_pair_t p = neg[k];
    degree[p.u] ++;
    degree[p.v] ++;
  }

  // ARPACK setup
  a_int ido = 0, info = 0;
  char bmat[] = "I", which[] = "SM";
  a_int n = (a_int) n_sentences;
  a_int nev = (a_int) n_hidden;
  a_int ncv = 2 * nev + 1;
  double tol = 1e-3;
  double *resid = tk_malloc(L, (size_t) n * sizeof(double));
  double *workd = tk_malloc(L, 3 * (size_t) n * sizeof(double));
  a_int iparam[11] = {1,0,spectral_iterations,0,0,0,1,0,0,0,0};
  a_int ipntr[14] = {0};
  int lworkl = ncv * (ncv + 8);
  double *workl = tk_malloc(L, (size_t) lworkl * sizeof(double));
  double *v = tk_malloc(L, (size_t) n * (size_t) ncv * sizeof(double));

  // Reverse-communication to build Lanczos basis v
  do {
    dsaupd_c(
      &ido, bmat, n, which, nev, tol, resid, ncv, v, n,
      iparam, ipntr, workd, workl, (a_int) lworkl, &info);
    if (ido == -1 || ido == 1) {
      double *in  = workd + ipntr[0] - 1;
      double *out = workd + ipntr[1] - 1;
      memset(out, 0, (size_t) n * sizeof(double));
      for (unsigned int e = 0; e < n_pos; e ++) {
        tm_pair_t p = pos[e];
        out[p.u] -= in[p.v];
        out[p.v] -= in[p.u];
      }
      for (unsigned int e = 0; e < n_neg; e ++) {
        tm_pair_t p = neg[e];
        out[p.u] += in[p.v];
        out[p.v] += in[p.u];
      }
      for (a_int i = 0; i < n; i ++)
        out[i] += degree[i] * in[i];
    }
  } while (ido == -1 || ido == 1);

  // Prepare containers
  a_int rvec = 1;
  char howmny[] = "A";
  a_int select[ncv];
  double d[nev];
  double *z = tk_malloc(L, (size_t) n * (size_t) nev * sizeof(double));
  double sigma = 0.0;
  dseupd_c(
    rvec, howmny, select, d, z, (a_int) n, sigma, bmat, (a_int) n, which, (a_int) nev,
    tol, resid, (a_int) ncv, v, (a_int) n, iparam, ipntr, workd, workl, (a_int) lworkl, &info);

  global_iter ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, global_iter);
    lua_pushinteger(L, iparam[4]);
    lua_call(L, 2, 0);
  }

  // Prepare context for LBFGS
  tm_ctx_t ctx = {
    pos, neg, n_pos, n_neg, n_sentences, n_hidden,
    L, i_each, &global_iter
  };

  // LBFGS refine
  int n_vars = n_sentences * n_hidden;
  if (lbfgs_iterations > 0) {
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = lbfgs_iterations;
    double fx = 0.0;
    lbfgs(n_vars, z, &fx,
          lbfgs_evaluate, lbfgs_progress,
          &ctx, &param);
  }

  // Threshold
  //
  // TODO: TCH discrete optimization, see two-step hashing paper
  //
  //   For t from 1 to m:
  //     Form the A matrix for bit t from pairwise loss
  //     Solve the small binary‐quadratic problem.
  //     Update the codebook t-th row with the new +/- 1 vector.
  //
  tk_bits_t *codes = tk_malloc(L, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  memset(codes, 0, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  for (unsigned int i = 0; i < n_sentences; i ++) {
    for (unsigned int f = 0; f < n_hidden; f ++) {
      if (z[i * n_hidden + f] > 0) {
        unsigned int chunk = BITS_DIV(f);
        unsigned int b = BITS_MOD(f);
        codes[i * BITS_DIV(n_hidden) + chunk] |= ((tk_bits_t) 1 << b);
      }
    }
  }

  // Cleanup
  free(resid);
  free(workd);
  free(workl);
  free(v);
  free(z);
  free(degree);

  lua_pushlightuserdata(L, pos);
  lua_pushinteger(L, (int64_t) n_pos);
  lua_pushinteger(L, 2);
  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 4, 0, "santoku.matrix.integer", "from_view");

  lua_pushlightuserdata(L, neg);
  lua_pushinteger(L, (int64_t) n_neg);
  lua_pushinteger(L, 2);
  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 4, 0, "santoku.matrix.integer", "from_view");

  lua_pushlstring(L, (char *) codes, n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  free(codes);
  return 1;
}

static luaL_Reg tm_codebook_fns[] =
{
  { "codeify", tm_codeify },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_codebook_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_codebook_fns, 0); // t
  return 1;
}

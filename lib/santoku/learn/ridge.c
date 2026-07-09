#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/csr.h>
#include <santoku/mtx.h>
#include <santoku/learn/gram.h>

#define TK_RIDGE_MT "tk_ridge_t"

typedef struct {
  tk_fvec_t *W;
  tk_fvec_t W_view;
  tk_dvec_t *intercept;
  int64_t n_dims;
  int64_t n_labels;
  float *sbuf;
  uint64_t sbuf_size;
  tk_rank_t *heap_buf;
  uint64_t heap_buf_size;
  bool destroyed;
} tk_ridge_t;

static inline tk_ridge_t *tk_ridge_peek (lua_State *L, int i) {
  return (tk_ridge_t *)luaL_checkudata(L, i, TK_RIDGE_MT);
}

static inline int tk_ridge_gc (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  free(r->sbuf);
  free(r->heap_buf);
  r->W = NULL;
  r->intercept = NULL;
  r->sbuf = NULL;
  r->heap_buf = NULL;
  r->destroyed = true;
  return 0;
}

static inline int64_t tk_ridge_block (int64_t nl) {
  int64_t block = 256;
  while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(float) > 64ULL * 1024 * 1024)
    block /= 2;
  return block;
}

static inline void tk_ridge_scores_block (
  tk_ridge_t *r, const float *codes_a, int64_t base, int64_t bs, float *out)
{
  int64_t nl = r->n_labels, d = r->n_dims;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)bs, (int)nl, (int)d, 1.0f, codes_a + base * d, (int)d,
    r->W->a, (int)nl, 0.0f, out, (int)nl);
  if (r->intercept)
    tk_gram_add_intercept_f(out, bs, nl, r->intercept->a);
}

static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  tk_mtx_t *Mx = tk_mtx_peek(L, 2, "codes");
  int64_t nl = r->n_labels;
  float *codes_a = ((tk_fvec_t *) Mx->v)->a;
  int64_t n = (int64_t) Mx->n_rows;
  int64_t k = (int64_t) luaL_checkinteger(L, 3);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n + 1)); offsets->n = (uint64_t)(n + 1);
  int offsets_idx = lua_gettop(L);
  tk_ivec_t *labels = tk_ivec_create(L, (uint64_t)(n * k)); labels->n = (uint64_t)(n * k);
  int labels_idx = lua_gettop(L);
  tk_fvec_t *scores_out = tk_fvec_create(L, (uint64_t)(n * k)); scores_out->n = (uint64_t)(n * k);
  int scores_out_idx = lua_gettop(L);
  for (int64_t i = 0; i <= n; i++) offsets->a[i] = i * k;
  int nt = omp_get_max_threads();
  uint64_t heap_need = (uint64_t)nt * (uint64_t)k;
  if (!r->heap_buf || r->heap_buf_size < heap_need) {
    free(r->heap_buf);
    r->heap_buf = (tk_rank_t *)malloc(heap_need * sizeof(tk_rank_t));
    r->heap_buf_size = heap_need;
  }
  int64_t block = tk_ridge_block(nl);
  uint64_t need = (uint64_t)block * (uint64_t)nl;
  if (!r->sbuf || r->sbuf_size < need) {
    free(r->sbuf);
    r->sbuf = (float *)malloc(need * sizeof(float));
    r->sbuf_size = need;
  }
  for (int64_t base = 0; base < n; base += block) {
    int64_t bs = (base + block <= n) ? block : n - base;
    tk_ridge_scores_block(r, codes_a, base, bs, r->sbuf);
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                         .a = r->heap_buf + (uint64_t)omp_get_thread_num() * (uint64_t)k };
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < bs; i++) {
        float *row = r->sbuf + i * nl;
        int64_t out_base = (base + i) * k;
        heap.n = 0;
        for (int64_t l = 0; l < nl; l++)
          tk_rvec_hmin(&heap, (size_t)k, tk_rank(l, (double)row[l]));
        tk_rvec_desc(&heap, 0, heap.n);
        for (int64_t j = 0; j < (int64_t)heap.n; j++) {
          labels->a[out_base + j] = heap.a[j].i;
          scores_out->a[out_base + j] = (float)heap.a[j].d;
        }
      }
    }
  }
  tk_csr_push(L, TK_TAG_F32, TK_TAG_I64, (uint64_t) nl,
    offsets_idx, offsets, labels_idx, (void *) labels, scores_out_idx, scores_out);
  return 1;
}

static inline int tk_ridge_persist_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKri", 1, 4, fh);
  uint8_t version = 4;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  uint8_t w_external = (r->W->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &w_external, sizeof(uint8_t), 1, fh);
  if (w_external) {
#if !defined(__EMSCRIPTEN__)
    msync(r->W->a, r->W->n * sizeof(float), MS_SYNC);
#endif
  } else {
    tk_fvec_persist(L, r->W, fh);
  }
  uint8_t has_intercept = r->intercept ? 1 : 0;
  tk_lua_fwrite(L, &has_intercept, sizeof(uint8_t), 1, fh);
  if (r->intercept)
    tk_dvec_persist(L, r->intercept, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

static inline int tk_ridge_transform_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels;
  tk_mtx_t *Mx = tk_mtx_peek(L, 2, "codes");
  float *codes_a = ((tk_fvec_t *) Mx->v)->a;
  int64_t n = (int64_t) Mx->n_rows;
  tk_fvec_t *buf = (lua_gettop(L) >= 3 && !lua_isnil(L, 3)) ? tk_fvec_peek(L, 3, "buf") : NULL;
  tk_fvec_t *out;
  if (buf) { tk_fvec_ensure(buf, (uint64_t)(n * nl)); buf->n = (uint64_t)(n * nl); out = buf; lua_pushvalue(L, 3); }
  else { out = tk_fvec_create(L, (uint64_t)(n * nl)); out->n = (uint64_t)(n * nl); }
  int out_idx = lua_gettop(L);
  int64_t block = tk_ridge_block(nl);
  for (int64_t base = 0; base < n; base += block) {
    int64_t bs = (base + block <= n) ? block : n - base;
    tk_ridge_scores_block(r, codes_a, base, bs, out->a + base * nl);
  }
  lua_pushvalue(L, out_idx);
  return 1;
}

static inline int tk_ridge_shrink_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  free(r->sbuf); r->sbuf = NULL; r->sbuf_size = 0;
  free(r->heap_buf); r->heap_buf = NULL; r->heap_buf_size = 0;
  return 0;
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "label", tk_ridge_encode_lua },
  { "persist", tk_ridge_persist_lua },
  { "regress", tk_ridge_transform_lua },
  { "shrink", tk_ridge_shrink_lua },
  { NULL, NULL }
};

static inline tk_ridge_t *tk_ridge_push (
  lua_State *L, tk_fvec_t *W, int W_idx, tk_dvec_t *intercept, int intercept_idx,
  int64_t d, int64_t nl)
{
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W;
  r->intercept = intercept;
  r->n_dims = d;
  r->n_labels = nl;
  r->sbuf = NULL;
  r->sbuf_size = 0;
  r->heap_buf = NULL;
  r->heap_buf_size = 0;
  r->destroyed = false;
  lua_newtable(L);
  if (W_idx > 0) {
    lua_pushvalue(L, W_idx);
    lua_setfield(L, -2, "W");
  }
  if (intercept_idx > 0) {
    lua_pushvalue(L, intercept_idx);
    lua_setfield(L, -2, "intercept");
  }
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return r;
}

static inline int tk_ridge_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "gram");
  if (!lua_isnil(L, -1)) {
    int gram_idx = lua_gettop(L);
    tk_gram_t *gram = tk_gram_peek(L, gram_idx);
    int64_t d = gram->n_dims, nl = gram->n_labels;
    uint64_t dnl = (uint64_t)d * (uint64_t)nl;
    if (!gram->baked)
      return luaL_error(L, "ridge create: gram must be baked (cholesky-only)");
    lua_getfield(L, 1, "w_buf");
    tk_fvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "w_buf");
    int w_buf_idx = w_buf ? lua_gettop(L) : 0;
    if (!w_buf) lua_pop(L, 1);
    tk_dvec_t *intercept_dv = NULL;
    int intercept_idx = 0;
    if (gram->intercept) {
      intercept_dv = tk_dvec_create(L, (uint64_t)nl);
      intercept_dv->n = (uint64_t)nl;
      memcpy(intercept_dv->a, gram->intercept, (uint64_t)nl * sizeof(double));
      intercept_idx = lua_gettop(L);
    }
    if (w_buf) {
      tk_fvec_ensure(w_buf, dnl);
      w_buf->n = dnl;
      memcpy(w_buf->a, gram->W_baked_f, dnl * sizeof(float));
      tk_ridge_push(L, w_buf, w_buf_idx, intercept_dv, intercept_idx, d, nl);
      return 1;
    }
    // adopt the gram's external w_buf when W was baked into it (keeps mmap-ness for persist)
    if (gram->W_baked_external) {
      lua_getfenv(L, gram_idx);
      lua_getfield(L, -1, "w_buf");
      tk_fvec_t *gwb = tk_fvec_peekopt(L, -1);
      if (gwb && gwb->a == gram->W_baked_f) {
        int gwb_idx = lua_gettop(L);
        gwb->n = dnl;
        tk_ridge_push(L, gwb, gwb_idx, intercept_dv, intercept_idx, d, nl);
        return 1;
      }
      lua_pop(L, 2);
    }
    tk_ridge_t *r = tk_ridge_push(L, NULL, 0, intercept_dv, intercept_idx, d, nl);
    r->W_view.n = dnl;
    r->W_view.m = dnl;
    r->W_view.a = gram->W_baked_f;
    r->W_view.lua_managed = 0;
    r->W = &r->W_view;
    lua_getfenv(L, -1);
    lua_pushvalue(L, gram_idx);
    lua_setfield(L, -2, "gram");
    lua_pop(L, 1);
    return 1;
  }
  lua_pop(L, 1);
  lua_getfield(L, 1, "W");
  if (!lua_isnil(L, -1)) {
    tk_fvec_t *W_fvec = tk_fvec_peek(L, -1, "W");
    int W_idx = lua_gettop(L);
    lua_getfield(L, 1, "n_dims");
    int64_t d = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, 1, "n_labels");
    int64_t nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
    tk_dvec_t *intercept_dv = NULL;
    int intercept_idx = 0;
    lua_getfield(L, 1, "intercept");
    if (!lua_isnil(L, -1)) {
      intercept_dv = tk_dvec_peek(L, -1, "intercept");
      intercept_idx = lua_gettop(L);
    } else {
      lua_pop(L, 1);
    }
    tk_ridge_push(L, W_fvec, W_idx, intercept_dv, intercept_idx, d, nl);
    return 1;
  }
  return luaL_error(L, "ridge create: gram or W required");
}

static inline int tk_ridge_load_lua (lua_State *L) {
  const char *data = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKri", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ridge file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 3 && version != 4) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  uint64_t dnl = (uint64_t)n_dims * (uint64_t)n_labels;
  tk_fvec_t *W = NULL;
  int W_idx = 0;
  bool have_arg_w = lua_gettop(L) >= 2 && !lua_isnil(L, 2);
  uint8_t w_external = 0;
  if (version == 4)
    tk_lua_fread(L, &w_external, sizeof(uint8_t), 1, fh);
  if (w_external) {
    if (!have_arg_w) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "ridge load: external W requires fvec arg 2");
    }
    W = tk_fvec_peek(L, 2, "W");
    if (W->n < dnl) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "ridge load: W buffer too small");
    }
    W_idx = 2;
  } else {
    if (have_arg_w) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "ridge load: file embeds W; do not pass a W buffer");
    }
    W = tk_fvec_load(L, fh);
    W_idx = lua_gettop(L);
  }
  tk_dvec_t *intercept = NULL;
  int b_idx = 0;
  uint8_t has_intercept;
  tk_lua_fread(L, &has_intercept, sizeof(uint8_t), 1, fh);
  if (has_intercept) {
    intercept = tk_dvec_load(L, fh);
    b_idx = lua_gettop(L);
  }
  tk_lua_fclose(L, fh);
  tk_ridge_push(L, W, W_idx, intercept, b_idx, n_dims, n_labels);
  return 1;
}

static luaL_Reg tk_ridge_fns[] = {
  { "create", tk_ridge_create_lua },
  { "load", tk_ridge_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  tk_lua_require_mod(L, "santoku.csr");
  tk_lua_require_mod(L, "santoku.mtx");
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}

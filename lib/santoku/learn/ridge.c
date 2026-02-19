#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>

#define TK_RIDGE_MT "tk_ridge_t"
#define TK_RIDGE_GRAM_MT "tk_ridge_gram_t"

typedef struct {
  tk_dvec_t *W;
  tk_dvec_t *feat_weights;
  int64_t n_dims;
  int64_t n_labels;
  bool is_sparse;
  bool destroyed;
} tk_ridge_t;

typedef struct {
  tk_dvec_t *Q;
  tk_dvec_t *eigenvalues;
  tk_dvec_t *QtXtY;
  tk_dvec_t *label_counts;
  tk_dvec_t *feat_weights;
  bool is_sparse;
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  bool destroyed;
} tk_ridge_gram_t;

static inline tk_ridge_t *tk_ridge_peek (lua_State *L, int i) {
  return (tk_ridge_t *)luaL_checkudata(L, i, TK_RIDGE_MT);
}

static inline tk_ridge_gram_t *tk_ridge_gram_peek (lua_State *L, int i) {
  return (tk_ridge_gram_t *)luaL_checkudata(L, i, TK_RIDGE_GRAM_MT);
}

static inline int tk_ridge_gc (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  r->W = NULL;
  r->feat_weights = NULL;
  r->destroyed = true;
  return 0;
}

static inline int tk_ridge_gram_gc (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  g->Q = NULL;
  g->eigenvalues = NULL;
  g->QtXtY = NULL;
  g->label_counts = NULL;
  g->feat_weights = NULL;
  g->destroyed = true;
  return 0;
}

typedef struct { double score; int64_t label; } tk_ridge_pair_t;

static inline void tk_ridge_heap_sift_down (tk_ridge_pair_t *h, int64_t n, int64_t i) {
  while (1) {
    int64_t s = i, l = 2 * i + 1, r = 2 * i + 2;
    if (l < n && h[l].score < h[s].score) s = l;
    if (r < n && h[r].score < h[s].score) s = r;
    if (s == i) break;
    tk_ridge_pair_t t = h[i]; h[i] = h[s]; h[s] = t;
    i = s;
  }
}

static int tk_ridge_pair_cmp_desc (const void *a, const void *b) {
  double da = ((const tk_ridge_pair_t *)a)->score;
  double db = ((const tk_ridge_pair_t *)b)->score;
  return (da > db) ? -1 : (da < db) ? 1 : 0;
}

static inline void tk_ridge_topk_block (
  double *sbuf, int64_t bs, int64_t nl, int64_t k, int64_t base,
  tk_ivec_t *offsets, tk_ivec_t *labels, tk_dvec_t *scores_out)
{
  #pragma omp parallel
  {
    tk_ridge_pair_t *heap = (tk_ridge_pair_t *)malloc((uint64_t)k * sizeof(tk_ridge_pair_t));
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < bs; i++) {
      double *row = sbuf + i * nl;
      int64_t out_base = (base + i) * k;
      int64_t hsize = 0;
      for (int64_t l = 0; l < nl; l++) {
        if (hsize < k) {
          heap[hsize].score = row[l];
          heap[hsize].label = l;
          hsize++;
          if (hsize == k)
            for (int64_t h = k / 2 - 1; h >= 0; h--)
              tk_ridge_heap_sift_down(heap, k, h);
        } else if (row[l] > heap[0].score) {
          heap[0].score = row[l];
          heap[0].label = l;
          tk_ridge_heap_sift_down(heap, k, 0);
        }
      }
      qsort(heap, (size_t)k, sizeof(tk_ridge_pair_t), tk_ridge_pair_cmp_desc);
      for (int64_t j = 0; j < k; j++) {
        labels->a[out_base + j] = heap[j].label;
        scores_out->a[out_base + j] = heap[j].score;
      }
    }
    free(heap);
  }
}

static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels, d = r->n_dims;
  int64_t n, k;
  tk_dvec_t *codes = NULL;
  tk_ivec_t *feat_off = NULL, *feat_idx = NULL;
  if (r->is_sparse) {
    feat_off = tk_ivec_peek(L, 2, "feature_offsets");
    feat_idx = tk_ivec_peek(L, 3, "feature_indices");
    n = (int64_t)luaL_checkinteger(L, 4);
    k = (int64_t)luaL_checkinteger(L, 5);
  } else {
    codes = tk_dvec_peek(L, 2, "codes");
    n = (int64_t)luaL_checkinteger(L, 3);
    k = (int64_t)luaL_checkinteger(L, 4);
  }
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
  int off_idx = lua_gettop(L);
  tk_ivec_t *labels = tk_ivec_create(L, (uint64_t)(n * k), NULL, NULL);
  int lab_idx = lua_gettop(L);
  tk_dvec_t *scores_out = tk_dvec_create(L, (uint64_t)(n * k), NULL, NULL);
  int sco_idx = lua_gettop(L);
  for (int64_t i = 0; i <= n; i++)
    offsets->a[i] = i * k;
  int64_t block = 256;
  while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(double) > 64ULL * 1024 * 1024)
    block /= 2;
  double *sbuf = (double *)malloc((uint64_t)block * (uint64_t)nl * sizeof(double));
  double *fw = r->feat_weights ? r->feat_weights->a : NULL;
  for (int64_t base = 0; base < n; base += block) {
    int64_t bs = (base + block <= n) ? block : n - base;
    if (r->is_sparse) {
      memset(sbuf, 0, (uint64_t)bs * (uint64_t)nl * sizeof(double));
      for (int64_t i = 0; i < bs; i++) {
        int64_t si = base + i;
        int64_t start = feat_off->a[si], end = feat_off->a[si + 1];
        double *row = sbuf + i * nl;
        for (int64_t a = start; a < end; a++) {
          int64_t f = feat_idx->a[a];
          double w = fw ? fw[f] : 1.0;
          cblas_daxpy((int)nl, w, r->W->a + f * nl, 1, row, 1);
        }
      }
    } else {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)bs, (int)nl, (int)d, 1.0, codes->a + base * d, (int)d,
        r->W->a, (int)nl, 0.0, sbuf, (int)nl);
    }
    tk_ridge_topk_block(sbuf, bs, nl, k, base, offsets, labels, scores_out);
  }
  free(sbuf);
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, lab_idx);
  lua_pushvalue(L, sco_idx);
  return 3;
}

static inline int tk_ridge_n_dims_lua (lua_State *L) {
  lua_pushinteger(L, (lua_Integer)tk_ridge_peek(L, 1)->n_dims);
  return 1;
}

static inline int tk_ridge_n_labels_lua (lua_State *L) {
  lua_pushinteger(L, (lua_Integer)tk_ridge_peek(L, 1)->n_labels);
  return 1;
}

static inline int tk_ridge_persist_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKri", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_persist(L, r->W, fh);
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      lua_pushlstring(L, data, len);
      free(data);
      tk_lua_fclose(L, fh);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "encode", tk_ridge_encode_lua },
  { "n_dims", tk_ridge_n_dims_lua },
  { "n_labels", tk_ridge_n_labels_lua },
  { "persist", tk_ridge_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_ridge_gram_mt_fns[] = {
  { NULL, NULL }
};

static inline void tk_ridge_build_xtx_xty (
  bool is_sparse, tk_dvec_t *codes,
  tk_ivec_t *feat_off, tk_ivec_t *feat_idx, double *fw,
  tk_ivec_t *lab_off, tk_ivec_t *lab_nbr,
  int64_t n, int64_t d, int64_t nl,
  double *XtX, double *XtY)
{
  if (is_sparse) {
    for (int64_t s = 0; s < n; s++) {
      int64_t fs = feat_off->a[s], fe = feat_off->a[s + 1];
      for (int64_t a = fs; a < fe; a++) {
        int64_t i = feat_idx->a[a];
        double wi = fw ? fw[i] : 1.0;
        XtX[i * d + i] += wi * wi;
        for (int64_t b = a + 1; b < fe; b++) {
          int64_t j = feat_idx->a[b];
          double wj = fw ? fw[j] : 1.0;
          double v = wi * wj;
          XtX[i * d + j] += v;
          XtX[j * d + i] += v;
        }
      }
      int64_t ls = lab_off->a[s], le = lab_off->a[s + 1];
      for (int64_t a = fs; a < fe; a++) {
        int64_t i = feat_idx->a[a];
        double wi = fw ? fw[i] : 1.0;
        for (int64_t b = ls; b < le; b++)
          XtY[i * nl + lab_nbr->a[b]] += wi;
      }
    }
  } else {
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
      (int)d, (int)n, 1.0, codes->a, (int)d, 0.0, XtX, (int)d);
    for (int64_t i = 0; i < d; i++)
      for (int64_t j = i + 1; j < d; j++)
        XtX[j * d + i] = XtX[i * d + j];
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
      double *xi = codes->a + i * d;
      for (int64_t j = lo; j < hi; j++) {
        int64_t l = lab_nbr->a[j];
        for (int64_t dd = 0; dd < d; dd++)
          XtY[dd * nl + l] += xi[dd];
      }
    }
  }
}

typedef struct {
  int64_t n, d, nl;
  bool is_sparse;
  tk_dvec_t *codes;
  tk_ivec_t *feat_off, *feat_idx;
  tk_dvec_t *feat_w;
  tk_ivec_t *lab_off, *lab_nbr;
} tk_ridge_data_t;

static inline void tk_ridge_parse_data (lua_State *L, char *fn, tk_ridge_data_t *out) {
  out->n = (int64_t)tk_lua_fcheckunsigned(L, 1, fn, "n_samples");
  out->nl = (int64_t)tk_lua_fcheckunsigned(L, 1, fn, "n_labels");
  lua_getfield(L, 1, "label_offsets");
  out->lab_off = tk_ivec_peek(L, -1, "label_offsets");
  lua_getfield(L, 1, "label_neighbors");
  out->lab_nbr = tk_ivec_peek(L, -1, "label_neighbors");
  lua_pop(L, 2);
  lua_getfield(L, 1, "feature_offsets");
  out->is_sparse = !lua_isnil(L, -1);
  lua_pop(L, 1);
  out->codes = NULL;
  out->feat_off = NULL;
  out->feat_idx = NULL;
  out->feat_w = NULL;
  if (out->is_sparse) {
    lua_getfield(L, 1, "feature_offsets");
    out->feat_off = tk_ivec_peek(L, -1, "feature_offsets");
    lua_getfield(L, 1, "feature_indices");
    out->feat_idx = tk_ivec_peek(L, -1, "feature_indices");
    lua_pop(L, 2);
    out->d = (int64_t)tk_lua_fcheckunsigned(L, 1, fn, "n_features");
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1))
      out->feat_w = tk_dvec_peek(L, -1, "feature_weights");
    lua_pop(L, 1);
  } else {
    lua_getfield(L, 1, "codes");
    out->codes = tk_dvec_peek(L, -1, "codes");
    lua_pop(L, 1);
    out->d = (int64_t)tk_lua_fcheckunsigned(L, 1, fn, "n_dims");
  }
}

static inline int tk_ridge_precompute_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  tk_ridge_data_t data;
  tk_ridge_parse_data(L, "precompute", &data);
  int64_t d = data.d, nl = data.nl, n = data.n;
  double *fw = data.feat_w ? data.feat_w->a : NULL;
  double *XtX = (double *)calloc((uint64_t)d * (uint64_t)d, sizeof(double));
  double *XtY = (double *)calloc((uint64_t)d * (uint64_t)nl, sizeof(double));
  tk_ridge_build_xtx_xty(data.is_sparse, data.codes,
    data.feat_off, data.feat_idx, fw,
    data.lab_off, data.lab_nbr, n, d, nl, XtX, XtY);
  double *counts = (double *)calloc((uint64_t)nl, sizeof(double));
  for (int64_t i = 0; i < n; i++) {
    int64_t lo = data.lab_off->a[i], hi = data.lab_off->a[i + 1];
    for (int64_t j = lo; j < hi; j++)
      counts[data.lab_nbr->a[j]] += 1.0;
  }
  double *evals = (double *)malloc((uint64_t)d * sizeof(double));
  int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', (int)d, XtX, (int)d, evals);
  if (info != 0) {
    free(XtX); free(XtY); free(counts); free(evals);
    return luaL_error(L, "ridge precompute: eigendecomposition failed (info=%d)", info);
  }
  double *QtXtY = (double *)malloc((uint64_t)d * (uint64_t)nl * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    (int)d, (int)nl, (int)d, 1.0, XtX, (int)d, XtY, (int)nl,
    0.0, QtXtY, (int)nl);
  free(XtY);
  tk_dvec_t *Q_dvec = tk_dvec_create(L, (uint64_t)(d * d), NULL, NULL);
  int Q_idx = lua_gettop(L);
  memcpy(Q_dvec->a, XtX, (uint64_t)d * (uint64_t)d * sizeof(double));
  free(XtX);
  tk_dvec_t *ev_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int ev_idx = lua_gettop(L);
  memcpy(ev_dvec->a, evals, (uint64_t)d * sizeof(double));
  free(evals);
  tk_dvec_t *qxty_dvec = tk_dvec_create(L, (uint64_t)(d * nl), NULL, NULL);
  int qxty_idx = lua_gettop(L);
  memcpy(qxty_dvec->a, QtXtY, (uint64_t)d * (uint64_t)nl * sizeof(double));
  free(QtXtY);
  tk_dvec_t *cnt_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
  int cnt_idx = lua_gettop(L);
  memcpy(cnt_dvec->a, counts, (uint64_t)nl * sizeof(double));
  free(counts);
  int fw_lua_idx = 0;
  if (data.is_sparse && data.feat_w) {
    lua_getfield(L, 1, "feature_weights");
    fw_lua_idx = lua_gettop(L);
  }
  tk_ridge_gram_t *g = tk_lua_newuserdata(L, tk_ridge_gram_t,
    TK_RIDGE_GRAM_MT, tk_ridge_gram_mt_fns, tk_ridge_gram_gc);
  int Gi = lua_gettop(L);
  g->Q = Q_dvec;
  g->eigenvalues = ev_dvec;
  g->QtXtY = qxty_dvec;
  g->label_counts = cnt_dvec;
  g->feat_weights = data.feat_w;
  g->is_sparse = data.is_sparse;
  g->n_dims = d;
  g->n_labels = nl;
  g->n_samples = n;
  g->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, Q_idx);
  lua_setfield(L, -2, "Q");
  lua_pushvalue(L, ev_idx);
  lua_setfield(L, -2, "eigenvalues");
  lua_pushvalue(L, qxty_idx);
  lua_setfield(L, -2, "QtXtY");
  lua_pushvalue(L, cnt_idx);
  lua_setfield(L, -2, "label_counts");
  if (fw_lua_idx) {
    lua_pushvalue(L, fw_lua_idx);
    lua_setfield(L, -2, "feat_weights");
  }
  lua_setfenv(L, Gi);
  lua_pushvalue(L, Gi);
  return 1;
}

static inline int tk_ridge_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "gram");
  if (!lua_isnil(L, -1)) {
    tk_ridge_gram_t *gram = tk_ridge_gram_peek(L, -1);
    int gram_lua_idx = lua_gettop(L);
    int64_t d = gram->n_dims, nl = gram->n_labels, n = gram->n_samples;
    lua_getfield(L, 1, "lambda");
    double lambda = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "propensity_a");
    bool do_prop = lua_isnumber(L, -1);
    double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "propensity_b");
    double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
    lua_pop(L, 1);
    uint64_t sz = (uint64_t)d * (uint64_t)nl;
    double *tmp = (double *)malloc(sz * sizeof(double));
    memcpy(tmp, gram->QtXtY->a, sz * sizeof(double));
    if (do_prop) {
      double C = (log((double)n) - 1.0) * pow(prop_b + 1.0, prop_a);
      double *pw = (double *)malloc((uint64_t)nl * sizeof(double));
      for (int64_t l = 0; l < nl; l++)
        pw[l] = 1.0 + C / pow(gram->label_counts->a[l] + prop_b, prop_a);
      for (int64_t i = 0; i < d; i++) {
        double inv_eig = 1.0 / (gram->eigenvalues->a[i] + lambda);
        double *row = tmp + i * nl;
        for (int64_t l = 0; l < nl; l++)
          row[l] *= pw[l] * inv_eig;
      }
      free(pw);
    } else {
      for (int64_t i = 0; i < d; i++)
        cblas_dscal((int)nl, 1.0 / (gram->eigenvalues->a[i] + lambda), tmp + i * nl, 1);
    }
    tk_dvec_t *W_dvec = tk_dvec_create(L, sz, NULL, NULL);
    int W_idx = lua_gettop(L);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)d, (int)nl, (int)d, 1.0, gram->Q->a, (int)d,
      tmp, (int)nl, 0.0, W_dvec->a, (int)nl);
    free(tmp);
    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_dvec;
    r->feat_weights = gram->feat_weights;
    r->n_dims = d;
    r->n_labels = nl;
    r->is_sparse = gram->is_sparse;
    r->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, W_idx);
    lua_setfield(L, -2, "W");
    lua_pushvalue(L, gram_lua_idx);
    lua_setfield(L, -2, "gram");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }
  lua_pop(L, 1);
  tk_ridge_data_t data;
  tk_ridge_parse_data(L, "create", &data);
  int64_t d = data.d, nl = data.nl, n = data.n;
  double *fw = data.feat_w ? data.feat_w->a : NULL;
  lua_getfield(L, 1, "lambda");
  double lambda = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0;
  lua_pop(L, 1);
  double *XtX = (double *)calloc((uint64_t)d * (uint64_t)d, sizeof(double));
  double *XtY = (double *)calloc((uint64_t)d * (uint64_t)nl, sizeof(double));
  tk_ridge_build_xtx_xty(data.is_sparse, data.codes,
    data.feat_off, data.feat_idx, fw,
    data.lab_off, data.lab_nbr, n, d, nl, XtX, XtY);
  for (int64_t i = 0; i < d; i++)
    XtX[i * d + i] += lambda;
  lua_getfield(L, 1, "propensity_a");
  bool do_prop = lua_isnumber(L, -1);
  double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "propensity_b");
  double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);
  if (do_prop) {
    double *counts = (double *)calloc((uint64_t)nl, sizeof(double));
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = data.lab_off->a[i], hi = data.lab_off->a[i + 1];
      for (int64_t j = lo; j < hi; j++)
        counts[data.lab_nbr->a[j]] += 1.0;
    }
    double C = (log((double)n) - 1.0) * pow(prop_b + 1.0, prop_a);
    for (int64_t l = 0; l < nl; l++) {
      double w = 1.0 + C / pow(counts[l] + prop_b, prop_a);
      for (int64_t dd = 0; dd < d; dd++)
        XtY[dd * nl + l] *= w;
    }
    free(counts);
  }
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', (int)d, XtX, (int)d);
  if (info != 0) {
    free(XtX); free(XtY);
    return luaL_error(L, "ridge: Cholesky failed (info=%d)", info);
  }
  info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', (int)d, (int)nl, XtX, (int)d, XtY, (int)nl);
  free(XtX);
  if (info != 0) {
    free(XtY);
    return luaL_error(L, "ridge: solve failed (info=%d)", info);
  }
  tk_dvec_t *W_dvec = tk_dvec_create(L, (uint64_t)(d * nl), NULL, NULL);
  int W_idx = lua_gettop(L);
  memcpy(W_dvec->a, XtY, (uint64_t)d * (uint64_t)nl * sizeof(double));
  free(XtY);
  int fw_ref_idx = 0;
  if (data.is_sparse && data.feat_w) {
    lua_getfield(L, 1, "feature_weights");
    fw_ref_idx = lua_gettop(L);
  }
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W_dvec;
  r->feat_weights = data.feat_w;
  r->n_dims = d;
  r->n_labels = nl;
  r->is_sparse = data.is_sparse;
  r->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, W_idx);
  lua_setfield(L, -2, "W");
  if (fw_ref_idx) {
    lua_pushvalue(L, fw_ref_idx);
    lua_setfield(L, -2, "feat_weights");
  }
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static inline int tk_ridge_load_lua (lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKri", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ridge file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_t *W = tk_dvec_load(L, fh);
  int W_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W;
  r->feat_weights = NULL;
  r->n_dims = n_dims;
  r->n_labels = n_labels;
  r->is_sparse = false;
  r->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, W_idx);
  lua_setfield(L, -2, "W");
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static luaL_Reg tk_ridge_fns[] = {
  { "create", tk_ridge_create_lua },
  { "precompute", tk_ridge_precompute_lua },
  { "load", tk_ridge_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}

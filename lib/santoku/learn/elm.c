#include <santoku/lua/utils.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/ivec.h>
#include <santoku/svec.h>
#include <santoku/cvec.h>
#include <santoku/learn/buf.h>
#include <santoku/learn/gram.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <santoku/learn/mathlibs.h>

static inline const int32_t *tk_peek_tokens (lua_State *L, int idx, uint64_t *out_n) {
  tk_svec_t *sv = tk_svec_peekopt(L, idx);
  if (sv) { *out_n = sv->n; return sv->a; }
  tk_ivec_t *iv = tk_ivec_peekopt(L, idx);
  if (!iv) { *out_n = 0; return NULL; }
  tk_svec_t *conv = tk_svec_create(L, iv->n);
  conv->n = iv->n;
  for (uint64_t i = 0; i < iv->n; i++) conv->a[i] = (int32_t)iv->a[i];
  lua_replace(L, idx);
  *out_n = conv->n;
  return conv->a;
}

#define TK_ELM_MOD_CSR   0
#define TK_ELM_MOD_DENSE 1

#define TK_ELM_MODE_RBF    0
#define TK_ELM_MODE_RELU   1
#define TK_ELM_MODE_LINEAR 2

#define TK_PHILOX_M0 0xD2511F53u
#define TK_PHILOX_M1 0xCD9E8D57u
#define TK_PHILOX_W0 0x9E3779B9u
#define TK_PHILOX_W1 0xBB67AE85u
#define TK_RFF_OMEGA_SALT 0x6f6d6567u

static inline void tk_philox4x32 (const uint32_t ctr_in[4], const uint32_t key_in[2], uint32_t out[4]) {
  uint32_t c0 = ctr_in[0], c1 = ctr_in[1], c2 = ctr_in[2], c3 = ctr_in[3];
  uint32_t k0 = key_in[0], k1 = key_in[1];
  for (int r = 0; r < 7; r++) {
    uint64_t p0 = (uint64_t)TK_PHILOX_M0 * c0;
    uint64_t p1 = (uint64_t)TK_PHILOX_M1 * c2;
    uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += TK_PHILOX_W0;
    k1 += TK_PHILOX_W1;
  }
  out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

static uint32_t tk_zig_kn[128];
static double tk_zig_wn[128], tk_zig_fn[128];
static int tk_zig_ready = 0;

static void tk_zig_init (void) {
  if (tk_zig_ready) return;
  const double m1 = 2147483648.0;
  const double vn = 9.91256303526217e-3;
  double dn = 3.442619855899, tn = dn;
  double q = vn / exp(-0.5 * dn * dn);
  tk_zig_kn[0] = (uint32_t)((dn / q) * m1);
  tk_zig_kn[1] = 0;
  tk_zig_wn[0] = q / m1;
  tk_zig_wn[127] = dn / m1;
  tk_zig_fn[0] = 1.0;
  tk_zig_fn[127] = exp(-0.5 * dn * dn);
  for (int i = 126; i >= 1; i--) {
    dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
    tk_zig_kn[i + 1] = (uint32_t)((dn / tn) * m1);
    tn = dn;
    tk_zig_fn[i] = exp(-0.5 * dn * dn);
    tk_zig_wn[i] = dn / m1;
  }
  tk_zig_ready = 1;
}

#define TK_ZIG_FIX_SALT 0x6669785au
typedef struct { uint32_t key[2]; uint32_t k; uint32_t salt; uint32_t ctr; uint32_t buf[4]; int pos; } tk_rng_t;

static inline uint32_t tk_rng_u32 (tk_rng_t *r) {
  if (r->pos >= 4) {
    uint32_t ctr[4] = { r->k, r->ctr++, r->salt, 0u };
    tk_philox4x32(ctr, r->key, r->buf);
    r->pos = 0;
  }
  return r->buf[r->pos++];
}

static inline double tk_rng_uni (tk_rng_t *r) {
  return ((double)tk_rng_u32(r) + 0.5) * (1.0 / 4294967296.0);
}

static double tk_zig_fixup (int32_t hz, const uint32_t key[2], uint32_t k, uint32_t f) {
  const double rr = 3.442619855899;
  tk_rng_t r = { { key[0], key[1] }, k, TK_ZIG_FIX_SALT + f, 0u, { 0u, 0u, 0u, 0u }, 4 };
  for (;;) {
    uint32_t iz = (uint32_t)hz & 127u;
    uint32_t az = (uint32_t)(hz < 0 ? -(int64_t)hz : hz);
    if (az < tk_zig_kn[iz]) return (double)hz * tk_zig_wn[iz];
    if (iz == 0) {
      double x, y;
      do { x = -log(tk_rng_uni(&r)) / rr; y = -log(tk_rng_uni(&r)); } while (y + y < x * x);
      return (hz > 0) ? (rr + x) : (-rr - x);
    }
    double x = (double)hz * tk_zig_wn[iz];
    if (tk_zig_fn[iz] + tk_rng_uni(&r) * (tk_zig_fn[iz - 1] - tk_zig_fn[iz]) < exp(-0.5 * x * x))
      return x;
    hz = (int32_t)tk_rng_u32(&r);
  }
}

static inline double tk_zig_one (uint32_t word, const uint32_t key[2], uint32_t k, uint32_t f) {
  int32_t hz = (int32_t)word;
  uint32_t iz = (uint32_t)hz & 127u;
  uint32_t az = (uint32_t)(hz < 0 ? -(int64_t)hz : hz);
  if (az < tk_zig_kn[iz]) return (double)hz * tk_zig_wn[iz];
  return tk_zig_fixup(hz, key, k, f);
}

static inline void tk_rff_gen (
  float *w, int64_t fa, int64_t fb, const uint32_t key[2], uint32_t k
) {
  for (int64_t f = fa; f < fb; f += 4) {
    uint32_t c[4] = { k, (uint32_t)(f >> 2), TK_RFF_OMEGA_SALT, 0u };
    uint32_t o[4];
    tk_philox4x32(c, key, o);
    int64_t lim = (f + 4 <= fb) ? 4 : fb - f;
    for (int64_t j = 0; j < lim; j++)
      w[f + j] = (float)tk_zig_one(o[j], key, k, (uint32_t)(f + j));
  }
}

typedef struct {
  float *acc;
  int32_t *head;
  int32_t *nxt;
  int32_t *samp;
  int32_t *touched;
} tk_rff_scratch_t;

static void tk_rff_fill_tile (
  uint64_t seed, int64_t nf, double scale, int mode,
  const int64_t *off, const int32_t *tok, const float *val,
  const float *codes_f, const double *codes_d, int64_t d_input,
  int64_t base, int64_t bs, float *tile_buf, float *wbuf, tk_rff_scratch_t *sc
) {
  uint32_t key[2] = { (uint32_t)seed, (uint32_t)(seed >> 32) };
  int has_csr = (off != NULL);
  float *acc = sc->acc;
  memset(acc, 0, (uint64_t)bs * (uint64_t)nf * sizeof(float));

  int64_t ntouch = 0, p0 = 0;
  if (has_csr) {
    p0 = off[base];
    for (int64_t i = 0; i < bs; i++)
      for (int64_t jj = off[base + i]; jj < off[base + i + 1]; jj++) {
        int64_t p = jj - p0;
        int32_t k = tok[jj];
        sc->samp[p] = (int32_t)i;
        sc->nxt[p] = sc->head[k];
        if (sc->head[k] < 0) sc->touched[ntouch++] = k;
        sc->head[k] = (int32_t)p;
      }
  }

  #pragma omp parallel
  {
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    int64_t fper = (((nf + nth - 1) / nth) + 3) & ~(int64_t)3;
    int64_t fa = (int64_t)tid * fper, fb = fa + fper;
    if (fb > nf) fb = nf;
    if (fa < nf) {
      float *restrict w = wbuf + (uint64_t)tid * (uint64_t)nf;
      if (has_csr) {
        for (int64_t t = 0; t < ntouch; t++) {
          uint32_t k = (uint32_t)sc->touched[t];
          tk_rff_gen(w, fa, fb, key, k);
          for (int32_t p = sc->head[k]; p >= 0; p = sc->nxt[p]) {
            float v = val ? val[p0 + p] : 1.0f;
            float *restrict arow = acc + (int64_t)sc->samp[p] * nf;
            for (int64_t f = fa; f < fb; f++) arow[f] += v * w[f];
          }
        }
      } else {
        for (int64_t kk = 0; kk < d_input; kk++) {
          tk_rff_gen(w, fa, fb, key, (uint32_t)kk);
          for (int64_t i = 0; i < bs; i++) {
            float v = codes_d ? (float)codes_d[(uint64_t)(base + i) * (uint64_t)d_input + (uint64_t)kk]
                              : codes_f[(uint64_t)(base + i) * (uint64_t)d_input + (uint64_t)kk];
            if (v == 0.0f) continue;
            float *restrict arow = acc + i * nf;
            for (int64_t f = fa; f < fb; f++) arow[f] += v * w[f];
          }
        }
      }
    }
  }

  if (has_csr)
    for (int64_t t = 0; t < ntouch; t++) sc->head[sc->touched[t]] = -1;

  float sc_f = (float)scale;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < bs; i++) {
    float *arow = acc + (int64_t)i * nf;
    if (mode == TK_ELM_MODE_RBF) {
      for (int64_t f = 0; f < nf; f++) {
        float s, c;
        sincosf(sc_f * arow[f], &s, &c);
        tile_buf[(uint64_t)(2 * f) * (uint64_t)bs + (uint64_t)i] = c;
        tile_buf[(uint64_t)(2 * f + 1) * (uint64_t)bs + (uint64_t)i] = s;
      }
    } else if (mode == TK_ELM_MODE_RELU) {
      for (int64_t f = 0; f < nf; f++) {
        float v = sc_f * arow[f];
        tile_buf[(uint64_t)f * (uint64_t)bs + (uint64_t)i] = v > 0.0f ? v : 0.0f;
      }
    } else {
      for (int64_t f = 0; f < nf; f++)
        tile_buf[(uint64_t)f * (uint64_t)bs + (uint64_t)i] = sc_f * arow[f];
    }
  }
}

static int64_t tk_rff_max_tnnz (const int64_t *off, int64_t n_samples, int64_t tile) {
  int64_t mx = 0;
  for (int64_t b = 0; b < n_samples; b += tile) {
    int64_t bs = (b + tile <= n_samples) ? tile : n_samples - b;
    int64_t tn = off[b + bs] - off[b];
    if (tn > mx) mx = tn;
  }
  return mx;
}

static void tk_rff_scratch_alloc (tk_rff_scratch_t *sc, int64_t tile, int64_t nf,
  uint64_t n_tokens, int64_t max_tnnz) {
  sc->acc = (float *)malloc((uint64_t)tile * (uint64_t)nf * sizeof(float));
  if (n_tokens > 0) {
    sc->head = (int32_t *)malloc((uint64_t)n_tokens * sizeof(int32_t));
    memset(sc->head, 0xff, (uint64_t)n_tokens * sizeof(int32_t));
    sc->nxt = (int32_t *)malloc((uint64_t)max_tnnz * sizeof(int32_t));
    sc->samp = (int32_t *)malloc((uint64_t)max_tnnz * sizeof(int32_t));
    sc->touched = (int32_t *)malloc((uint64_t)max_tnnz * sizeof(int32_t));
  } else {
    sc->head = NULL; sc->nxt = NULL; sc->samp = NULL; sc->touched = NULL;
  }
}

static void tk_rff_scratch_free (tk_rff_scratch_t *sc) {
  free(sc->acc); free(sc->head); free(sc->nxt); free(sc->samp); free(sc->touched);
}

#define TK_ELM_ENCODER_MT "tk_elm_encoder_t"

typedef struct {
  uint64_t seed;
  uint64_t d;
  uint8_t mod_type;
  uint8_t mode;
  uint64_t csr_n_tokens;
  int64_t d_input;
  double scale;
  bool destroyed;
} tk_elm_encoder_t;

static inline tk_elm_encoder_t *tk_elm_encoder_peek (lua_State *L, int i) {
  return (tk_elm_encoder_t *)luaL_checkudata(L, i, TK_ELM_ENCODER_MT);
}

static inline int tk_elm_encoder_gc (lua_State *L) {
  tk_elm_encoder_peek(L, 1)->destroyed = true;
  return 0;
}

static inline int tk_elm_encode_lua (lua_State *L) {
  tk_elm_encoder_t *enc = tk_elm_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "encode: encoder released");
  int64_t d = (int64_t)enc->d;
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "n_samples");
  int64_t n_samples = (int64_t)luaL_checkinteger(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 2, "offsets");
  tk_ivec_t *in_off = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "tokens");
  uint64_t in_tok_n;
  const int32_t *in_tok = tk_peek_tokens(L, -1, &in_tok_n);
  lua_setfield(L, 2, "tokens");
  lua_getfield(L, 2, "values");
  tk_fvec_t *in_val_f = tk_fvec_peekopt(L, -1);
  tk_dvec_t *in_val_d = in_val_f ? NULL : tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "codes");
  tk_dvec_t *in_codes_d = tk_dvec_peekopt(L, -1);
  tk_fvec_t *in_codes_f = in_codes_d ? NULL : tk_fvec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "output");
  tk_fvec_t *out_fv = tk_fvec_peekopt(L, -1);
  int out_fv_idx = out_fv ? lua_gettop(L) : 0;
  if (!out_fv) lua_pop(L, 1);

  tk_fvec_t *out;
  if (out_fv) {
    tk_fvec_ensure(out_fv, (uint64_t)(n_samples * d));
    out_fv->n = (uint64_t)(n_samples * d);
    out = out_fv;
  } else {
    out = tk_fvec_create(L, (uint64_t)(n_samples * d));
    out->n = (uint64_t)(n_samples * d);
    out_fv_idx = lua_gettop(L);
  }

  int64_t nf = (enc->mode == TK_ELM_MODE_RBF) ? d / 2 : d;
  int64_t tile = 1024;
  if (tile > n_samples) tile = n_samples > 0 ? n_samples : 1;
  float *tile_buf = (float *)malloc((uint64_t)d * (uint64_t)tile * sizeof(float));
  double scale = enc->scale;
  tk_zig_init();

  int nt = omp_get_max_threads();
  float *sketch_buf = (float *)malloc((uint64_t)nt * (uint64_t)nf * sizeof(float));
  float *csr_val = NULL;
  const float *csr_val_p = NULL;
  int is_csr = (enc->mod_type == TK_ELM_MOD_CSR);
  if (is_csr) {
    if (!in_off) { free(tile_buf); free(sketch_buf);
      return luaL_error(L, "encode: CSR encoder but no offsets"); }
    if (!in_val_f && in_val_d) {
      csr_val = (float *)malloc(in_tok_n * sizeof(float));
      for (uint64_t i = 0; i < in_tok_n; i++) csr_val[i] = (float)in_val_d->a[i];
      csr_val_p = csr_val;
    } else if (in_val_f) {
      csr_val_p = in_val_f->a;
    }
  }
  tk_rff_scratch_t sc;
  int64_t max_tnnz = (is_csr && n_samples > 0) ? tk_rff_max_tnnz(in_off->a, n_samples, tile) : 0;
  tk_rff_scratch_alloc(&sc, tile, nf, is_csr ? enc->csr_n_tokens : 0, max_tnnz);

  for (int64_t base = 0; base < n_samples; base += tile) {
    int64_t bs = (base + tile <= n_samples) ? tile : n_samples - base;
    tk_rff_fill_tile(enc->seed, nf, scale, enc->mode,
      is_csr ? in_off->a : NULL, in_tok, csr_val_p,
      in_codes_f ? in_codes_f->a : NULL, in_codes_d ? in_codes_d->a : NULL, enc->d_input,
      base, bs, tile_buf, sketch_buf, &sc);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < bs; i++) {
      float *orow = out->a + (uint64_t)(base + i) * (uint64_t)d;
      for (int64_t h = 0; h < d; h++)
        orow[h] = tile_buf[h * bs + i];
    }
  }

  free(tile_buf); free(sketch_buf); free(csr_val);
  tk_rff_scratch_free(&sc);
  lua_pushvalue(L, out_fv_idx);
  return 1;
}

static inline int tk_elm_dims_lua (lua_State *L) {
  tk_elm_encoder_t *enc = tk_elm_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->d);
  return 1;
}

static inline int tk_elm_persist_lua (lua_State *L) {
  tk_elm_encoder_t *enc = tk_elm_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "cannot persist a released encoder");
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKel", 1, 4, fh);
  uint8_t version = 10;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->seed, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->mode, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d_input, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &enc->scale, sizeof(double), 1, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

static inline int tk_elm_shrink_lua (lua_State *L) {
  tk_elm_encoder_peek(L, 1);
  return 0;
}

static luaL_Reg tk_elm_encoder_mt_fns[] = {
  { "encode", tk_elm_encode_lua },
  { "dims", tk_elm_dims_lua },
  { "persist", tk_elm_persist_lua },
  { "shrink", tk_elm_shrink_lua },
  { NULL, NULL }
};

static inline int tm_encode (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");
  uint64_t seed = tk_lua_foptunsigned(L, 1, "encode", "seed", 1);
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "encode", "n_hidden");
  int mode = TK_ELM_MODE_RBF;
  lua_getfield(L, 1, "mode");
  if (lua_isstring(L, -1)) {
    const char *ms = lua_tostring(L, -1);
    if (strcmp(ms, "rbf") == 0) mode = TK_ELM_MODE_RBF;
    else if (strcmp(ms, "relu") == 0) mode = TK_ELM_MODE_RELU;
    else if (strcmp(ms, "linear") == 0) mode = TK_ELM_MODE_LINEAR;
    else { lua_pop(L, 1); return luaL_error(L, "encode: unknown mode (rbf|relu|linear)"); }
  }
  lua_pop(L, 1);
  if (mode == TK_ELM_MODE_RBF && d % 2 != 0)
    return luaL_error(L, "encode: rbf n_hidden must be even (cos/sin pairs)");
  int64_t nf = (mode == TK_ELM_MODE_RBF) ? d / 2 : d;
  if (nf < 1) return luaL_error(L, "encode: n_hidden must be >= 1");
  double gamma = tk_lua_foptnumber(L, 1, "encode", "gamma", 1.0);

  uint8_t mod_type;
  tk_ivec_t *off_iv = NULL;
  const int32_t *tok_a = NULL;
  tk_fvec_t *val_fv = NULL;
  uint64_t csr_n_tokens = 0;
  tk_dvec_t *codes_dv = NULL;
  tk_fvec_t *codes_fv = NULL;
  int64_t d_input = 0;
  lua_getfield(L, 1, "offsets");
  int has_csr = !lua_isnil(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "codes");
  int has_dense = !lua_isnil(L, -1);
  lua_pop(L, 1);
  if (has_csr == has_dense)
    return luaL_error(L, "encode: provide exactly one modality (csr offsets or dense codes)");
  if (has_csr) {
    lua_getfield(L, 1, "offsets");
    off_iv = tk_ivec_peek(L, -1, "offsets");
    lua_pop(L, 1);
    lua_getfield(L, 1, "tokens");
    uint64_t tok_n;
    tok_a = tk_peek_tokens(L, -1, &tok_n);
    if (!tok_a) return luaL_error(L, "tokens: expected svec or ivec");
    lua_setfield(L, 1, "tokens");
    csr_n_tokens = tk_lua_fcheckunsigned(L, 1, "encode", "n_tokens");
    lua_getfield(L, 1, "values");
    val_fv = tk_fvec_peekopt(L, -1);
    lua_pop(L, 1);
    mod_type = TK_ELM_MOD_CSR;
  } else {
    lua_getfield(L, 1, "codes");
    codes_dv = tk_dvec_peekopt(L, -1);
    codes_fv = codes_dv ? NULL : tk_fvec_peekopt(L, -1);
    if (!codes_dv && !codes_fv) return luaL_error(L, "codes: expected dvec or fvec");
    lua_pop(L, 1);
    uint64_t cn = codes_dv ? codes_dv->n : codes_fv->n;
    lua_getfield(L, 1, "d_input");
    d_input = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : (int64_t)((uint64_t)cn / (uint64_t)n_samples);
    lua_pop(L, 1);
    mod_type = TK_ELM_MOD_DENSE;
  }

  lua_getfield(L, 1, "label_offsets");
  int has_labels = !lua_isnil(L, -1);
  tk_ivec_t *lbl_off = has_labels ? tk_ivec_peek(L, -1, "label_offsets") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_neighbors");
  tk_ivec_t *lbl_nbr = has_labels ? tk_ivec_peek(L, -1, "label_neighbors") : NULL;
  lua_pop(L, 1);
  int64_t nl = 0;
  if (has_labels) {
    lua_getfield(L, 1, "n_labels");
    nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "targets");
  int has_targets = !lua_isnil(L, -1);
  tk_dvec_t *targets_dv = has_targets ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  if (has_targets) {
    lua_getfield(L, 1, "n_targets");
    nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }

  int64_t tile_labels = 0;
  lua_getfield(L, 1, "tile_labels");
  if (lua_isnumber(L, -1)) tile_labels = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (tile_labels <= 0 && has_labels) tile_labels = 1024;
  int64_t tile_size = 1024;
  lua_getfield(L, 1, "tile_samples");
  if (lua_isnumber(L, -1)) tile_size = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "solve_lambda");
  int do_solve = lua_isnumber(L, -1);
  double solve_lambda = do_solve ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "solve_propensity_a");
  int solve_do_prop = do_solve && lua_isnumber(L, -1);
  double solve_prop_a = solve_do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "solve_propensity_b");
  double solve_prop_b = solve_do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);

  int build_gram = has_labels || has_targets;
  // Bake (Cholesky/tiled-spotrf -> W/intercept) whenever a solve_lambda is given -- this is the locked
  // path for EVERY mode (rbf always; relu/linear when not searching). Eigen-finalize (-> gram udata for a
  // lambda/propensity sweep) only when no solve_lambda is provided (the relu/linear search path).
  int eigen = build_gram && !do_solve;
  int build_tiled = has_labels && !eigen;
  const float *csr_val_p = (has_csr && val_fv) ? val_fv->a : NULL;

  uint64_t ud = (uint64_t)d;
  uint64_t unl = (uint64_t)nl;

  float *XtX = NULL;
  if (build_gram) {
    XtX = (float *)tk_gram_pool_get(L, ud * ud * sizeof(float));
    memset(XtX, 0, ud * ud * sizeof(float));
  }
  float *xty = build_gram ? (float *)calloc(ud * (unl ? unl : 1), sizeof(float)) : NULL;
  double *colsum = (double *)calloc(ud, sizeof(double));
  float *tile_buf = (float *)malloc(ud * (uint64_t)tile_size * sizeof(float));
  float *tgt_tile = has_targets ? (float *)malloc(unl * (uint64_t)tile_size * sizeof(float)) : NULL;

  int nthreads = omp_get_max_threads();
  uint64_t unf = (uint64_t)nf;
  float *sketch_buf = (float *)malloc((uint64_t)nthreads * unf * sizeof(float));
  tk_zig_init();
  tk_rff_scratch_t sc;
  int64_t max_tnnz = (has_csr && n_samples > 0) ? tk_rff_max_tnnz(off_iv->a, n_samples, tile_size) : 0;
  tk_rff_scratch_alloc(&sc, tile_size, nf, has_csr ? csr_n_tokens : 0, max_tnnz);

  double scale = (mode == TK_ELM_MODE_RBF) ? gamma : 1.0;

  for (int64_t base = 0; base < n_samples; base += tile_size) {
    int64_t bs = (base + tile_size <= n_samples) ? tile_size : n_samples - base;
    uint64_t ubs = (uint64_t)bs;
    tk_rff_fill_tile(seed, nf, scale, mode,
      has_csr ? off_iv->a : NULL, tok_a, csr_val_p,
      codes_fv ? codes_fv->a : NULL, codes_dv ? codes_dv->a : NULL, d_input,
      base, bs, tile_buf, sketch_buf, &sc);
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < d; h++) {
      float *col = tile_buf + (uint64_t)h * ubs;
      double s = 0.0;
      for (uint64_t i = 0; i < ubs; i++) s += (double)col[i];
      colsum[h] += s;
    }
    if (build_gram) {
      cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
        (int)d, (int)bs, 1.0f, tile_buf, (int)bs, 1.0f, XtX, (int)d);
      if (has_targets) {
        for (uint64_t i = 0; i < ubs * unl; i++)
          tgt_tile[i] = (float)targets_dv->a[(uint64_t)base * unl + i];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)d, (int)nl, (int)bs, 1.0f, tile_buf, (int)bs,
          tgt_tile, (int)nl, 1.0f, xty, (int)nl);
      } else {
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < d; k++) {
          float *col = tile_buf + (uint64_t)k * ubs;
          for (uint64_t i = 0; i < ubs; i++) {
            uint64_t si = (uint64_t)base + i;
            for (int64_t j = lbl_off->a[si]; j < lbl_off->a[si + 1]; j++)
              xty[k * nl + lbl_nbr->a[j]] += col[i];
          }
        }
      }
    }
  }
  free(sketch_buf); free(tgt_tile); tk_rff_scratch_free(&sc);

  tk_elm_encoder_t *enc = (tk_elm_encoder_t *)tk_lua_newuserdata(L, tk_elm_encoder_t,
    TK_ELM_ENCODER_MT, tk_elm_encoder_mt_fns, tk_elm_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->seed = seed;
  enc->d = (uint64_t)d;
  enc->mod_type = mod_type;
  enc->mode = (uint8_t)mode;
  enc->csr_n_tokens = csr_n_tokens;
  enc->d_input = d_input;
  enc->scale = scale;
  enc->destroyed = false;
  lua_newtable(L);
  lua_setfenv(L, enc_idx);

  int W_idx = 0, int_idx = 0;   // W (fvec) + intercept (dvec) the elm rbf solve hands to ridge.create
  int gram_idx = 0;             // eigen gram udata (relu/linear) the lambda/prop sweep consumes
  if (build_gram) {
    double *col_mean = (double *)malloc(ud * sizeof(double));
    for (int64_t h = 0; h < d; h++) col_mean[h] = colsum[h] / (double)n_samples;
    double *y_mean = (double *)malloc((unl ? unl : 1) * sizeof(double));
    tk_dvec_t *lc = NULL;
    int lc_idx = 0;
    if (has_labels) {
      lc = tk_dvec_create(L, unl); lc->n = unl;
      lc_idx = lua_gettop(L);
      memset(lc->a, 0, unl * sizeof(double));
      for (int64_t s = 0; s < n_samples; s++)
        for (int64_t j = lbl_off->a[s]; j < lbl_off->a[s + 1]; j++)
          lc->a[lbl_nbr->a[j]] += 1.0;
      for (int64_t l = 0; l < nl; l++) y_mean[l] = lc->a[l] / (double)n_samples;
    } else {
      for (int64_t l = 0; l < nl; l++) {
        double s = 0.0;
        for (int64_t i = 0; i < n_samples; i++) s += targets_dv->a[(uint64_t)i * unl + (uint64_t)l];
        y_mean[l] = s / (double)n_samples;
      }
    }

    float *cm_f = (float *)malloc(ud * sizeof(float));
    for (int64_t h = 0; h < d; h++) cm_f[h] = (float)col_mean[h];
    float *ym_f = (float *)malloc((unl ? unl : 1) * sizeof(float));
    for (int64_t l = 0; l < nl; l++) ym_f[l] = (float)y_mean[l];

    if (eigen) {
      // relu/linear: feature map is gamma-free so the gram is invariant across trials -> eigen-finalize
      // once (consumes XtX/xty/col_mean/y_mean), then optimize.lua sweeps lambda/prop on the eigenbasis.
      double *eigenvals = (double *)malloc(ud * sizeof(double));
      tk_gram_finalize_f_native(L, XtX, xty, cm_f, ym_f, col_mean, y_mean, eigenvals,
        lc, lc_idx, n_samples, d, nl);
      free(cm_f); free(ym_f);
      gram_idx = lua_gettop(L);
    } else if (!build_tiled) {
      // regression: finalize solves W + intercept into fvec/dvec (consumes XtX/xty/col_mean/y_mean)
      tk_gram_finalize_cholesky_f(L, XtX, xty, cm_f, ym_f, col_mean, y_mean,
        NULL, n_samples, d, nl, solve_lambda, false, 0.0, 0.0);
      free(cm_f); free(ym_f);
      int_idx = lua_gettop(L); W_idx = int_idx - 1;
    } else {
      int64_t B = tile_labels < nl ? tile_labels : nl;
      cblas_ssyr(CblasColMajor, CblasUpper, (int)d, -(float)n_samples, cm_f, 1, XtX, (int)d);
      double mean_eig = 0.0;
      for (uint64_t i = 0; i < ud; i++) mean_eig += (double)XtX[i * ud + i];
      mean_eig /= (double)d;
      if (tk_spotrf_escalate(XtX, d, solve_lambda, mean_eig) < 0.0) {
        free(cm_f); free(ym_f); free(col_mean); free(y_mean); free(XtX); free(xty);
        return luaL_error(L, "elm tiled cholesky: spotrf failed (singular even after jitter escalation)"); }
      tk_fvec_t *W_fv = tk_fvec_create(L, ud * unl); W_fv->n = ud * unl;
      float *W_baked = W_fv->a;
      tk_dvec_t *int_dv = tk_dvec_create(L, unl); int_dv->n = unl;
      double *intercept = int_dv->a;
      float *Bcm = (float *)malloc(ud * (uint64_t)B * sizeof(float));
      double C = solve_do_prop ? (log((double)n_samples) - 1.0) * pow(solve_prop_b + 1.0, solve_prop_a) : 0.0;
      for (int64_t tl = 0; tl < nl; tl += B) {
        int64_t aB = (tl + B <= nl) ? B : nl - tl;
        for (int64_t k = 0; k < d; k++)
          for (int64_t l = 0; l < aB; l++)
            Bcm[(uint64_t)l * ud + (uint64_t)k] = xty[(uint64_t)k * unl + (uint64_t)(tl + l)];
        cblas_sger(CblasColMajor, (int)d, (int)aB, -(float)n_samples,
          cm_f, 1, ym_f + tl, 1, Bcm, (int)d);
        if (solve_do_prop)
          for (int64_t l = 0; l < aB; l++) {
            float p = (float)(1.0 + C / pow(lc->a[tl + l] + solve_prop_b, solve_prop_a));
            for (int64_t k = 0; k < d; k++) Bcm[(uint64_t)l * ud + (uint64_t)k] *= p;
          }
        LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', (int)d, (int)aB, XtX, (int)d, Bcm, (int)d);
        for (int64_t k = 0; k < d; k++)
          for (int64_t l = 0; l < aB; l++)
            W_baked[(uint64_t)k * unl + (uint64_t)(tl + l)] = Bcm[(uint64_t)l * ud + (uint64_t)k];
        for (int64_t l = 0; l < aB; l++) {
          double prop = solve_do_prop ? (1.0 + C / pow(lc->a[tl + l] + solve_prop_b, solve_prop_a)) : 1.0;
          intercept[tl + l] = prop * y_mean[tl + l]
            - (double)cblas_sdot((int)d, Bcm + (uint64_t)l * ud, 1, cm_f, 1);
        }
      }
      free(Bcm); free(XtX); free(xty); free(cm_f); free(ym_f); free(col_mean); free(y_mean);
      int_idx = lua_gettop(L); W_idx = int_idx - 1;
    }
  }

  free(colsum); free(tile_buf);

  lua_pushnil(L);
  lua_pushvalue(L, enc_idx);
  if (gram_idx > 0) {
    lua_pushvalue(L, gram_idx);
    return 3;    // (nil, encoder, gram) -- relu/linear eigen path
  }
  if (W_idx > 0) {
    lua_pushvalue(L, W_idx);
    lua_pushvalue(L, int_idx);
    return 4;   // (nil, encoder, W, intercept) -- rbf cholesky path
  }
  return 2;
}

static inline int tk_elm_encoder_load_lua (lua_State *L) {
  lua_settop(L, 1);
  const char *path = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, path, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKel", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid elm encoder file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 10) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported elm encoder version %d", (int)version);
  }
  tk_elm_encoder_t *enc = (tk_elm_encoder_t *)tk_lua_newuserdata(L, tk_elm_encoder_t,
    TK_ELM_ENCODER_MT, tk_elm_encoder_mt_fns, tk_elm_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->destroyed = false;
  tk_lua_fread(L, &enc->seed, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->mode, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->d_input, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &enc->scale, sizeof(double), 1, fh);
  tk_lua_fclose(L, fh);
  lua_newtable(L);
  lua_setfenv(L, enc_idx);
  lua_pushvalue(L, enc_idx);
  return 1;
}

static luaL_Reg tm_fns[] = {
  { "encode", tm_encode },
  { "load", tk_elm_encoder_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_elm (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}

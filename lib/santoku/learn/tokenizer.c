




#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <santoku/lua/utils.h>
#include <santoku/learn/normalize.h>
#include <santoku/ivec/ext.h>
#include <santoku/svec.h>
#include <santoku/fvec.h>
#include <santoku/cvec.h>
#include <santoku/iumap/ext.h>

#define TK_TOK_MT "tk_tokenizer_t"
#define TK_TOK_MAXTYPES 64
#define TK_TOK_NSHAPE   9
#define TK_TOK_BPXMAX   64
#define TK_TOK_FORK_DEF 4
#define TK_TOK_FORK_MAX 6

typedef enum { TK_STREAM_TEXT = 0, TK_STREAM_TYPE = 1, TK_STREAM_SHAPE = 2 } tk_stream_t;
typedef enum { TK_FOCUS_NONE = 0, TK_FOCUS_TRUE = 1 } tk_focus_t;




typedef struct {

  int ngram_min, ngram_max;
  int n_types;
  tk_stream_t stream;
  int normalize;
  int bpx;
  int terminals;
  tk_focus_t focus;


  uint8_t b_bos, b_eos;
  uint8_t b_focus_open, b_focus_close;
  uint8_t b_type[TK_TOK_MAXTYPES + 2];
  uint8_t b_shape[TK_TOK_NSHAPE];
  int n_assigned;




  int bpx_fork_budget;
  int bpx_pool_n;
  uint8_t bpx_pool[TK_TOK_BPXMAX];
  int bpx_n;
  uint8_t bpx_lhs[TK_TOK_BPXMAX];
  uint8_t bpx_rhs[TK_TOK_BPXMAX];
  uint8_t bpx_sym[TK_TOK_BPXMAX];

  tk_iumap_t *ngram_map;
} tk_tokenizer_t;

static inline tk_tokenizer_t *tk_tokenizer_peek (lua_State *L, int i) {
  return (tk_tokenizer_t *) luaL_checkudata(L, i, TK_TOK_MT);
}

static luaL_Reg tk_tokenizer_mt_fns[];
static int tk_tokenizer_gc (lua_State *L);







typedef struct { uint8_t pool[64]; int n, i; } tk_assigner_t;

static void tk_assigner_init (tk_assigner_t *a, int normalize) {
  a->n = 0; a->i = 0;
  for (int b = 0x01; b <= 0x08; b++) a->pool[a->n++] = (uint8_t) b;
  for (int b = 0x0E; b <= 0x1F; b++) a->pool[a->n++] = (uint8_t) b;
  a->pool[a->n++] = 0x7F;
  if (normalize) {
    for (int b = 0x09; b <= 0x0D; b++) a->pool[a->n++] = (uint8_t) b;
    for (int b = 'A'; b <= 'Z'; b++) a->pool[a->n++] = (uint8_t) b;
  }
}


static uint8_t tk_assign (lua_State *L, tk_assigner_t *a, const char *role, int need_total) {
  if (a->i >= a->n)
    return (uint8_t) luaL_error(L,
      "tokenizer: out of marker bytes assigning %s (have %d, need %d; enable normalize to free 31 more)",
      role, a->n, need_total);
  return a->pool[a->i++];
}


static int tk_tokenizer_assign (lua_State *L, tk_tokenizer_t *t) {
  tk_assigner_t a;
  tk_assigner_init(&a, t->normalize);
  int nt = t->n_types;

  int need = (t->terminals ? 2 : 0)
           + (t->focus == TK_FOCUS_TRUE ? 2 : 0)
           + (t->stream == TK_STREAM_TYPE ? (nt + 2) : 0)
           + (t->stream == TK_STREAM_SHAPE ? TK_TOK_NSHAPE : 0);

  if (t->terminals) {
    t->b_bos = tk_assign(L, &a, "terminals", need);
    t->b_eos = tk_assign(L, &a, "terminals", need);
  }
  if (t->focus == TK_FOCUS_TRUE) {
    t->b_focus_open = tk_assign(L, &a, "focus", need);
    t->b_focus_close = tk_assign(L, &a, "focus", need);
  }
  if (t->stream == TK_STREAM_TYPE) {
    for (int k = 0; k < nt + 2; k++)
      t->b_type[k] = tk_assign(L, &a, "type", need);
  }
  if (t->stream == TK_STREAM_SHAPE) {
    for (int k = 0; k < TK_TOK_NSHAPE; k++)
      t->b_shape[k] = tk_assign(L, &a, "shape", need);
  }


  t->bpx_pool_n = 0;
  if (t->bpx) {
    for (int k = a.i; k < a.n; k++)
      t->bpx_pool[t->bpx_pool_n++] = a.pool[k];
    a.i = a.n;
  }
  t->n_assigned = a.i;
  return a.i;
}




static int tk_tokenizer_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  tk_tokenizer_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.ngram_min = (int) tk_lua_fcheckunsigned(L, 1, "tokenizer", "ngram_min");
  cfg.ngram_max = (int) tk_lua_fcheckunsigned(L, 1, "tokenizer", "ngram_max");
  if (cfg.ngram_min < 1 || cfg.ngram_min > cfg.ngram_max)
    return luaL_error(L, "tokenizer: need 1 <= ngram_min <= ngram_max");




  int want_types = tk_lua_foptboolean(L, 1, "tokenizer", "types", false);
  int want_shapes = tk_lua_foptboolean(L, 1, "tokenizer", "shapes", false);
  if (want_types && want_shapes)
    return luaL_error(L, "tokenizer: types and shapes are mutually exclusive");
  cfg.stream = want_types ? TK_STREAM_TYPE : (want_shapes ? TK_STREAM_SHAPE : TK_STREAM_TEXT);

  cfg.normalize = tk_lua_foptboolean(L, 1, "tokenizer", "normalize", false);
  cfg.bpx = tk_lua_foptboolean(L, 1, "tokenizer", "bpx", false);
  cfg.bpx_fork_budget = TK_TOK_FORK_DEF;
  lua_getfield(L, 1, "bpx_fork");
  if (!lua_isnil(L, -1)) cfg.bpx_fork_budget = (int) lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (cfg.bpx_fork_budget < 1) cfg.bpx_fork_budget = 1;
  if (cfg.bpx_fork_budget > TK_TOK_FORK_MAX) cfg.bpx_fork_budget = TK_TOK_FORK_MAX;
  cfg.terminals = tk_lua_foptboolean(L, 1, "tokenizer", "terminals", false);

  lua_getfield(L, 1, "focus");
  if (lua_isnil(L, -1) || (lua_isboolean(L, -1) && !lua_toboolean(L, -1))) cfg.focus = TK_FOCUS_NONE;
  else if (lua_isboolean(L, -1) && lua_toboolean(L, -1)) cfg.focus = TK_FOCUS_TRUE;
  else return luaL_error(L, "tokenizer: focus must be false|true");
  lua_pop(L, 1);


  lua_getfield(L, 1, "n_types");
  if (!lua_isnil(L, -1)) cfg.n_types = (int) lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (cfg.stream == TK_STREAM_TYPE && cfg.n_types <= 0)
    return luaL_error(L, "tokenizer: n_types required when types=true");
  if (cfg.n_types > TK_TOK_MAXTYPES)
    return luaL_error(L, "tokenizer: n_types exceeds ceiling %d", TK_TOK_MAXTYPES);


  if (cfg.normalize && cfg.stream != TK_STREAM_TEXT)
    return luaL_error(L, "tokenizer: normalize only valid on the text stream (no types/shapes)");

  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = NULL;
  tk_tokenizer_assign(L, t);


  if (t->bpx && t->bpx_pool_n == 0)
    return luaL_error(L, "tokenizer: bpx=true but no free bytes for merges (all consumed by roles)");
  return 1;
}




static int tk_tokenizer_n_tokens_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  lua_pushinteger(L, t->ngram_map ? (lua_Integer) tk_iumap_size(t->ngram_map) : 0);
  return 1;
}

static int tk_tokenizer_shrink_lua (lua_State *L) {
  tk_tokenizer_peek(L, 1);
  return 0;
}

static int tk_tokenizer_gc (lua_State *L) {
  tk_tokenizer_t *t = (tk_tokenizer_t *) luaL_checkudata(L, 1, TK_TOK_MT);
  if (t->ngram_map) { tk_iumap_destroy(t->ngram_map); t->ngram_map = NULL; }
  return 0;
}





static inline size_t tk_pack_ngrams (const uint8_t *d, size_t n_elems, int n, int64_t *out) {
  if (n_elems < (size_t) n) return 0;
  size_t count = n_elems - (size_t) n + 1;
  if (n <= 8) {
    uint64_t mask = (n < 8) ? ((1ULL << (n * 8)) - 1) : ~0ULL;
    uint64_t id = 0;
    for (int i = 0; i < n - 1; i++) id = (id << 8) | d[i];
    for (size_t i = 0; i < count; i++) {
      id = ((id << 8) | d[(size_t)(n - 1) + i]) & mask;
      out[i] = (int64_t) id;
    }
  } else {
    const uint64_t P = 0x9E3779B97F4A7C15ULL;
    uint64_t p_pow_n = 1;
    for (int j = 0; j < n - 1; j++) p_pow_n *= P;
    uint64_t h = 0;
    for (int j = 0; j < n; j++) h = h * P + d[j];
    out[0] = (int64_t) h;
    for (size_t i = 1; i < count; i++) {
      h = (h - d[i - 1] * p_pow_n) * P + d[i + (size_t) n - 1];
      out[i] = (int64_t) h;
    }
  }
  return count;
}

static inline size_t tk_pack_row (const uint8_t *buf, size_t len, int nmin, int nmax, int64_t *out) {
  size_t count = 0;
  for (int ng = nmin; ng <= nmax; ng++)
    count += tk_pack_ngrams(buf, len, ng, out + count);
  return count;
}

static int tk_i64_cmp (const void *a, const void *b) {
  int64_t x = *(const int64_t *) a, y = *(const int64_t *) b;
  return (x > y) - (x < y);
}







static inline uint8_t tk_scrub (uint8_t b) {
  if ((b >= 0x01 && b <= 0x08) || (b >= 0x0E && b <= 0x1F) || b == 0x7F) return ' ';
  return b;
}






static int tk_shape_class (const char *t, int64_t s, int64_t e) {
  int up = 0, lo = 0, dig = 0, pun = 0, oth = 0, first_up = 0, n = 0;
  for (int64_t i = s; i < e; i++) {
    uint8_t c = (uint8_t) t[i];
    if (n == 0 && c >= 'A' && c <= 'Z') first_up = 1;
    if (c >= 'A' && c <= 'Z') up++;
    else if (c >= 'a' && c <= 'z') lo++;
    else if (c >= '0' && c <= '9') dig++;
    else if (c < 0x80) pun++;
    else oth++;
    n++;
  }
  if (up && !lo && !dig && !pun && !oth) return (n == 1) ? 6 : 1;
  if (lo && !up && !dig && !pun && !oth) return 2;
  if (dig && !up && !lo && !pun && !oth) return 3;
  if (pun && !up && !lo && !dig && !oth) return 4;
  if (dig && (up || lo) && !oth) return 8;
  if (up && pun && !lo && !dig && !oth) return 7;
  if (first_up && lo && !dig && !oth) return 0;
  return 5;
}



static inline int tk_type_slot (int t, int n_types) {
  if (t < 0) return n_types + 1;
  if (t >= n_types) return n_types;
  return t;
}







typedef struct { uint8_t *buf; uint8_t *mask; size_t w; int norm; tk_norm_stream_t ns; } tk_render_t;
static inline void tk_render_init (tk_render_t *r, uint8_t *buf, uint8_t *mask, int norm) {
  r->buf = buf; r->mask = mask; r->w = 0; r->norm = norm;
  tk_norm_stream_init(&r->ns, buf);
}
static inline void tk_render_lit (tk_render_t *r, const char *text, size_t a, size_t b) {
  if (b <= a) return;
  size_t w0 = r->w;
  if (r->norm) { tk_norm_stream_run(&r->ns, text + a, b - a); r->w = r->ns.nlen; }
  else { for (size_t i = a; i < b; i++) r->buf[r->w++] = tk_scrub((uint8_t) text[i]); }
  if (r->mask) for (size_t i = w0; i < r->w; i++) r->mask[i] = 0;
}



static inline void tk_render_byte (tk_render_t *r, uint8_t byte, int is_marker) {
  size_t w0 = r->w;
  if (r->norm) { tk_norm_stream_mark(&r->ns, byte); r->w = r->ns.nlen; }
  else r->buf[r->w++] = byte;
  if (r->mask) r->mask[w0] = (uint8_t) is_marker;
}
static inline void tk_render_mark (tk_render_t *r, uint8_t byte) { tk_render_byte(r, byte, 1); }
static inline void tk_render_content (tk_render_t *r, uint8_t byte) { tk_render_byte(r, byte, 0); }
static inline size_t tk_render_finish (tk_render_t *r) {
  if (r->norm) r->w = tk_norm_stream_finish(&r->ns);
  return r->w;
}






static size_t tk_render_row (
  tk_tokenizer_t *t, uint8_t *rowbuf, uint8_t *maskbuf,
  const char *text, size_t tlen, int per_span, size_t s, size_t e,
  int64_t c0, int64_t c1,
  tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty, int suppress_focus)
{
  (void) per_span;
  tk_focus_t efocus = suppress_focus ? TK_FOCUS_NONE : t->focus;
  tk_render_t r;
  tk_render_init(&r, rowbuf, maskbuf, (t->stream == TK_STREAM_TEXT && t->normalize));
  if (t->terminals) tk_render_mark(&r, t->b_bos);

  if (t->stream == TK_STREAM_TEXT) {
    if (efocus == TK_FOCUS_NONE) {
      tk_render_lit(&r, text, 0, tlen);
    } else {
      tk_render_lit(&r, text, 0, s);
      tk_render_mark(&r, t->b_focus_open);
      tk_render_lit(&r, text, s, e);
      tk_render_mark(&r, t->b_focus_close);
      tk_render_lit(&r, text, e, tlen);
    }

  } else if (t->stream == TK_STREAM_TYPE) {
    bool fo_done = false, fc_done = false;
    for (int64_t cj = c0; cj < c1; cj++) {
      size_t cstart = (size_t) cs->a[cj];
      int slot = tk_type_slot(cty ? (int) cty->a[cj] : t->n_types, t->n_types);
      if (efocus != TK_FOCUS_NONE && !fo_done && cstart >= s) { tk_render_mark(&r, t->b_focus_open); fo_done = true; }
      if (efocus != TK_FOCUS_NONE && fo_done && !fc_done && cstart >= e) { tk_render_mark(&r, t->b_focus_close); fc_done = true; }
      tk_render_content(&r, t->b_type[slot]);
    }
    if (efocus != TK_FOCUS_NONE && fo_done && !fc_done) tk_render_mark(&r, t->b_focus_close);

  } else {
    bool fo_done = false, fc_done = false;
    size_t p = 0;
    while (p < tlen) {
      while (p < tlen && text[p] == ' ') p++;
      if (p >= tlen) break;
      size_t ws = p;
      while (p < tlen && text[p] != ' ') p++;
      int sc = tk_shape_class(text, (int64_t) ws, (int64_t) p);
      if (efocus != TK_FOCUS_NONE && !fo_done && ws >= s) { tk_render_mark(&r, t->b_focus_open); fo_done = true; }
      if (efocus != TK_FOCUS_NONE && fo_done && !fc_done && ws >= e) { tk_render_mark(&r, t->b_focus_close); fc_done = true; }
      tk_render_content(&r, t->b_shape[sc]);
    }
    if (efocus != TK_FOCUS_NONE && fo_done && !fc_done) tk_render_mark(&r, t->b_focus_close);
  }

  if (t->terminals) tk_render_mark(&r, t->b_eos);
  return tk_render_finish(&r);
}










static inline int64_t tk_gram_one (const uint8_t *b, int n) {
  if (n <= 8) {
    uint64_t mask = (n < 8) ? ((1ULL << (n * 8)) - 1) : ~0ULL;
    uint64_t id = 0;
    for (int i = 0; i < n; i++) id = (id << 8) | b[i];
    return (int64_t) (id & mask);
  } else {
    const uint64_t P = 0x9E3779B97F4A7C15ULL;
    uint64_t h = 0;
    for (int j = 0; j < n; j++) h = h * P + b[j];
    return (int64_t) h;
  }
}





static size_t tk_bpx_compress (tk_tokenizer_t *t, const uint8_t *in, size_t len, uint8_t *out) {
  memcpy(out, in, len);
  for (int i = 0; i < t->bpx_n; i++) {
    uint8_t a = t->bpx_lhs[i], b = t->bpx_rhs[i], sym = t->bpx_sym[i];
    size_t w = 0;
    for (size_t j = 0; j < len; ) {
      if (j + 1 < len && out[j] == a && out[j + 1] == b) { out[w++] = sym; j += 2; }
      else out[w++] = out[j++];
    }
    len = w;
  }
  return len;
}




static void tk_bpx_tables (tk_tokenizer_t *t, int cap, int16_t *sym2idx, uint8_t *expbuf, int *exp_len) {
  for (int b = 0; b < 256; b++) {
    sym2idx[b] = -1;
    expbuf[b * cap] = (uint8_t) b;
    exp_len[b] = 1;
  }
  for (int i = 0; i < t->bpx_n; i++) {
    uint8_t s = t->bpx_sym[i], a = t->bpx_lhs[i], b = t->bpx_rhs[i];
    sym2idx[s] = (int16_t) i;
    int w = 0;
    for (int k = 0; k < exp_len[a] && w < cap; k++) expbuf[s * cap + w++] = expbuf[a * cap + k];
    for (int k = 0; k < exp_len[b] && w < cap; k++) expbuf[s * cap + w++] = expbuf[b * cap + k];
    exp_len[s] = w;
  }
}




typedef struct {
  const uint8_t *units; int ulen;
  const int16_t *sym2idx; const uint8_t *expbuf; const int *exp_len; int cap;
  int nmax; uint8_t *cur;
  uint8_t *vars; int *vlen; int nvars; int varcap;
} tk_bpx_enum_t;

static void tk_bpx_dfs (tk_bpx_enum_t *E, int ui, int curlen) {
  if (E->nvars >= E->varcap) return;
  if (curlen >= E->nmax || ui >= E->ulen) {
    if (curlen > 0) {
      memcpy(E->vars + (size_t) E->nvars * (size_t) E->nmax, E->cur, (size_t) curlen);
      E->vlen[E->nvars++] = curlen;
    }
    return;
  }
  uint8_t b = E->units[ui];
  int mi = E->sym2idx[b];
  if (mi < 0) {
    E->cur[curlen] = b;
    tk_bpx_dfs(E, ui + 1, curlen + 1);
  } else {
    E->cur[curlen] = b;
    tk_bpx_dfs(E, ui + 1, curlen + 1);
    int add = E->exp_len[b];




    if (curlen + add > E->nmax) add = E->nmax - curlen;
    memcpy(E->cur + curlen, E->expbuf + (size_t) b * (size_t) E->cap, (size_t) add);
    tk_bpx_dfs(E, ui + 1, curlen + add);
  }
}













static size_t tk_bpx_pack_row (
  tk_tokenizer_t *t, const int16_t *sym2idx, const uint8_t *expbuf, const int *exp_len, int cap,
  const uint8_t *rowbuf, const uint8_t *mask, size_t w,
  uint8_t *units_buf, uint8_t *cur_buf, uint8_t *vars_buf, int *vlen_buf, int64_t *aids, int64_t *packed)
{
  int nmin = t->ngram_min, nmax = t->ngram_max, F = t->bpx_fork_budget;
  int varcap = 1 << F; if (varcap > (1 << TK_TOK_FORK_MAX)) varcap = 1 << TK_TOK_FORK_MAX;


  size_t count = tk_pack_row(rowbuf, w, nmin, nmax, packed);


  int ulen = 0;
  for (size_t i = 0; i < w; ) {
    if (mask[i]) { units_buf[ulen++] = rowbuf[i]; i++; continue; }
    size_t j = i; while (j < w && !mask[j]) j++;
    ulen += (int) tk_bpx_compress(t, rowbuf + i, j - i, units_buf + ulen);
    i = j;
  }


  for (int a = 0; a < ulen; a++) {
    int symc = 0, hi = a + nmax; if (hi > ulen) hi = ulen;
    for (int k = a; k < hi; k++) if (sym2idx[units_buf[k]] >= 0) symc++;
    if (symc == 0) continue;

    int nv;
    if (symc > F) {
      int L = 0;
      for (int k = a; k < ulen && L < nmax; k++) vars_buf[L++] = units_buf[k];
      vlen_buf[0] = L; nv = 1;
    } else {
      tk_bpx_enum_t E = { units_buf, ulen, sym2idx, expbuf, exp_len, cap,
        nmax, cur_buf, vars_buf, vlen_buf, 0, varcap };
      tk_bpx_dfs(&E, a, 0);
      nv = E.nvars;
    }

    int na = 0;
    for (int v = 0; v < nv; v++) {
      const uint8_t *s = vars_buf + (size_t) v * (size_t) nmax; int Lv = vlen_buf[v];
      for (int n = nmin; n <= nmax && n <= Lv; n++) {
        int has = 0;
        for (int q = 0; q < n; q++) if (sym2idx[s[q]] >= 0) { has = 1; break; }
        if (has) aids[na++] = tk_gram_one(s, n);
      }
    }
    qsort(aids, (size_t) na, sizeof(int64_t), tk_i64_cmp);
    for (int i = 0; i < na; i++)
      if (i == 0 || aids[i] != aids[i - 1]) packed[count++] = aids[i];
  }
  return count;
}






static void tk_bpx_learn (
  tk_tokenizer_t *t, int64_t n_samples,
  const char **text_ptrs, const size_t *text_lens,
  tk_ivec_t *co, tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty,
  uint8_t *rowbuf, uint8_t *maskbuf)
{



  size_t ccap = 4096, clen = 0;
  uint8_t *corpus = (uint8_t *) malloc(ccap);
  size_t rcap = 256, rn = 0;
  size_t *run_off = (size_t *) malloc((rcap + 1) * sizeof(size_t));
  run_off[0] = 0;

  for (int64_t d = 0; d < n_samples; d++) {
    const char *text = text_ptrs[d]; size_t tlen = text_lens[d];
    int64_t c0 = co ? co->a[d] : 0, c1 = co ? co->a[d + 1] : 0;


    size_t w = tk_render_row(t, rowbuf, maskbuf, text, tlen, 0, tlen, tlen, c0, c1, cs, ce, cty, 1);
    for (size_t i = 0; i < w; ) {
      if (maskbuf[i]) { i++; continue; }
      size_t j = i; while (j < w && !maskbuf[j]) j++;
      size_t rl = j - i;
      if (clen + rl > ccap) { while (clen + rl > ccap) ccap *= 2; corpus = (uint8_t *) realloc(corpus, ccap); }
      memcpy(corpus + clen, rowbuf + i, rl); clen += rl;
      if (rn + 1 > rcap) { rcap *= 2; run_off = (size_t *) realloc(run_off, (rcap + 1) * sizeof(size_t)); }
      run_off[++rn] = clen;
      i = j;
    }
  }



  int *counts = (int *) malloc(256 * 256 * sizeof(int));
  t->bpx_n = 0;
  while (t->bpx_n < t->bpx_pool_n) {
    memset(counts, 0, 256 * 256 * sizeof(int));
    for (size_t r = 0; r < rn; r++)
      for (size_t i = run_off[r]; i + 1 < run_off[r + 1]; i++)
        counts[((int) corpus[i] << 8) | corpus[i + 1]]++;
    int best = -1, bestc = 1;
    for (int p = 0; p < 256 * 256; p++)
      if (counts[p] > bestc) { bestc = counts[p]; best = p; }
    if (best < 0) break;
    uint8_t a = (uint8_t) (best >> 8), b = (uint8_t) (best & 0xFF);
    uint8_t sym = t->bpx_pool[t->bpx_n];
    t->bpx_lhs[t->bpx_n] = a; t->bpx_rhs[t->bpx_n] = b; t->bpx_sym[t->bpx_n] = sym;
    t->bpx_n++;
    size_t wpos = 0;
    for (size_t r = 0; r < rn; r++) {
      size_t rs = run_off[r], re = run_off[r + 1];
      run_off[r] = wpos;
      for (size_t i = rs; i < re; ) {
        if (i + 1 < re && corpus[i] == a && corpus[i + 1] == b) { corpus[wpos++] = sym; i += 2; }
        else corpus[wpos++] = corpus[i++];
      }
    }
    run_off[rn] = wpos;
  }
  free(corpus); free(run_off); free(counts);
}





static int tk_tokenizer_tokenize_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  bool grow = tk_lua_foptboolean(L, 2, "tokenize", "grow", false);
  int64_t n_samples = (int64_t) tk_lua_fcheckunsigned(L, 2, "tokenize", "n_samples");





  lua_getfield(L, 2, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int texts_idx = lua_gettop(L);



  tk_ivec_t *fo = NULL, *fs = NULL, *fe = NULL;
  lua_getfield(L, 2, "focus");
  if (!lua_isnil(L, -1)) {
    luaL_checktype(L, -1, LUA_TTABLE);
    lua_getfield(L, -1, "offsets"); fo = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
    lua_getfield(L, -1, "starts");  fs = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
    lua_getfield(L, -1, "ends");    fe = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  }
  lua_pop(L, 1);
  tk_ivec_t *co = NULL, *cs = NULL, *ce = NULL, *cty = NULL;
  lua_getfield(L, 2, "types");
  if (!lua_isnil(L, -1)) {
    luaL_checktype(L, -1, LUA_TTABLE);
    lua_getfield(L, -1, "offsets"); co = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
    lua_getfield(L, -1, "starts");  cs = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
    lua_getfield(L, -1, "ends");    ce = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
    lua_getfield(L, -1, "types");   cty = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  }
  lua_pop(L, 1);


  bool per_span = (fo != NULL);
  if (per_span && (!fs || !fe))
    return luaL_error(L, "tokenizer: focus.offsets given but focus.starts/ends missing");
  if (t->stream == TK_STREAM_TYPE && (!co || !cs || !ce || !cty))
    return luaL_error(L, "tokenizer: types=true requires types.offsets/starts/ends/types");
  if (t->focus != TK_FOCUS_NONE && !per_span)
    return luaL_error(L, "tokenizer: focus set at create but no focus spans passed");


  const char **text_ptrs = (const char **) malloc((size_t) n_samples * sizeof(char *));
  size_t *text_lens = (size_t *) malloc((size_t) n_samples * sizeof(size_t));
  for (int64_t d = 0; d < n_samples; d++) {
    lua_rawgeti(L, texts_idx, (int) (d + 1));
    text_ptrs[d] = lua_tolstring(L, -1, &text_lens[d]);
    lua_pop(L, 1);
  }

  int64_t n_rows = per_span ? fo->a[(int64_t)(fo->n - 1)] : n_samples;



  size_t maxbuf = 8;
  for (int64_t d = 0; d < n_samples; d++) {
    size_t need = text_lens[d] + 8;
    if (need > maxbuf) maxbuf = need;
  }
  if (co && t->stream == TK_STREAM_TYPE) {
    for (int64_t d = 0; d + 1 < (int64_t) co->n; d++) {
      size_t nc = (size_t) (co->a[d + 1] - co->a[d]) + 8;
      if (nc > maxbuf) maxbuf = nc;
    }
  }
  size_t nrange = (size_t) (t->ngram_max - t->ngram_min + 1);
  uint8_t *rowbuf = (uint8_t *) malloc(maxbuf);



  int bpx_varcap = t->bpx ? (1 << t->bpx_fork_budget) : 1;
  if (bpx_varcap < 2) bpx_varcap = 2;

  size_t packed_cap = nrange * maxbuf * (size_t) (t->bpx ? bpx_varcap + 1 : 1);
  int64_t *packed = (int64_t *) malloc(packed_cap * sizeof(int64_t));
  uint8_t *maskbuf = NULL, *units_buf = NULL, *cur_buf = NULL, *vars_buf = NULL, *expbuf = NULL;
  int *vlen_buf = NULL, *exp_len = NULL; int16_t *sym2idx = NULL; int64_t *aids = NULL;
  int exp_cap = t->ngram_max;
  if (t->bpx) {
    maskbuf = (uint8_t *) malloc(maxbuf);
    units_buf = (uint8_t *) malloc(maxbuf);
    cur_buf = (uint8_t *) malloc((size_t) t->ngram_max);
    vars_buf = (uint8_t *) malloc((size_t) bpx_varcap * (size_t) t->ngram_max);
    vlen_buf = (int *) malloc((size_t) bpx_varcap * sizeof(int));
    aids = (int64_t *) malloc((size_t) bpx_varcap * nrange * sizeof(int64_t));
    sym2idx = (int16_t *) malloc(256 * sizeof(int16_t));
    expbuf = (uint8_t *) malloc(256 * (size_t) exp_cap);
    exp_len = (int *) malloc(256 * sizeof(int));
  }

  if (grow && !t->ngram_map) t->ngram_map = tk_iumap_create(NULL, 0);
  if (!t->ngram_map) { free(rowbuf); free(packed); free(text_ptrs); free(text_lens);
    free(maskbuf); free(units_buf); free(cur_buf); free(vars_buf); free(vlen_buf);
    free(aids); free(sym2idx); free(expbuf); free(exp_len);
    return luaL_error(L, "tokenizer: frozen tokenize before any grow=true pass"); }
  tk_iumap_t *map = t->ngram_map;
  uint32_t mend = tk_iumap_end(map);


  if (t->bpx) {
    if (grow && t->bpx_n == 0 && t->bpx_pool_n > 0)
      tk_bpx_learn(t, n_samples, text_ptrs, text_lens,
        co, cs, ce, cty, rowbuf, maskbuf);
    tk_bpx_tables(t, exp_cap, sym2idx, expbuf, exp_len);
  }

  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t) (n_rows + 1));
  offsets->n = (uint64_t) (n_rows + 1); offsets->a[0] = 0;
  tk_svec_t *toks = tk_svec_create(L, 0);
  tk_fvec_t *vals = tk_fvec_create(L, 0);

  int64_t row = 0;
  for (int64_t d = 0; d < n_samples; d++) {
    const char *text = text_ptrs[d];
    size_t tlen = text_lens[d];
    int64_t fa = per_span ? fo->a[d] : 0;
    int64_t fb = per_span ? fo->a[d + 1] : 1;
    int64_t c0 = co ? co->a[d] : 0, c1 = co ? co->a[d + 1] : 0;

    for (int64_t fi = fa; fi < fb; fi++) {
      size_t s = per_span ? (size_t) fs->a[fi] : 0;
      size_t e = per_span ? (size_t) fe->a[fi] : tlen;
      size_t w = tk_render_row(t, rowbuf, maskbuf, text, tlen, per_span, s, e, c0, c1, cs, ce, cty, 0);


      size_t count = t->bpx
        ? tk_bpx_pack_row(t, sym2idx, expbuf, exp_len, exp_cap, rowbuf, maskbuf, w,
            units_buf, cur_buf, vars_buf, vlen_buf, aids, packed)
        : tk_pack_row(rowbuf, w, t->ngram_min, t->ngram_max, packed);
      int64_t nv = 0;
      for (size_t i = 0; i < count; i++) {
        if (grow) {
          int absent;
          uint32_t it = tk_iumap_put(map, packed[i], &absent);
          if (absent) tk_iumap_setval(map, it, (int64_t) tk_iumap_size(map) - 1);
          packed[nv++] = tk_iumap_val(map, it);
        } else {
          uint32_t it = tk_iumap_get(map, packed[i]);
          if (it != mend) packed[nv++] = tk_iumap_val(map, it);
        }
      }
      qsort(packed, (size_t) nv, sizeof(int64_t), tk_i64_cmp);
      for (int64_t i = 0; i < nv; ) {
        int64_t tk = packed[i]; float c = 0.0f;
        while (i < nv && packed[i] == tk) { c += 1.0f; i++; }
        tk_svec_push(toks, (int32_t) tk);
        tk_fvec_push(vals, c);
      }
      offsets->a[++row] = (int64_t) toks->n;
    }
  }

  free(rowbuf); free(packed); free(text_ptrs); free(text_lens);
  free(maskbuf); free(units_buf); free(cur_buf); free(vars_buf); free(vlen_buf);
  free(aids); free(sym2idx); free(expbuf); free(exp_len);
  lua_pushvalue(L, lua_gettop(L) - 2);
  lua_pushvalue(L, lua_gettop(L) - 2);
  lua_pushvalue(L, lua_gettop(L) - 2);
  return 3;
}







static int tk_tokenizer_persist_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKtk", 1, 4, fh);
  uint8_t version = 4;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  size_t cfgsz = offsetof(tk_tokenizer_t, ngram_map);
  tk_lua_fwrite(L, t, 1, cfgsz, fh);
  uint8_t has_map = t->ngram_map ? 1 : 0;
  tk_lua_fwrite(L, &has_map, sizeof(uint8_t), 1, fh);
  if (has_map) tk_iumap_persist(L, t->ngram_map, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

static int tk_tokenizer_load_lua (lua_State *L) {
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 1), "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKtk", 4) != 0) { tk_lua_fclose(L, fh); return luaL_error(L, "tokenizer.load: bad magic"); }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 4) { tk_lua_fclose(L, fh);
    return luaL_error(L, "tokenizer.load: unsupported version %d (old layout; refit required)", (int) version); }
  tk_tokenizer_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  size_t cfgsz = offsetof(tk_tokenizer_t, ngram_map);
  tk_lua_fread(L, &cfg, 1, cfgsz, fh);
  uint8_t has_map;
  tk_lua_fread(L, &has_map, sizeof(uint8_t), 1, fh);



  tk_iumap_t *map = has_map ? tk_iumap_load(NULL, fh) : NULL;
  tk_lua_fclose(L, fh);




  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = map;




  tk_tokenizer_t chk = cfg; chk.ngram_map = NULL;
  tk_tokenizer_assign(L, &chk);
  if (memcmp(&cfg, &chk, offsetof(tk_tokenizer_t, bpx_n)) != 0)
    return luaL_error(L, "tokenizer.load: byte-table drift (assigner changed since persist; refit required)");
  return 1;
}







static int tk_tokenizer_tokenize_raw_lua (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n_samples = (int64_t) tk_lua_fcheckunsigned(L, 1, "tokenize_raw", "n_samples");
  int ngram_min = (int) tk_lua_fcheckunsigned(L, 1, "tokenize_raw", "ngram_min");
  int ngram_max = (int) tk_lua_fcheckunsigned(L, 1, "tokenize_raw", "ngram_max");
  if (ngram_min < 1 || ngram_min > ngram_max)
    return luaL_error(L, "tokenize_raw: need 1 <= ngram_min <= ngram_max");
  int normalize = tk_lua_foptboolean(L, 1, "tokenize_raw", "normalize", false);

  lua_getfield(L, 1, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int texts_idx = lua_gettop(L);

  size_t maxlen = 0;
  for (int64_t d = 0; d < n_samples; d++) {
    lua_rawgeti(L, texts_idx, (int) (d + 1));
    size_t len = 0; lua_tolstring(L, -1, &len);
    if (len > maxlen) maxlen = len;
    lua_pop(L, 1);
  }

  size_t nrange = (size_t) (ngram_max - ngram_min + 1);
  uint8_t *buf = (uint8_t *) malloc(maxlen + 1);
  int64_t *packed = (int64_t *) malloc(nrange * (maxlen + 1) * sizeof(int64_t));
  if (!buf || !packed) { free(buf); free(packed); return luaL_error(L, "tokenize_raw: out of memory"); }

  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t) (n_samples + 1));
  offsets->n = (uint64_t) (n_samples + 1); offsets->a[0] = 0;
  tk_ivec_t *tokens = tk_ivec_create(L, 0);
  tk_fvec_t *values = tk_fvec_create(L, 0);

  for (int64_t d = 0; d < n_samples; d++) {
    lua_rawgeti(L, texts_idx, (int) (d + 1));
    size_t tlen; const char *text = lua_tolstring(L, -1, &tlen);
    lua_pop(L, 1);

    size_t blen;
    if (normalize) {
      tk_norm_stream_t ns;
      tk_norm_stream_init(&ns, buf);
      tk_norm_stream_run(&ns, text, tlen);
      blen = tk_norm_stream_finish(&ns);
    } else {
      memcpy(buf, text, tlen);
      blen = tlen;
    }

    size_t count = tk_pack_row(buf, blen, ngram_min, ngram_max, packed);
    qsort(packed, count, sizeof(int64_t), tk_i64_cmp);
    for (size_t i = 0; i < count; ) {
      int64_t key = packed[i]; float c = 0.0f;
      while (i < count && packed[i] == key) { c += 1.0f; i++; }
      tk_ivec_push(tokens, key);
      tk_fvec_push(values, c);
    }
    offsets->a[d + 1] = (int64_t) tokens->n;
  }

  free(buf); free(packed);
  return 3;
}

static luaL_Reg tk_tokenizer_mt_fns[] = {
  { "tokenize", tk_tokenizer_tokenize_lua },
  { "n_tokens", tk_tokenizer_n_tokens_lua },
  { "persist", tk_tokenizer_persist_lua },
  { "shrink", tk_tokenizer_shrink_lua },
  { NULL, NULL }
};

static luaL_Reg tk_tokenizer_fns[] = {
  { "create", tk_tokenizer_create_lua },
  { "load", tk_tokenizer_load_lua },
  { "tokenize_raw", tk_tokenizer_tokenize_raw_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_tokenizer (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_tokenizer_fns, 0);
  return 1;
}

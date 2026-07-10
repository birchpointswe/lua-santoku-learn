#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <santoku/lua/utils.h>
#include <santoku/learn/normalize.h>
#include <santoku/ivec/ext.h>
#include <santoku/svec.h>
#include <santoku/fvec.h>
#include <santoku/iumap/ext.h>
#include <santoku/csr.h>
#include <santoku/spans.h>

#define TK_TOK_MT "tk_tokenizer_t"
#define TK_TOK_MAXTYPES 250

typedef enum { TK_STREAM_TEXT = 0, TK_STREAM_TYPE = 1 } tk_stream_t;
typedef enum { TK_FOCUS_NONE = 0, TK_FOCUS_TRUE = 1 } tk_focus_t;

typedef struct {
  int ngram_min, ngram_max;
  int n_types;
  tk_stream_t stream;
  int normalize;
  int terminals;
  tk_focus_t focus;
  int words;

  uint8_t b_bos, b_eos;
  uint8_t b_focus_open, b_focus_close;
  uint8_t b_sep;
  uint8_t b_type[TK_TOK_MAXTYPES + 2];
  int n_assigned;
  uint8_t content[256];

  tk_iumap_t *ngram_map;
} tk_tokenizer_t;

static inline tk_tokenizer_t *tk_tokenizer_peek (lua_State *L, int i) {
  return (tk_tokenizer_t *) luaL_checkudata(L, i, TK_TOK_MT);
}

static luaL_Reg tk_tokenizer_mt_fns[];
static int tk_tokenizer_gc (lua_State *L);

typedef struct { uint8_t pool[256]; int n, i; } tk_assigner_t;

static void tk_assigner_init (tk_assigner_t *a, int normalize, int full) {
  a->n = 0; a->i = 0;
  if (full) {
    for (int b = 0x01; b <= 0xFF; b++) a->pool[a->n++] = (uint8_t) b;
    return;
  }
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
  tk_assigner_init(&a, t->normalize, t->stream != TK_STREAM_TEXT);
  int nt = t->n_types;
  int has_boundary = 0;
  for (int i = 0; i < 256; i++) if (t->content[i]) { has_boundary = 1; break; }
  int flatten = has_boundary && !t->words;
  int need = (t->terminals ? 2 : 0)
           + (t->focus == TK_FOCUS_TRUE ? 2 : 0)
           + (flatten ? 1 : 0)
           + (t->stream == TK_STREAM_TYPE ? (nt + 2) : 0);

  if (t->terminals) {
    t->b_bos = tk_assign(L, &a, "terminals", need);
    t->b_eos = tk_assign(L, &a, "terminals", need);
  }
  if (t->focus == TK_FOCUS_TRUE) {
    t->b_focus_open = tk_assign(L, &a, "focus", need);
    t->b_focus_close = tk_assign(L, &a, "focus", need);
  }
  if (flatten) t->b_sep = tk_assign(L, &a, "boundary", need);
  if (t->stream == TK_STREAM_TYPE) {
    for (int k = 0; k < nt + 2; k++)
      t->b_type[k] = tk_assign(L, &a, "type", need);
  }
  t->n_assigned = a.i;
  return a.i;
}

static int tk_tokenizer_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  tk_tokenizer_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.ngram_min = (int) tk_lua_foptunsigned(L, 1, "tokenizer", "ngram_min", 1);
  cfg.ngram_max = (int) tk_lua_fcheckunsigned(L, 1, "tokenizer", "ngram_max");
  if (cfg.ngram_min < 1 || cfg.ngram_min > cfg.ngram_max)
    return luaL_error(L, "tokenizer: need 1 <= ngram_min <= ngram_max");

  int want_types = tk_lua_foptboolean(L, 1, "tokenizer", "types", false);
  cfg.stream = want_types ? TK_STREAM_TYPE : TK_STREAM_TEXT;

  cfg.normalize = tk_lua_foptboolean(L, 1, "tokenizer", "normalize", false);
  cfg.terminals = tk_lua_foptboolean(L, 1, "tokenizer", "terminals", false);
  cfg.words = tk_lua_foptboolean(L, 1, "tokenizer", "words", false);

  int has_boundary = 0;
  lua_getfield(L, 1, "word_characters");
  if (!lua_isnil(L, -1)) {
    size_t blen; const char *bs = lua_tolstring(L, -1, &blen);
    for (size_t i = 0; i < blen; i++) { cfg.content[(uint8_t) bs[i]] = 1; has_boundary = 1; }
  }
  lua_pop(L, 1);
  if (cfg.words && !has_boundary)
    return luaL_error(L, "tokenizer: words=true requires word_characters");

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
    return luaL_error(L, "tokenizer: normalize only valid on the text stream (not types)");

  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = NULL;
  tk_tokenizer_assign(L, t);
  return 1;
}

static int tk_tokenizer_n_tokens_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  lua_pushinteger(L, t->ngram_map ? (lua_Integer) tk_iumap_size(t->ngram_map) : 0);
  return 1;
}

static int tk_tokenizer_gc (lua_State *L) {
  tk_tokenizer_t *t = (tk_tokenizer_t *) luaL_checkudata(L, 1, TK_TOK_MT);
  if (t->ngram_map) { tk_iumap_destroy(t->ngram_map); t->ngram_map = NULL; }
  return 0;
}

#define tk_pack_roll(a, n, count, out) do { \
  const uint64_t P = 0x9E3779B97F4A7C15ULL; \
  uint64_t p_pow_n = 1; \
  for (int j = 0; j < n - 1; j++) p_pow_n *= P; \
  uint64_t h = 0; \
  for (int j = 0; j < n; j++) h = h * P + (uint64_t) a[j]; \
  out[0] = (int64_t) h; \
  for (size_t i = 1; i < count; i++) { \
    h = (h - (uint64_t) a[i - 1] * p_pow_n) * P + (uint64_t) a[i + (size_t) n - 1]; \
    out[i] = (int64_t) h; \
  } \
} while (0)

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
    tk_pack_roll(d, n, count, out);
  }
  return count;
}

static inline size_t tk_pack_row (const uint8_t *buf, size_t len, int nmin, int nmax, int64_t *out) {
  size_t count = 0;
  for (int ng = nmin; ng <= nmax; ng++)
    count += tk_pack_ngrams(buf, len, ng, out + count);
  return count;
}

static inline size_t tk_pack_symbols_ng (const int64_t *sym, size_t n_elems, int n, int64_t *out) {
  if (n_elems < (size_t) n) return 0;
  size_t count = n_elems - (size_t) n + 1;
  tk_pack_roll(sym, n, count, out);
  return count;
}

static inline size_t tk_pack_symbols (const int64_t *sym, size_t len, int nmin, int nmax, int64_t *out) {
  size_t count = 0;
  for (int ng = nmin; ng <= nmax; ng++)
    count += tk_pack_symbols_ng(sym, len, ng, out + count);
  return count;
}

static inline size_t tk_boundary_flatten (uint8_t *buf, size_t w,
    const uint8_t *content, const uint8_t *is_marker, uint8_t b_sep) {
  size_t o = 0; int in_sep = 0;
  for (size_t i = 0; i < w; i++) {
    uint8_t c = buf[i];
    if (content[c] || is_marker[c]) { buf[o++] = c; in_sep = 0; }
    else if (!in_sep) { buf[o++] = b_sep; in_sep = 1; }
  }
  return o;
}

static inline size_t tk_parse_symbols (const uint8_t *buf, size_t w,
    const uint8_t *content, const uint8_t *is_marker, int64_t *out) {
  size_t o = 0, i = 0;
  while (i < w) {
    if (content[buf[i]]) {
      uint64_t h = 0; size_t j = i;
      while (j < w && content[buf[j]]) { h = h * 0x9E3779B97F4A7C15ULL + (uint64_t) buf[j]; j++; }
      out[o++] = (int64_t) h; i = j;
    } else if (is_marker[buf[i]]) {
      out[o++] = (int64_t) buf[i]; i++;
    } else {
      i++;
    }
  }
  return o;
}

static inline uint8_t tk_scrub (uint8_t b) {
  if ((b >= 0x01 && b <= 0x08) || (b >= 0x0E && b <= 0x1F) || b == 0x7F) return ' ';
  return b;
}

static inline int tk_type_slot (int t, int n_types) {
  if (t < 0) return n_types + 1;
  if (t >= n_types) return n_types;
  return t;
}

typedef struct { uint8_t *buf; size_t w; int norm; tk_norm_stream_t ns; } tk_render_t;
static inline void tk_render_init (tk_render_t *r, uint8_t *buf, int norm) {
  r->buf = buf; r->w = 0; r->norm = norm;
  tk_norm_stream_init(&r->ns, buf);
}
static inline void tk_render_lit (tk_render_t *r, const char *text, size_t a, size_t b) {
  if (b <= a) return;
  if (r->norm) { tk_norm_stream_run(&r->ns, text + a, b - a); r->w = r->ns.nlen; }
  else { for (size_t i = a; i < b; i++) r->buf[r->w++] = tk_scrub((uint8_t) text[i]); }
}
static inline void tk_render_byte (tk_render_t *r, uint8_t byte) {
  if (r->norm) { tk_norm_stream_mark(&r->ns, byte); r->w = r->ns.nlen; }
  else r->buf[r->w++] = byte;
}
static inline size_t tk_render_finish (tk_render_t *r) {
  if (r->norm) r->w = tk_norm_stream_finish(&r->ns);
  return r->w;
}

static size_t tk_render_row (
  tk_tokenizer_t *t, uint8_t *rowbuf,
  const char *text, size_t tlen, size_t s, size_t e,
  int64_t c0, int64_t c1, tk_ivec_t *cs, tk_ivec_t *cty)
{
  if (s > tlen) s = tlen;
  if (e > tlen) e = tlen;
  if (s > e) s = e;
  tk_focus_t efocus = t->focus;
  tk_render_t r;
  tk_render_init(&r, rowbuf, (t->stream == TK_STREAM_TEXT && t->normalize));
  if (t->terminals) tk_render_byte(&r, t->b_bos);

  if (t->stream == TK_STREAM_TEXT) {
    if (efocus == TK_FOCUS_NONE) {
      tk_render_lit(&r, text, 0, tlen);
    } else {
      tk_render_lit(&r, text, 0, s);
      tk_render_byte(&r, t->b_focus_open);
      tk_render_lit(&r, text, s, e);
      tk_render_byte(&r, t->b_focus_close);
      tk_render_lit(&r, text, e, tlen);
    }

  } else if (t->stream == TK_STREAM_TYPE) {
    bool fo_done = false, fc_done = false;
    for (int64_t cj = c0; cj < c1; cj++) {
      size_t cstart = (size_t) cs->a[cj];
      int slot = tk_type_slot(cty ? (int) cty->a[cj] : t->n_types, t->n_types);
      if (efocus != TK_FOCUS_NONE && !fo_done && cstart >= s) { tk_render_byte(&r, t->b_focus_open); fo_done = true; }
      if (efocus != TK_FOCUS_NONE && fo_done && !fc_done && cstart >= e) { tk_render_byte(&r, t->b_focus_close); fc_done = true; }
      tk_render_byte(&r, t->b_type[slot]);
    }
    if (efocus != TK_FOCUS_NONE && fo_done && !fc_done) tk_render_byte(&r, t->b_focus_close);
  }

  if (t->terminals) tk_render_byte(&r, t->b_eos);
  return tk_render_finish(&r);
}

typedef struct { uint8_t *rb; int64_t *pk; int64_t *sm; } tk_scratch_t;

static inline void tk_scratch_init (tk_scratch_t *sc, size_t maxbuf, size_t packed_cap, int word_mode) {
  sc->rb = (uint8_t *) malloc(maxbuf);
  sc->pk = (int64_t *) malloc(packed_cap * sizeof(int64_t));
  sc->sm = word_mode ? (int64_t *) malloc(maxbuf * sizeof(int64_t)) : NULL;
}

static inline void tk_scratch_free (tk_scratch_t *sc) {
  free(sc->rb); free(sc->pk); free(sc->sm);
}

static size_t tk_render_pack (
  tk_tokenizer_t *t, const char *text, size_t tlen, size_t s, size_t e,
  int64_t c0, int64_t c1, tk_ivec_t *cs, tk_ivec_t *cty,
  int word_mode, int flatten_mode, const uint8_t *is_marker, tk_scratch_t *sc)
{
  size_t w = tk_render_row(t, sc->rb, text, tlen, s, e, c0, c1, cs, cty);
  if (word_mode) {
    size_t ns = tk_parse_symbols(sc->rb, w, t->content, is_marker, sc->sm);
    return tk_pack_symbols(sc->sm, ns, t->ngram_min, t->ngram_max, sc->pk);
  }
  if (flatten_mode) w = tk_boundary_flatten(sc->rb, w, t->content, is_marker, t->b_sep);
  return tk_pack_row(sc->rb, w, t->ngram_min, t->ngram_max, sc->pk);
}

static int64_t tk_emit_row (
  tk_tokenizer_t *t, tk_iumap_t *map, uint32_t mend,
  const char *text, size_t tlen, size_t s, size_t e, int64_t c0, int64_t c1,
  tk_ivec_t *cs, tk_ivec_t *cty,
  int word_mode, int flatten_mode, const uint8_t *is_marker,
  tk_scratch_t *sc, int32_t *otok, float *oval)
{
  size_t count = tk_render_pack(t, text, tlen, s, e, c0, c1, cs, cty,
    word_mode, flatten_mode, is_marker, sc);
  int64_t *packed = sc->pk;
  int64_t nv = 0;
  for (size_t i = 0; i < count; i++) {
    uint32_t it = tk_iumap_get(map, packed[i]);
    if (it != mend) packed[nv++] = tk_iumap_val(map, it);
  }
  ks_introsort(tk_ivec_asc, (size_t) nv, packed);
  int64_t nnz = 0;
  for (int64_t i = 0; i < nv; ) {
    int64_t tk = packed[i]; float c = 0.0f;
    while (i < nv && packed[i] == tk) { c += 1.0f; i++; }
    if (otok) { otok[nnz] = (int32_t) tk; oval[nnz] = c; }
    nnz++;
  }
  return nnz;
}

static int tk_tokenize_core (lua_State *L, bool grow) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int has_boundary = 0;
  for (int i = 0; i < 256; i++) if (t->content[i]) { has_boundary = 1; break; }
  int word_mode = has_boundary && t->words;
  int flatten_mode = has_boundary && !t->words;
  uint8_t is_marker[256]; memset(is_marker, 0, 256);
  if (has_boundary) {
    if (t->b_bos) is_marker[t->b_bos] = 1;
    if (t->b_eos) is_marker[t->b_eos] = 1;
    if (t->b_focus_open) is_marker[t->b_focus_open] = 1;
    if (t->b_focus_close) is_marker[t->b_focus_close] = 1;
    if (t->b_sep) is_marker[t->b_sep] = 1;
    for (int k = 0; k < t->n_types + 2; k++) if (t->b_type[k]) is_marker[t->b_type[k]] = 1;
  }

  lua_getfield(L, 2, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int texts_idx = lua_gettop(L);
  int64_t n_samples = (int64_t) lua_objlen(L, texts_idx);
  tk_ivec_t *fo = NULL, *fs = NULL, *fe = NULL;
  lua_getfield(L, 2, "focus");
  if (!lua_isnil(L, -1)) {
    tk_spans_t *F = tk_spans_peek(L, -1, "focus");
    int64_t is = tk_spans_colidx(F, "s"), ie = tk_spans_colidx(F, "e");
    if (is < 0 || ie < 0) return luaL_error(L, "tokenizer: focus spans need columns \"s\" and \"e\"");
    fo = F->offsets; fs = F->cols[is]; fe = F->cols[ie];
  }
  lua_pop(L, 1);
  tk_ivec_t *co = NULL, *cs = NULL, *ce = NULL, *cty = NULL;
  lua_getfield(L, 2, "types");
  if (!lua_isnil(L, -1)) {
    tk_spans_t *C = tk_spans_peek(L, -1, "types");
    int64_t is = tk_spans_colidx(C, "s"), ie = tk_spans_colidx(C, "e"), it = tk_spans_colidx(C, "ty");
    if (is < 0 || ie < 0 || it < 0) return luaL_error(L, "tokenizer: types spans need columns \"s\", \"e\", \"ty\"");
    co = C->offsets; cs = C->cols[is]; ce = C->cols[ie]; cty = C->cols[it];
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
  size_t packed_cap = nrange * maxbuf;

  if (grow && !t->ngram_map) t->ngram_map = tk_iumap_create(NULL, 0);
  if (!t->ngram_map) { free(text_ptrs); free(text_lens);
    return luaL_error(L, "tokenizer: frozen tokenize before any grow=true pass"); }
  tk_iumap_t *map = t->ngram_map;

  int64_t *row_doc = (int64_t *) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int64_t));
  if (per_span) {
    for (int64_t d = 0; d < n_samples; d++)
      for (int64_t r = fo->a[d]; r < fo->a[d + 1]; r++) row_doc[r] = d;
  } else {
    for (int64_t r = 0; r < n_rows; r++) row_doc[r] = r;
  }

  int64_t *nnz = (int64_t *) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int64_t));

  if (grow) {
    #pragma omp parallel
    {
      tk_iumap_t *lmap = tk_iumap_create(NULL, 0);
      tk_scratch_t sc;
      tk_scratch_init(&sc, maxbuf, packed_cap, word_mode);
      #pragma omp for schedule(dynamic, 64)
      for (int64_t row = 0; row < n_rows; row++) {
        int64_t d = row_doc[row]; size_t tlen = text_lens[d];
        size_t s = per_span ? (size_t) fs->a[row] : 0;
        size_t e = per_span ? (size_t) fe->a[row] : tlen;
        size_t count = tk_render_pack(t, text_ptrs[d], tlen, s, e,
          co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, cty,
          word_mode, flatten_mode, is_marker, &sc);
        for (size_t i = 0; i < count; i++) { int absent; tk_iumap_put(lmap, sc.pk[i], &absent); }
        ks_introsort(tk_ivec_asc, (size_t) count, sc.pk);
        int64_t dc = 0;
        for (size_t i = 0; i < count; i++) if (i == 0 || sc.pk[i] != sc.pk[i - 1]) dc++;
        nnz[row] = dc;
      }
      #pragma omp critical (tk_grow_merge)
      {
        int64_t lk;
        tk_umap_foreach_keys(lmap, lk, ({ int absent; tk_iumap_put(map, lk, &absent); }));
      }
      tk_iumap_destroy(lmap);
      tk_scratch_free(&sc);
    }
    int64_t V = (int64_t) tk_iumap_size(map);
    int64_t *keys = (int64_t *) malloc((size_t) (V > 0 ? V : 1) * sizeof(int64_t));
    int64_t ki = 0, k;
    tk_umap_foreach_keys(map, k, ({ keys[ki++] = k; }));
    ks_introsort(tk_ivec_asc, (size_t) V, keys);
    for (int64_t i = 0; i < V; i++) { uint32_t it = tk_iumap_get(map, keys[i]); tk_iumap_setval(map, it, i); }
    free(keys);
  }
  uint32_t mend = tk_iumap_end(map);

  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t) (n_rows + 1));
  offsets->n = (uint64_t) (n_rows + 1); offsets->a[0] = 0;
  tk_svec_t *toks = NULL; tk_fvec_t *vals = NULL;

  {
    if (!grow) {
      #pragma omp parallel
      {
        tk_scratch_t sc;
        tk_scratch_init(&sc, maxbuf, packed_cap, word_mode);
        #pragma omp for schedule(dynamic, 64)
        for (int64_t row = 0; row < n_rows; row++) {
          int64_t d = row_doc[row]; size_t tlen = text_lens[d];
          size_t s = per_span ? (size_t) fs->a[row] : 0;
          size_t e = per_span ? (size_t) fe->a[row] : tlen;
          nnz[row] = tk_emit_row(t, map, mend, text_ptrs[d], tlen, s, e,
            co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, cty,
            word_mode, flatten_mode, is_marker, &sc, NULL, NULL);
        }
        tk_scratch_free(&sc);
      }
    }
    int64_t acc = 0;
    for (int64_t row = 0; row < n_rows; row++) { acc += nnz[row]; offsets->a[row + 1] = acc; }
    toks = tk_svec_create(L, (uint64_t) acc); toks->n = (uint64_t) acc;
    vals = tk_fvec_create(L, (uint64_t) acc); vals->n = (uint64_t) acc;
    #pragma omp parallel
    {
      tk_scratch_t sc;
      tk_scratch_init(&sc, maxbuf, packed_cap, word_mode);
      #pragma omp for schedule(dynamic, 64)
      for (int64_t row = 0; row < n_rows; row++) {
        int64_t d = row_doc[row]; size_t tlen = text_lens[d];
        size_t s = per_span ? (size_t) fs->a[row] : 0;
        size_t e = per_span ? (size_t) fe->a[row] : tlen;
        int64_t off = offsets->a[row];
        tk_emit_row(t, map, mend, text_ptrs[d], tlen, s, e,
          co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, cty,
          word_mode, flatten_mode, is_marker, &sc,
          toks->a + off, vals->a + off);
      }
      tk_scratch_free(&sc);
    }
    free(nnz);
  }

  free(row_doc); free(text_ptrs); free(text_lens);
  int iv = lua_gettop(L);
  int in_ = iv - 1;
  int io = iv - 2;
  uint64_t vocab = (uint64_t) tk_iumap_size(map);
  tk_csr_push(L, TK_TAG_F32, TK_TAG_I32, vocab, io, offsets, in_, toks, iv, vals);
  return 1;
}

static int tk_tokenizer_fit_lua (lua_State *L) { return tk_tokenize_core(L, true); }
static int tk_tokenizer_tokenize_lua (lua_State *L) { return tk_tokenize_core(L, false); }
static int tk_tokenizer_persist_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKtk", 1, 4, fh);
  uint8_t version = 9;
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
  if (version != 9) { tk_lua_fclose(L, fh);
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
  if (memcmp(&cfg, &chk, offsetof(tk_tokenizer_t, ngram_map)) != 0)
    return luaL_error(L, "tokenizer.load: byte-table drift (assigner changed since persist; refit required)");
  return 1;
}

static int tk_tokenizer_tokenize_raw_lua (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n_samples = (int64_t) tk_lua_fcheckunsigned(L, 1, "tokenize_raw", "n_samples");
  int ngram_min = (int) tk_lua_foptunsigned(L, 1, "tokenize_raw", "ngram_min", 1);
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
    ks_introsort(tk_ivec_asc, (size_t) count, packed);
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

static int tk_tokenizer_words_lua (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  int n = (int) tk_lua_fcheckunsigned(L, 1, "words", "n");
  int punct = tk_lua_foptboolean(L, 1, "words", "punctuation", 0);
  uint8_t cls[256];
  memset(cls, 0, 256);
  lua_getfield(L, 1, "word_characters");
  size_t wlen; const char *wc = lua_tolstring(L, -1, &wlen);
  if (!wc || wlen == 0)
    return luaL_error(L, "words: word_characters required");
  for (size_t i = 0; i < wlen; i++) cls[(uint8_t) wc[i]] = 1;
  lua_pop(L, 1);
  if (punct)
    for (int b = 0; b < 256; b++)
      if (!cls[b] && b != 0x20 && !(b >= 0x09 && b <= 0x0D)) cls[b] = 2;
  lua_getfield(L, 1, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int ti = lua_gettop(L);
  tk_ivec_t *off = tk_ivec_create(L, 0), *st = tk_ivec_create(L, 0), *en = tk_ivec_create(L, 0);
  tk_ivec_push(off, 0);
  for (int d = 0; d < n; d++) {
    lua_rawgeti(L, ti, d + 1);
    size_t len; const unsigned char *b = (const unsigned char *) lua_tolstring(L, -1, &len);
    lua_pop(L, 1);
    size_t i = 0;
    while (i < len) {
      uint8_t c = cls[b[i]];
      if (!c) { i++; continue; }
      size_t j = i + 1;
      while (j < len && cls[b[j]] == c) j++;
      tk_ivec_push(st, (int64_t) i);
      tk_ivec_push(en, (int64_t) j);
      i = j;
    }
    tk_ivec_push(off, (int64_t) st->n);
  }
  return 3;
}

static luaL_Reg tk_tokenizer_mt_fns[] = {
  { "fit", tk_tokenizer_fit_lua },
  { "tokenize", tk_tokenizer_tokenize_lua },
  { "n_tokens", tk_tokenizer_n_tokens_lua },
  { "persist", tk_tokenizer_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_tokenizer_fns[] = {
  { "create", tk_tokenizer_create_lua },
  { "load", tk_tokenizer_load_lua },
  { "tokenize_raw", tk_tokenizer_tokenize_raw_lua },
  { "words", tk_tokenizer_words_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_tokenizer (lua_State *L) {
  tk_lua_require_mod(L, "santoku.csr");
  lua_newtable(L);
  tk_lua_register(L, tk_tokenizer_fns, 0);
  return 1;
}

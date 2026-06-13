// santoku.learn.tokenizer -- fused render+ngram tokenizer (design v6.1).
// One rendering block == one tokenizer instance (own byte alphabet + ngram map).
// This file: config + byte assigner + object lifecycle + module wiring. The fused
// scan (tokenize) and persist/load are added in the following units.

#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <santoku/lua/utils.h>
#include <santoku/learn/normalize.h>
#include <santoku/ivec/ext.h>
#include <santoku/svec.h>
#include <santoku/fvec.h>
#include <santoku/cvec.h>
#include <santoku/iumap/ext.h>

#define TK_TOK_MT "tk_tokenizer_t"
#define TK_TOK_MAXTYPES 64        // entity types ceiling (per instance)
#define TK_TOK_NSHAPE   9         // {Xx, XX, xx, dd, punct, mix, X, Xp, an}

typedef enum { TK_STREAM_TEXT = 0, TK_STREAM_TYPE = 1, TK_STREAM_SHAPE = 2 } tk_stream_t;
typedef enum { TK_FOCUS_NONE = 0, TK_FOCUS_TRUE = 1 } tk_focus_t;

// Byte roles, assigned in this fixed order from the free-byte pool (terminals,
// focus, type marks, UNK, shapes). UNK is the type-mark slot at index n_types+1
// (slot n_types is the O role); they are standing roles of the type axis.
typedef struct {
  // config
  int ngram_min, ngram_max;
  int n_types;
  tk_stream_t stream;
  int normalize;
  int terminals;
  tk_focus_t focus;

  // byte assignments (0 = unassigned)
  uint8_t b_bos, b_eos;                          // terminals
  uint8_t b_focus_open, b_focus_close;           // focus = true
  uint8_t b_type[TK_TOK_MAXTYPES + 2];           // type byte per {types, O, UNK}
  uint8_t b_shape[TK_TOK_NSHAPE];                // shape classes
  int n_assigned;                                // count drawn from pool (for persist check)

  tk_iumap_t *ngram_map;                         // dense gram-id map (owned)
} tk_tokenizer_t;

static inline tk_tokenizer_t *tk_tokenizer_peek (lua_State *L, int i) {
  return (tk_tokenizer_t *) luaL_checkudata(L, i, TK_TOK_MT);
}

static luaL_Reg tk_tokenizer_mt_fns[];
static int tk_tokenizer_gc (lua_State *L);

// ---------------------------------------------------------------------------
// Byte assigner: a deterministic-from-config draw over the free-byte pool.
// Pool: 0x01-0x08, 0x0E-0x1F, 0x7F (27) always; + 'A'-'Z' and ws-controls
// 0x09-0x0D when normalize (they can never appear in normalized literal text).
// >= 0x80 is never used (UTF-8 passthrough). Roles drawn in fixed order.
// ---------------------------------------------------------------------------
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

// draw next free byte for `role` (name only used in the error message)
static uint8_t tk_assign (lua_State *L, tk_assigner_t *a, const char *role, int need_total) {
  if (a->i >= a->n)
    return (uint8_t) luaL_error(L,
      "tokenizer: out of marker bytes assigning %s (have %d, need %d; enable normalize to free 31 more)",
      role, a->n, need_total);
  return a->pool[a->i++];
}

// Run the assigner over the config's roles (fixed order). Returns total assigned.
static int tk_tokenizer_assign (lua_State *L, tk_tokenizer_t *t) {
  tk_assigner_t a;
  tk_assigner_init(&a, t->normalize);
  int nt = t->n_types;
  // upper bound on roles, for the error message
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
  t->n_assigned = a.i;
  return a.i;
}

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
static int tk_tokenizer_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  tk_tokenizer_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.ngram_min = (int) tk_lua_fcheckunsigned(L, 1, "tokenizer", "ngram_min");
  cfg.ngram_max = (int) tk_lua_fcheckunsigned(L, 1, "tokenizer", "ngram_max");
  if (cfg.ngram_min < 1 || cfg.ngram_min > cfg.ngram_max)
    return luaL_error(L, "tokenizer: need 1 <= ngram_min <= ngram_max");

  // types=true: replace token spans with the external tagger's type byte (skeleton).
  // shapes=true: per-token shape byte, boundaries+class derived internally (no input).
  // Mutually exclusive; neither => plain text stream.
  int want_types = tk_lua_foptboolean(L, 1, "tokenizer", "types", false);
  int want_shapes = tk_lua_foptboolean(L, 1, "tokenizer", "shapes", false);
  if (want_types && want_shapes)
    return luaL_error(L, "tokenizer: types and shapes are mutually exclusive");
  cfg.stream = want_types ? TK_STREAM_TYPE : (want_shapes ? TK_STREAM_SHAPE : TK_STREAM_TEXT);

  cfg.normalize = tk_lua_foptboolean(L, 1, "tokenizer", "normalize", false);
  cfg.terminals = tk_lua_foptboolean(L, 1, "tokenizer", "terminals", false);

  lua_getfield(L, 1, "focus");
  if (lua_isnil(L, -1) || (lua_isboolean(L, -1) && !lua_toboolean(L, -1))) cfg.focus = TK_FOCUS_NONE;
  else if (lua_isboolean(L, -1) && lua_toboolean(L, -1)) cfg.focus = TK_FOCUS_TRUE;
  else return luaL_error(L, "tokenizer: focus must be false|true");
  lua_pop(L, 1);

  // n_types: required by the type stream (sizes the external tagger's byte block)
  lua_getfield(L, 1, "n_types");
  if (!lua_isnil(L, -1)) cfg.n_types = (int) lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (cfg.stream == TK_STREAM_TYPE && cfg.n_types <= 0)
    return luaL_error(L, "tokenizer: n_types required when types=true");
  if (cfg.n_types > TK_TOK_MAXTYPES)
    return luaL_error(L, "tokenizer: n_types exceeds ceiling %d", TK_TOK_MAXTYPES);

  // VALIDITY hard errors
  if (cfg.normalize && cfg.stream != TK_STREAM_TEXT)
    return luaL_error(L, "tokenizer: normalize only valid on the text stream (no types/shapes)");

  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = NULL;
  tk_tokenizer_assign(L, t);   // raises on budget shortfall
  return 1;
}

// ---------------------------------------------------------------------------
// lifecycle: n_tokens, shrink, gc
// ---------------------------------------------------------------------------
static int tk_tokenizer_n_tokens_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  lua_pushinteger(L, t->ngram_map ? (lua_Integer) tk_iumap_size(t->ngram_map) : 0);
  return 1;
}

static int tk_tokenizer_shrink_lua (lua_State *L) {
  tk_tokenizer_peek(L, 1);  // scratch-only; nothing persistent to drop yet
  return 0;
}

static int tk_tokenizer_gc (lua_State *L) {
  tk_tokenizer_t *t = (tk_tokenizer_t *) luaL_checkudata(L, 1, TK_TOK_MT);
  if (t->ngram_map) { tk_iumap_destroy(t->ngram_map); t->ngram_map = NULL; }
  return 0;
}

// ---------------------------------------------------------------------------
// ngram packing (byte n-grams; bit-concat for n<=8, rolling hash else) -- a
// self-contained copy of the proven csr packer.
// ---------------------------------------------------------------------------
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

// Marker-collision guard (DECIDED: scrub, not assert): fold any byte in the
// assigner's non-normalize marker pool that appears in literal text to space, so
// literal bytes can never alias an injected marker. normalize=true makes this moot
// (control-fold + the wider folded alphabet); it only acts on raw (normalize=false)
// input. Silent by design -- we accept the rare in-text control byte becoming space
// rather than hard-failing the corpus.
static inline uint8_t tk_scrub (uint8_t b) {
  if ((b >= 0x01 && b <= 0x08) || (b >= 0x0E && b <= 0x1F) || b == 0x7F) return ' ';
  return b;
}

// shape class of one context span's ORIGINAL bytes (one single-byte symbol per token):
//   0=Xx Titlecase   1=XX all-caps(>=2)   2=xx lowercase   3=dd digits   4=punct
//   5=mix (fallback / non-ASCII / camelCase)   6=X single cap (initial)
//   7=Xp caps + internal punct, no lower/digit (U.N., AT&T, J.)
//   8=an alphanumeric: digit + letter (B2B, 3M, F1)
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
  if (up && !lo && !dig && !pun && !oth) return (n == 1) ? 6 : 1;   // X (single cap) / XX
  if (lo && !up && !dig && !pun && !oth) return 2;                  // xx
  if (dig && !up && !lo && !pun && !oth) return 3;                  // dd
  if (pun && !up && !lo && !dig && !oth) return 4;                  // punct
  if (dig && (up || lo) && !oth) return 8;                          // an (alphanumeric)
  if (up && pun && !lo && !dig && !oth) return 7;                   // Xp (dotted/amp caps)
  if (first_up && lo && !dig && !oth) return 0;                     // Xx
  return 5;                                                         // mix
}

// map a context_types value to the type-mark slot: types 0..n-1, O=n, UNK=n+1.
// context_types uses -1 for UNK (no type provided == below-confidence downstream).
static inline int tk_type_slot (int t, int n_types) {
  if (t < 0) return n_types + 1;        // UNK
  if (t >= n_types) return n_types;     // O (or out-of-range) -> O slot
  return t;
}

// Row render context: unifies literal-append and marker-insert across normalize
// modes. text stream may normalize (literal runs via the stream; prev_space carries,
// markers reset it); type/shape are symbol streams (norm = 0, all writes are marks).
typedef struct { uint8_t *buf; size_t w; int norm; tk_norm_stream_t ns; } tk_render_t;
static inline void tk_render_init (tk_render_t *r, uint8_t *buf, int norm) {
  r->buf = buf; r->w = 0; r->norm = norm;
  tk_norm_stream_init(&r->ns, buf);   // always init: ns is unused when !norm, but
}                                     // this keeps it fully defined (silences -Wmaybe-uninitialized)
static inline void tk_render_lit (tk_render_t *r, const char *text, size_t a, size_t b) {
  if (b <= a) return;
  if (r->norm) { tk_norm_stream_run(&r->ns, text + a, b - a); r->w = r->ns.nlen; }
  else { for (size_t i = a; i < b; i++) r->buf[r->w++] = tk_scrub((uint8_t) text[i]); }
}
static inline void tk_render_byte (tk_render_t *r, uint8_t byte) {
  if (r->norm) { tk_norm_stream_mark(&r->ns, byte); r->w = r->ns.nlen; }
  else r->buf[r->w++] = byte;
}
static inline void tk_render_mark (tk_render_t *r, uint8_t byte) { tk_render_byte(r, byte); }
static inline void tk_render_content (tk_render_t *r, uint8_t byte) { tk_render_byte(r, byte); }
static inline size_t tk_render_finish (tk_render_t *r) {
  if (r->norm) r->w = tk_norm_stream_finish(&r->ns);
  return r->w;
}

// Render one output row's byte stream into rowbuf. `s`,`e` are the focus span
// (ignored when per_span==0).
static size_t tk_render_row (
  tk_tokenizer_t *t, uint8_t *rowbuf,
  const char *text, size_t tlen, int per_span, size_t s, size_t e,
  int64_t c0, int64_t c1,
  tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty, int suppress_focus)
{
  (void) per_span;
  tk_focus_t efocus = suppress_focus ? TK_FOCUS_NONE : t->focus;
  tk_render_t r;
  tk_render_init(&r, rowbuf, (t->stream == TK_STREAM_TEXT && t->normalize));
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

  } else { // TK_STREAM_SHAPE: word tokens derived from whitespace (no external spans)
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

// Emit one FROZEN row: render + pack + read-only map lookup + sort + dedup. Writes the deduped
// (token,count) pairs to otok/oval when non-NULL (caller sizes them to the row's nnz); returns nnz either
// way. Pure per-row work over thread-private scratch -> safe to call concurrently across rows (the map is
// read-only here). Deterministic, so a count pass and a write pass agree exactly.
static int64_t tk_emit_row (
  tk_tokenizer_t *t, tk_iumap_t *map, uint32_t mend,
  const char *text, size_t tlen, int per_span, size_t s, size_t e, int64_t c0, int64_t c1,
  tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty,
  uint8_t *rowbuf, int64_t *packed, int32_t *otok, float *oval)
{
  size_t w = tk_render_row(t, rowbuf, text, tlen, per_span, s, e, c0, c1, cs, ce, cty, 0);
  size_t count = tk_pack_row(rowbuf, w, t->ngram_min, t->ngram_max, packed);
  int64_t nv = 0;
  for (size_t i = 0; i < count; i++) {
    uint32_t it = tk_iumap_get(map, packed[i]);
    if (it != mend) packed[nv++] = tk_iumap_val(map, it);
  }
  qsort(packed, (size_t) nv, sizeof(int64_t), tk_i64_cmp);
  int64_t nnz = 0;
  for (int64_t i = 0; i < nv; ) {
    int64_t tk = packed[i]; float c = 0.0f;
    while (i < nv && packed[i] == tk) { c += 1.0f; i++; }
    if (otok) { otok[nnz] = (int32_t) tk; oval[nnz] = c; }
    nnz++;
  }
  return nnz;
}

// ---------------------------------------------------------------------------
// tokenize (fused render + ngram). One output row per focus span (or per doc
// if no focus spans given). Returns offsets, tokens (svec), values (fvec).
//   grow:   SERIAL (dense ids are assigned in first-seen order -> kept on one thread for determinism).
//   frozen: FULLY PARALLEL -- read-only map makes rows independent; two passes (count then write to the
//           prefix-summed offsets) keep memory at 1x and the output bit-identical to the serial path.
// ---------------------------------------------------------------------------
static int tk_tokenizer_tokenize_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  bool grow = tk_lua_foptboolean(L, 2, "tokenize", "grow", false);
  int64_t n_samples = (int64_t) tk_lua_fcheckunsigned(L, 2, "tokenize", "n_samples");

  // texts table + optional focus/context span arrays (peeks only; no allocation).
  // INPUT IS TABLE-ONLY BY DESIGN: the old function-iterator path is retired;
  // streaming callers (e.g. eurlex) materialize a table caller-side. checktype
  // fails loud here so this is never an accident.
  lua_getfield(L, 2, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int texts_idx = lua_gettop(L);
  // focus = { offsets, starts, ends } -- which span to bracket (rows).
  // types  = { offsets, starts, ends, types } -- token spans + tagger byte (skeleton).
  // shapes need no input (boundaries + class derived from the text internally).
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

  // VALIDITY -- hoisted above the mallocs below so an error never leaks them.
  bool per_span = (fo != NULL);
  if (per_span && (!fs || !fe))
    return luaL_error(L, "tokenizer: focus.offsets given but focus.starts/ends missing");
  if (t->stream == TK_STREAM_TYPE && (!co || !cs || !ce || !cty))
    return luaL_error(L, "tokenizer: types=true requires types.offsets/starts/ends/types");
  if (t->focus != TK_FOCUS_NONE && !per_span)
    return luaL_error(L, "tokenizer: focus set at create but no focus spans passed");

  // text pointers per doc
  const char **text_ptrs = (const char **) malloc((size_t) n_samples * sizeof(char *));
  size_t *text_lens = (size_t *) malloc((size_t) n_samples * sizeof(size_t));
  for (int64_t d = 0; d < n_samples; d++) {
    lua_rawgeti(L, texts_idx, (int) (d + 1));
    text_ptrs[d] = lua_tolstring(L, -1, &text_lens[d]);
    lua_pop(L, 1);
  }

  int64_t n_rows = per_span ? fo->a[(int64_t)(fo->n - 1)] : n_samples;

  // worst-case row buffer: doc bytes (text/shape rows <= text+markers) OR the
  // type-skeleton's one-byte-per-token count, whichever is larger, + markers.
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
  size_t packed_cap = nrange * maxbuf;   // tk_pack_row emits <= one key per (ngram-size, anchor)

  if (grow && !t->ngram_map) t->ngram_map = tk_iumap_create(NULL, 0);
  if (!t->ngram_map) { free(text_ptrs); free(text_lens);
    return luaL_error(L, "tokenizer: frozen tokenize before any grow=true pass"); }
  tk_iumap_t *map = t->ngram_map;

  // flat row -> doc map (per-span: many rows per doc; else row == doc). Lets the passes run a flat
  // parallel-for over rows; the focus span for a per-span row is fs->a[row]/fe->a[row].
  int64_t *row_doc = (int64_t *) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int64_t));
  if (per_span) {
    for (int64_t d = 0; d < n_samples; d++)
      for (int64_t r = fo->a[d]; r < fo->a[d + 1]; r++) row_doc[r] = d;
  } else {
    for (int64_t r = 0; r < n_rows; r++) row_doc[r] = r;
  }

  int64_t *nnz = (int64_t *) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int64_t));

  // GROW: build the vocab in parallel with THREAD-LOCAL maps (no contention), then merge under one
  // critical section per thread. The per-row dedup'd key count is captured here (nnz), folding the
  // count pass into the build -- since grow inserts every key, no key can miss the merged map, so this
  // count exactly equals what the write pass would recompute. Ids are then assigned by SORTED KEY VALUE
  // -- a pure function of the gram bytes, so the result is deterministic and independent of thread count
  // / row order. (One-time consequence vs the old first-seen scheme: the dense column order changes, so
  // downstream numbers rebase once.)
  if (grow) {
    #pragma omp parallel
    {
      tk_iumap_t *lmap = tk_iumap_create(NULL, 0);
      uint8_t *rb = (uint8_t *) malloc(maxbuf);
      int64_t *pk = (int64_t *) malloc(packed_cap * sizeof(int64_t));
      #pragma omp for schedule(dynamic, 64)
      for (int64_t row = 0; row < n_rows; row++) {
        int64_t d = row_doc[row]; size_t tlen = text_lens[d];
        size_t s = per_span ? (size_t) fs->a[row] : 0;
        size_t e = per_span ? (size_t) fe->a[row] : tlen;
        size_t w = tk_render_row(t, rb, text_ptrs[d], tlen, per_span, s, e,
          co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, ce, cty, 0);
        size_t count = tk_pack_row(rb, w, t->ngram_min, t->ngram_max, pk);
        for (size_t i = 0; i < count; i++) { int absent; tk_iumap_put(lmap, pk[i], &absent); }
        qsort(pk, count, sizeof(int64_t), tk_i64_cmp);
        int64_t dc = 0;
        for (size_t i = 0; i < count; i++) if (i == 0 || pk[i] != pk[i - 1]) dc++;
        nnz[row] = dc;
      }
      #pragma omp critical (tk_grow_merge)
      {
        int64_t lk;
        tk_umap_foreach_keys(lmap, lk, ({ int absent; tk_iumap_put(map, lk, &absent); }));
      }
      tk_iumap_destroy(lmap);
      free(rb); free(pk);
    }
    int64_t V = (int64_t) tk_iumap_size(map);
    int64_t *keys = (int64_t *) malloc((size_t) (V > 0 ? V : 1) * sizeof(int64_t));
    int64_t ki = 0, k;
    tk_umap_foreach_keys(map, k, ({ keys[ki++] = k; }));
    qsort(keys, (size_t) V, sizeof(int64_t), tk_i64_cmp);
    for (int64_t i = 0; i < V; i++) { uint32_t it = tk_iumap_get(map, keys[i]); tk_iumap_setval(map, it, i); }
    free(keys);
  }
  uint32_t mend = tk_iumap_end(map);

  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t) (n_rows + 1));
  offsets->n = (uint64_t) (n_rows + 1); offsets->a[0] = 0;
  tk_svec_t *toks = NULL; tk_fvec_t *vals = NULL;

  // WRITE: the map is built/read-only now, so rows are independent. For grow, nnz is already filled (the
  // build folded the count in); for frozen, a count pass is needed since keys may miss the map. Then
  // prefix-sum -> offsets and write directly into the slices. tk_emit_row is deterministic, so the count
  // and write agree exactly.
  {
    if (!grow) {
      #pragma omp parallel
      {
        uint8_t *rb = (uint8_t *) malloc(maxbuf);
        int64_t *pk = (int64_t *) malloc(packed_cap * sizeof(int64_t));
        #pragma omp for schedule(dynamic, 64)
        for (int64_t row = 0; row < n_rows; row++) {
          int64_t d = row_doc[row]; size_t tlen = text_lens[d];
          size_t s = per_span ? (size_t) fs->a[row] : 0;
          size_t e = per_span ? (size_t) fe->a[row] : tlen;
          nnz[row] = tk_emit_row(t, map, mend, text_ptrs[d], tlen, per_span, s, e,
            co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, ce, cty, rb, pk, NULL, NULL);
        }
        free(rb); free(pk);
      }
    }
    int64_t acc = 0;
    for (int64_t row = 0; row < n_rows; row++) { acc += nnz[row]; offsets->a[row + 1] = acc; }
    toks = tk_svec_create(L, (uint64_t) acc); toks->n = (uint64_t) acc;
    vals = tk_fvec_create(L, (uint64_t) acc); vals->n = (uint64_t) acc;
    #pragma omp parallel
    {
      uint8_t *rb = (uint8_t *) malloc(maxbuf);
      int64_t *pk = (int64_t *) malloc(packed_cap * sizeof(int64_t));
      #pragma omp for schedule(dynamic, 64)
      for (int64_t row = 0; row < n_rows; row++) {
        int64_t d = row_doc[row]; size_t tlen = text_lens[d];
        size_t s = per_span ? (size_t) fs->a[row] : 0;
        size_t e = per_span ? (size_t) fe->a[row] : tlen;
        int64_t off = offsets->a[row];
        tk_emit_row(t, map, mend, text_ptrs[d], tlen, per_span, s, e,
          co ? co->a[d] : 0, co ? co->a[d + 1] : 0, cs, ce, cty, rb, pk,
          toks->a + off, vals->a + off);
      }
      free(rb); free(pk);
    }
    free(nnz);
  }

  free(row_doc); free(text_ptrs); free(text_lens);
  lua_pushvalue(L, lua_gettop(L) - 2);   // offsets
  lua_pushvalue(L, lua_gettop(L) - 2);   // toks
  lua_pushvalue(L, lua_gettop(L) - 2);   // vals
  return 3;
}
// Persist: "TKtk" + version + config-and-byte-table block + ngram map. The whole
// config region [0, offsetof(ngram_map)) is dumped as a RAW STRUCT -- this is
// ABI/padding/endianness-fragile and deviates from the field-wise house style;
// accepted deliberately for terseness since these models are not moved across
// builds. The version byte gates layout changes; load recomputes the
// config-derived byte table and asserts it matches (catches assigner drift).
static int tk_tokenizer_persist_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKtk", 1, 4, fh);
  uint8_t version = 5;   // v5 removed bpx from the config block
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
  if (version != 5) { tk_lua_fclose(L, fh);
    return luaL_error(L, "tokenizer.load: unsupported version %d (old layout; refit required)", (int) version); }
  tk_tokenizer_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  size_t cfgsz = offsetof(tk_tokenizer_t, ngram_map);
  tk_lua_fread(L, &cfg, 1, cfgsz, fh);
  uint8_t has_map;
  tk_lua_fread(L, &has_map, sizeof(uint8_t), 1, fh);
  // load UNMANAGED (NULL) to match grow's tk_iumap_create(NULL,0): the tokenizer
  // owns the map and frees it in gc. Loading with L would make a Lua-managed
  // userdata whose own gc also frees it -> double-free of the kh buckets.
  tk_iumap_t *map = has_map ? tk_iumap_load(NULL, fh) : NULL;
  tk_lua_fclose(L, fh);

  // Create the userdata and attach the (unmanaged) map FIRST, so gc owns it on
  // every exit path below -- including the drift assert's tk_tokenizer_assign,
  // which longjmps on a budget shortfall.
  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = map;

  // drift assert: recompute the config-derived byte table, compare to what was persisted. Catches an
  // assigner change not covered by the version byte. The whole config block is config-derived now.
  tk_tokenizer_t chk = cfg; chk.ngram_map = NULL;
  tk_tokenizer_assign(L, &chk);
  if (memcmp(&cfg, &chk, offsetof(tk_tokenizer_t, ngram_map)) != 0)
    return luaL_error(L, "tokenizer.load: byte-table drift (assigner changed since persist; refit required)");
  return 1;
}

// tokenize_raw: stateless raw-hash tokenization -- no instance, no ngram map, no
// grow/freeze. Normalizes (optional), packs char ngrams, returns the RAW int64
// packed keys (sorted+deduped per row) with TF counts. For callers that want
// content-derived hashes rather than mapped dense ids (e.g. encrypted full-text
// search). Output `tokens` is an ivec (int64) -- NOT compatible with the CSR
// pipeline (apply_bns/normalize/spectral.encode expect svec int32 dense ids).
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

  size_t maxlen = 0;   // for buffer sizing (normalize is contraction-only)
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
    lua_pop(L, 1);   // texts table keeps the string alive while we hold `text`

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
  return 3;   // offsets, tokens, values (top 3 on the stack)
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

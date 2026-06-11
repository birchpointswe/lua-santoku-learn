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
#define TK_TOK_BPXMAX   64        // merge-symbol ceiling (>= max free-byte pool)
#define TK_TOK_FORK_DEF 4         // default bpx fork budget (<= 2^F variants/window)
#define TK_TOK_FORK_MAX 6         // hard cap on fork budget

typedef enum { TK_STREAM_TEXT = 0, TK_STREAM_TYPE = 1, TK_STREAM_SHAPE = 2 } tk_stream_t;
typedef enum { TK_FOCUS_NONE = 0, TK_FOCUS_TRUE = 1, TK_FOCUS_TYPED = 2 } tk_focus_t;
typedef enum { TK_MARKS_NONE = 0, TK_MARKS_BRACKET = 1, TK_MARKS_REPLACE = 2 } tk_marks_t;

// Byte roles, assigned in this fixed order from the free-byte pool (terminals,
// focus, type marks, UNK, shapes, [bpx last]). UNK is the type-mark slot at index
// n_types+1 (slot n_types is the O role); they are standing roles of the type axis.
typedef struct {
  // config
  int ngram_min, ngram_max;
  int n_types;
  tk_stream_t stream;
  int normalize;
  int bpx;
  int terminals;
  tk_focus_t focus;
  tk_marks_t marks;

  // byte assignments (0 = unassigned)
  uint8_t b_bos, b_eos;                          // terminals
  uint8_t b_focus_open, b_focus_close;           // focus = true
  uint8_t b_tfocus_open[TK_TOK_MAXTYPES + 2];    // focus = typed: per {types, O, UNK}
  uint8_t b_tfocus_close[TK_TOK_MAXTYPES + 2];
  uint8_t b_type[TK_TOK_MAXTYPES + 2];           // type/replace byte per {types, O, UNK}
  uint8_t b_mopen[TK_TOK_MAXTYPES + 2];          // marks = bracket: open per {types, O, UNK}
  uint8_t b_mclose[TK_TOK_MAXTYPES + 2];
  uint8_t b_shape[TK_TOK_NSHAPE];                // shape classes
  int n_assigned;                                // count drawn from pool (for persist check)

  // bpx (Byte-Pair Expansion). All of the following are persisted in the config
  // block (offsetof(ngram_map)). bpx_pool is config-derived (the free bytes left
  // after roles); bpx_lhs/rhs/sym + bpx_n are DATA-derived (learned on first grow).
  int bpx_fork_budget;                           // F
  int bpx_pool_n;                                // free bytes available for merges
  uint8_t bpx_pool[TK_TOK_BPXMAX];               // the reserved free bytes (symbol pool)
  int bpx_n;                                      // merges learned (<= bpx_pool_n)
  uint8_t bpx_lhs[TK_TOK_BPXMAX];                // merge i: (lhs,rhs) -> sym
  uint8_t bpx_rhs[TK_TOK_BPXMAX];
  uint8_t bpx_sym[TK_TOK_BPXMAX];

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
           + (t->focus == TK_FOCUS_TYPED ? 2 * (nt + 2) : 0)
           + (t->stream == TK_STREAM_TYPE ? (nt + 2) : 0)
           + (t->stream == TK_STREAM_SHAPE ? TK_TOK_NSHAPE : 0)
           + (t->marks == TK_MARKS_REPLACE ? (nt + 2) : 0)
           + (t->marks == TK_MARKS_BRACKET ? 2 * (nt + 2) : 0);

  if (t->terminals) {
    t->b_bos = tk_assign(L, &a, "terminals", need);
    t->b_eos = tk_assign(L, &a, "terminals", need);
  }
  if (t->focus == TK_FOCUS_TRUE) {
    t->b_focus_open = tk_assign(L, &a, "focus", need);
    t->b_focus_close = tk_assign(L, &a, "focus", need);
  } else if (t->focus == TK_FOCUS_TYPED) {
    for (int k = 0; k < nt + 2; k++) {
      t->b_tfocus_open[k] = tk_assign(L, &a, "typed-focus", need);
      t->b_tfocus_close[k] = tk_assign(L, &a, "typed-focus", need);
    }
  }
  if (t->stream == TK_STREAM_TYPE || t->marks == TK_MARKS_REPLACE) {
    for (int k = 0; k < nt + 2; k++)
      t->b_type[k] = tk_assign(L, &a, "type", need);
  }
  if (t->stream == TK_STREAM_SHAPE) {
    for (int k = 0; k < TK_TOK_NSHAPE; k++)
      t->b_shape[k] = tk_assign(L, &a, "shape", need);
  }
  if (t->marks == TK_MARKS_BRACKET) {
    for (int k = 0; k < nt + 2; k++) {
      t->b_mopen[k] = tk_assign(L, &a, "mark", need);
      t->b_mclose[k] = tk_assign(L, &a, "mark", need);
    }
  }
  // bpx takes ALL remaining free bytes, last. Merges are allocated from this pool
  // (in order) during learning; an empty pool degrades bpx to the all-raw path.
  t->bpx_pool_n = 0;
  if (t->bpx) {
    for (int k = a.i; k < a.n; k++)
      t->bpx_pool[t->bpx_pool_n++] = a.pool[k];
    a.i = a.n;
  }
  t->n_assigned = a.i;
  return a.i;
}

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
static const char *tk_optstr (lua_State *L, int idx, const char *field, const char *dflt) {
  lua_getfield(L, idx, field);
  const char *s = lua_isnil(L, -1) ? dflt : lua_tostring(L, -1);
  lua_pop(L, 1);
  return s;
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

  const char *sstream = tk_optstr(L, 1, "stream", "text");
  if (!strcmp(sstream, "text")) cfg.stream = TK_STREAM_TEXT;
  else if (!strcmp(sstream, "type")) cfg.stream = TK_STREAM_TYPE;
  else if (!strcmp(sstream, "shape")) cfg.stream = TK_STREAM_SHAPE;
  else return luaL_error(L, "tokenizer: stream must be text|type|shape");

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
  else if (lua_isstring(L, -1) && !strcmp(lua_tostring(L, -1), "typed")) cfg.focus = TK_FOCUS_TYPED;
  else return luaL_error(L, "tokenizer: focus must be false|true|\"typed\"");
  lua_pop(L, 1);

  lua_getfield(L, 1, "marks");   // reject a present-but-non-string marks (e.g. marks=true)
  if (!lua_isnil(L, -1) && !lua_isstring(L, -1)) return luaL_error(L, "tokenizer: marks must be \"bracket\"|\"replace\"");
  lua_pop(L, 1);
  const char *smarks = tk_optstr(L, 1, "marks", NULL);
  if (!smarks) cfg.marks = TK_MARKS_NONE;
  else if (!strcmp(smarks, "bracket")) cfg.marks = TK_MARKS_BRACKET;
  else if (!strcmp(smarks, "replace")) cfg.marks = TK_MARKS_REPLACE;
  else return luaL_error(L, "tokenizer: marks must be \"bracket\"|\"replace\"");

  // n_types: required by typed focus, type stream, or marks
  int needs_types = (cfg.focus == TK_FOCUS_TYPED) || (cfg.stream == TK_STREAM_TYPE) || (cfg.marks != TK_MARKS_NONE);
  lua_getfield(L, 1, "n_types");
  if (!lua_isnil(L, -1)) cfg.n_types = (int) lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (needs_types && cfg.n_types <= 0)
    return luaL_error(L, "tokenizer: n_types required for typed focus / type stream / marks");
  if (cfg.n_types > TK_TOK_MAXTYPES)
    return luaL_error(L, "tokenizer: n_types exceeds ceiling %d", TK_TOK_MAXTYPES);

  // VALIDITY hard errors
  if (cfg.normalize && cfg.stream != TK_STREAM_TEXT)
    return luaL_error(L, "tokenizer: normalize only valid on stream=text");
  if (cfg.marks != TK_MARKS_NONE && cfg.stream != TK_STREAM_TEXT)
    return luaL_error(L, "tokenizer: marks only valid on stream=text");
  // typed focus brackets are only rendered in the plain-text branch; the
  // type/shape/marks branches would emit the untyped b_focus_open instead.
  if (cfg.focus == TK_FOCUS_TYPED && (cfg.stream != TK_STREAM_TEXT || cfg.marks != TK_MARKS_NONE))
    return luaL_error(L, "tokenizer: focus=\"typed\" requires stream=text and marks=nil");

  tk_tokenizer_t *t = tk_lua_newuserdata(L, tk_tokenizer_t, TK_TOK_MT,
    tk_tokenizer_mt_fns, tk_tokenizer_gc);
  *t = cfg;
  t->ngram_map = NULL;
  tk_tokenizer_assign(L, t);   // raises on budget shortfall
  // bpx=true with no free bytes left for merges would silently behave as all-raw;
  // a config that does nothing is a lie, so fail loud rather than no-op.
  if (t->bpx && t->bpx_pool_n == 0)
    return luaL_error(L, "tokenizer: bpx=true but no free bytes for merges (all consumed by roles)");
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

// map a focus_type value to the typed-focus / type-mark slot: types 0..n-1, O=n, UNK=n+1.
// focus_types uses -1 for UNK (no entity-typed tokens handled caller-side as O = n).
static inline int tk_type_slot (int t, int n_types) {
  if (t < 0) return n_types + 1;        // UNK
  if (t >= n_types) return n_types;     // O (or out-of-range) -> O slot
  return t;
}

// Row render context: unifies literal-append and marker-insert across normalize
// modes. text stream may normalize (literal runs via the stream; prev_space carries,
// markers reset it); type/shape are symbol streams (norm = 0, all writes are marks).
// `mask` (optional, sized like buf): 1 at inserted-marker positions, 0 at literal
// bytes. bpx uses it to confine merges to literal runs (markers are hard stops and
// never participate in or fork during emission). NULL when bpx is off.
typedef struct { uint8_t *buf; uint8_t *mask; size_t w; int norm; tk_norm_stream_t ns; } tk_render_t;
static inline void tk_render_init (tk_render_t *r, uint8_t *buf, uint8_t *mask, int norm) {
  r->buf = buf; r->mask = mask; r->w = 0; r->norm = norm;
  if (norm) tk_norm_stream_init(&r->ns, buf);
}
static inline void tk_render_lit (tk_render_t *r, const char *text, size_t a, size_t b) {
  if (b <= a) return;
  size_t w0 = r->w;
  if (r->norm) { tk_norm_stream_run(&r->ns, text + a, b - a); r->w = r->ns.nlen; }
  else { for (size_t i = a; i < b; i++) r->buf[r->w++] = tk_scrub((uint8_t) text[i]); }
  if (r->mask) for (size_t i = w0; i < r->w; i++) r->mask[i] = 0;
}
// is_marker=1: a STRUCTURAL insertion (bos/eos, focus/mark brackets) -- a bpx hard
// stop. is_marker=0: STREAM CONTENT (type/shape symbol bytes) -- mergeable like
// literal text. (Literal text goes through tk_render_lit, also content.)
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

// Render one output row's byte stream into rowbuf (+ maskbuf marker flags if
// non-NULL). `s`,`e` are the focus span (ignored when per_span==0). `focus_type`
// is the raw type of the focus span for focus="typed" (else -1). Returns length.
// `suppress_focus` (bpx doc-level learning): render the whole doc/context as one
// stream with NO focus brackets, so merges are learned over doc-level streams
// (v6.1: "learned GLOBALLY per block over doc-level streams; APPLIED PER ROW").
static size_t tk_render_row (
  tk_tokenizer_t *t, uint8_t *rowbuf, uint8_t *maskbuf,
  const char *text, size_t tlen, int per_span, size_t s, size_t e,
  int focus_type, int64_t c0, int64_t c1,
  tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty, int suppress_focus)
{
  (void) per_span;
  tk_focus_t efocus = suppress_focus ? TK_FOCUS_NONE : t->focus;
  tk_render_t r;
  tk_render_init(&r, rowbuf, maskbuf, (t->stream == TK_STREAM_TEXT && t->normalize));
  if (t->terminals) tk_render_mark(&r, t->b_bos);

  if (t->stream == TK_STREAM_TEXT && t->marks != TK_MARKS_NONE) {
    size_t pos = 0; bool fdone = false;
    for (int64_t cj = c0; cj <= c1; cj++) {
      int64_t cstart = (cj < c1) ? cs->a[cj] : (int64_t) tlen;
      int64_t cend = (cj < c1) ? ce->a[cj] : (int64_t) tlen;
      int slot = (cj < c1) ? tk_type_slot(cty ? (int) cty->a[cj] : t->n_types, t->n_types) : 0;
      if (!fdone && (cj == c1 || (size_t) cstart >= s)) {
        tk_render_lit(&r, text, pos, s);
        if (efocus != TK_FOCUS_NONE) tk_render_mark(&r, t->b_focus_open);
        tk_render_lit(&r, text, s, e);
        if (efocus != TK_FOCUS_NONE) tk_render_mark(&r, t->b_focus_close);
        pos = e; fdone = true;
      }
      if (cj == c1) break;
      if ((size_t) cstart < e && (size_t) cend > s) continue;
      tk_render_lit(&r, text, pos, (size_t) cstart);
      if (t->marks == TK_MARKS_BRACKET) {
        tk_render_mark(&r, t->b_mopen[slot]);
        tk_render_lit(&r, text, (size_t) cstart, (size_t) cend);
        tk_render_mark(&r, t->b_mclose[slot]);
      } else {
        tk_render_mark(&r, t->b_type[slot]);
      }
      pos = (size_t) cend;
    }
    tk_render_lit(&r, text, pos, tlen);

  } else if (t->stream == TK_STREAM_TEXT) {
    int ft_slot = (t->focus == TK_FOCUS_TYPED) ? tk_type_slot(focus_type, t->n_types) : 0;
    uint8_t fopen = (t->focus == TK_FOCUS_TYPED) ? t->b_tfocus_open[ft_slot] : t->b_focus_open;
    uint8_t fclose = (t->focus == TK_FOCUS_TYPED) ? t->b_tfocus_close[ft_slot] : t->b_focus_close;
    if (efocus == TK_FOCUS_NONE) {
      tk_render_lit(&r, text, 0, tlen);
    } else {
      tk_render_lit(&r, text, 0, s);
      tk_render_mark(&r, fopen);
      tk_render_lit(&r, text, s, e);
      tk_render_mark(&r, fclose);
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

  } else { // TK_STREAM_SHAPE
    bool fo_done = false, fc_done = false;
    for (int64_t cj = c0; cj < c1; cj++) {
      size_t cstart = (size_t) cs->a[cj];
      int sc = tk_shape_class(text, cs->a[cj], ce->a[cj]);
      if (efocus != TK_FOCUS_NONE && !fo_done && cstart >= s) { tk_render_mark(&r, t->b_focus_open); fo_done = true; }
      if (efocus != TK_FOCUS_NONE && fo_done && !fc_done && cstart >= e) { tk_render_mark(&r, t->b_focus_close); fc_done = true; }
      tk_render_content(&r, t->b_shape[sc]);
    }
    if (efocus != TK_FOCUS_NONE && fo_done && !fc_done) tk_render_mark(&r, t->b_focus_close);
  }

  if (t->terminals) tk_render_mark(&r, t->b_eos);
  return tk_render_finish(&r);
}

// ===========================================================================
// bpx (Byte-Pair Expansion): learn merges on first grow, then emit a lattice
// of ngrams where each compressed merge-symbol independently forks {keep its
// byte | its full raw expansion}. See the header comment block for the model.
// ===========================================================================

// One ngram key from n bytes -- byte-identical to tk_pack_ngrams' i-th output
// (bit-concat for n<=8, rolling hash otherwise), so the all-raw lattice path
// reproduces the non-bpx keys exactly (a bpx block subsumes a raw block).
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

// Apply learned merges (in learned order, left-to-right non-overlapping) to one
// literal run in `in`, writing the compressed unit bytes into `out` (sized >=
// len). Returns compressed length. Sequential replay reproduces the encoding
// that learning produced (each merge was applied corpus-wide before the next).
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

// Build sym2idx[256] (byte -> merge index or -1) and per-byte raw expansions
// truncated to `cap` bytes (expbuf is 256*cap; exp_len[b] in [1,cap]). Symbols
// are processed in learned order so each merge's children are already resolved.
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

// Enumerate keep/expand variants of units[ui..], building byte strings up to
// nmax; record each into vars (row-major, stride nmax) with its length in vlen.
// Bounded by varcap; backtracking over fork points.
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
  if (mi < 0) {                         // leaf (raw byte or marker): no fork
    E->cur[curlen] = b;
    tk_bpx_dfs(E, ui + 1, curlen + 1);
  } else {                              // fork point: keep symbol, or expand
    E->cur[curlen] = b;                 // keep
    tk_bpx_dfs(E, ui + 1, curlen + 1);
    int add = E->exp_len[b];            // expand
    // No phantom grams: cap == nmax, so a TRUNCATED expansion (true length > nmax)
    // has exp_len == cap == nmax >= nmax-curlen, hence add == nmax-curlen and the
    // window FILLS -> the next dfs call stops. The DFS only continues past an
    // expansion when it was NOT truncated (true length fit), which is correct.
    if (curlen + add > E->nmax) add = E->nmax - curlen;
    memcpy(E->cur + curlen, E->expbuf + (size_t) b * (size_t) E->cap, (size_t) add);
    tk_bpx_dfs(E, ui + 1, curlen + add);
  }
}

// Pack one row's bpx lattice ngram KEYS into `packed`; returns the count.
//
// Two components, so the row SUBSUMES a raw block exactly while adding reach:
//   (1) RAW: tk_pack_row over the rendered stream (== the all-expand lattice
//       path), giving every raw ngram at its true position/count.
//   (2) REACH: build the unit sequence (literal/content runs compressed), and at
//       each unit-boundary anchor enumerate keep/expand variants; emit only the
//       prefix ngrams that contain >= 1 kept symbol byte (pure-raw prefixes are
//       already covered by (1), so this is exactly the compressed-reach delta).
// Known scope limit: a reach gram that STARTS inside a symbol's expansion and
// also contains a later kept symbol is not enumerated (anchors are unit-aligned);
// raw subsumption is unaffected since (1) is position-exact.
static size_t tk_bpx_pack_row (
  tk_tokenizer_t *t, const int16_t *sym2idx, const uint8_t *expbuf, const int *exp_len, int cap,
  const uint8_t *rowbuf, const uint8_t *mask, size_t w,
  uint8_t *units_buf, uint8_t *cur_buf, uint8_t *vars_buf, int *vlen_buf, int64_t *aids, int64_t *packed)
{
  int nmin = t->ngram_min, nmax = t->ngram_max, F = t->bpx_fork_budget;
  int varcap = 1 << F; if (varcap > (1 << TK_TOK_FORK_MAX)) varcap = 1 << TK_TOK_FORK_MAX;

  // (1) raw component
  size_t count = tk_pack_row(rowbuf, w, nmin, nmax, packed);

  // build units: markers pass through; maximal literal/content runs are compressed
  int ulen = 0;
  for (size_t i = 0; i < w; ) {
    if (mask[i]) { units_buf[ulen++] = rowbuf[i]; i++; continue; }
    size_t j = i; while (j < w && !mask[j]) j++;
    ulen += (int) tk_bpx_compress(t, rowbuf + i, j - i, units_buf + ulen);
    i = j;
  }

  // (2) reach component
  for (int a = 0; a < ulen; a++) {
    int symc = 0, hi = a + nmax; if (hi > ulen) hi = ulen;
    for (int k = a; k < hi; k++) if (sym2idx[units_buf[k]] >= 0) symc++;
    if (symc == 0) continue;            // no compressed reach available here

    int nv;
    if (symc > F) {                     // overflow: degrade to the all-keep path
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
        int has = 0;                    // skip pure-raw prefixes (already in (1))
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

// Learn bpx merges (classic Gage BPE) over each DOC's literal/content runs --
// rendered ONCE per doc with focus suppressed (v6.1: doc-level streams), NOT once
// per candidate row, so the pair statistics aren't weighted by candidate density.
// Pairs never cross run boundaries (markers); runs are separated by a sentinel in
// the int32 corpus. Fills bpx_lhs/rhs/sym/bpx_n.
static void tk_bpx_learn (
  tk_tokenizer_t *t, int64_t n_samples,
  const char **text_ptrs, const size_t *text_lens,
  tk_ivec_t *co, tk_ivec_t *cs, tk_ivec_t *ce, tk_ivec_t *cty,
  uint8_t *rowbuf, uint8_t *maskbuf)
{
  // corpus = runs concatenated as raw bytes (1x memory; merge symbols are bytes).
  // run boundaries are OUT OF BAND in run_off (rn+1 offsets): pairs/merges never
  // cross a boundary. run r occupies corpus[run_off[r], run_off[r+1]).
  size_t ccap = 4096, clen = 0;
  uint8_t *corpus = (uint8_t *) malloc(ccap);
  size_t rcap = 256, rn = 0;
  size_t *run_off = (size_t *) malloc((rcap + 1) * sizeof(size_t));
  run_off[0] = 0;

  for (int64_t d = 0; d < n_samples; d++) {
    const char *text = text_ptrs[d]; size_t tlen = text_lens[d];
    int64_t c0 = co ? co->a[d] : 0, c1 = co ? co->a[d + 1] : 0;
    // s = e = tlen: focus never matches (suppressed anyway), and the marks branch
    // renders + splits its runs correctly instead of swallowing the whole doc.
    size_t w = tk_render_row(t, rowbuf, maskbuf, text, tlen, 0, tlen, tlen, -1, c0, c1, cs, ce, cty, 1);
    for (size_t i = 0; i < w; ) {       // each maximal literal/content run -> one run
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

  // iterative BPE: most-frequent adjacent pair -> next free symbol, replace,
  // repeat until pool exhausted or best count < 2. Ties: smallest (a,b).
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
    size_t wpos = 0;                     // in-place per-run compaction (a,b -> sym)
    for (size_t r = 0; r < rn; r++) {
      size_t rs = run_off[r], re = run_off[r + 1];   // read both before overwriting run_off[r]
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

// ---------------------------------------------------------------------------
// tokenize (fused render + ngram). One output row per focus span (or per doc
// if no focus spans given). Returns offsets, tokens (svec), values (fvec).
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
  lua_getfield(L, 2, "doc_span_offsets");
  tk_ivec_t *fo = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "span_starts"); tk_ivec_t *fs = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "span_ends");   tk_ivec_t *fe = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "focus_types"); tk_ivec_t *fty = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "context_offsets"); tk_ivec_t *co = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "context_starts"); tk_ivec_t *cs = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "context_ends");   tk_ivec_t *ce = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "context_types");  tk_ivec_t *cty = tk_ivec_peekopt(L, -1); lua_pop(L, 1);

  // VALIDITY -- hoisted above the mallocs below so an error never leaks them.
  bool per_span = (fo != NULL);
  if (per_span && (!fs || !fe))
    return luaL_error(L, "tokenizer: doc_span_offsets given but span_starts/ends missing");
  if (t->stream != TK_STREAM_TEXT && (!co || !cs || !ce))
    return luaL_error(L, "tokenizer: stream=type/shape requires context spans + starts/ends");
  if (t->stream == TK_STREAM_TYPE && !cty)
    return luaL_error(L, "tokenizer: stream=type requires context_types");
  if (t->marks != TK_MARKS_NONE && (!per_span || !co || !cs || !ce))
    return luaL_error(L, "tokenizer: marks requires focus spans + context spans");
  if (t->focus != TK_FOCUS_NONE && !per_span)
    return luaL_error(L, "tokenizer: focus set but no focus spans (doc_span_offsets/span_starts/ends)");
  if (t->focus == TK_FOCUS_TYPED && !fty)
    return luaL_error(L, "tokenizer: focus=\"typed\" requires focus_types");

  // text pointers per doc
  const char **text_ptrs = (const char **) malloc((size_t) n_samples * sizeof(char *));
  size_t *text_lens = (size_t *) malloc((size_t) n_samples * sizeof(size_t));
  for (int64_t d = 0; d < n_samples; d++) {
    lua_rawgeti(L, texts_idx, (int) (d + 1));
    text_ptrs[d] = lua_tolstring(L, -1, &text_lens[d]);
    lua_pop(L, 1);
  }

  int64_t n_rows = per_span ? fo->a[(int64_t)(fo->n - 1)] : n_samples;

  // worst-case row buffer: doc bytes (+2/ctx-span for marks) OR ctx-span count, + markers
  size_t maxbuf = 8;
  for (int64_t d = 0; d < n_samples; d++) {
    size_t need = text_lens[d] + 8;
    if (t->marks != TK_MARKS_NONE && co) need += 2 * (size_t) (co->a[d + 1] - co->a[d]);
    if (need > maxbuf) maxbuf = need;
  }
  if (co && t->stream != TK_STREAM_TEXT) {
    for (int64_t d = 0; d + 1 < (int64_t) co->n; d++) {
      size_t nc = (size_t) (co->a[d + 1] - co->a[d]) + 8;
      if (nc > maxbuf) maxbuf = nc;
    }
  }
  size_t nrange = (size_t) (t->ngram_max - t->ngram_min + 1);
  uint8_t *rowbuf = (uint8_t *) malloc(maxbuf);

  // bpx scratch (NULL when bpx off): marker mask, unit/variant buffers, expansion
  // tables. packed is widened to hold up to varcap lattice variants per anchor.
  int bpx_varcap = t->bpx ? (1 << t->bpx_fork_budget) : 1;
  if (bpx_varcap < 2) bpx_varcap = 2;
  // bpx packs raw (<= nrange*maxbuf) + reach (<= varcap per anchor) -> (varcap+1)
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

  // bpx: learn merges once (first grow), then build the apply/expansion tables.
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
    int64_t fb = per_span ? fo->a[d + 1] : 1;   // non-per-span: a single pseudo-row
    int64_t c0 = co ? co->a[d] : 0, c1 = co ? co->a[d + 1] : 0;

    for (int64_t fi = fa; fi < fb; fi++) {
      size_t s = per_span ? (size_t) fs->a[fi] : 0;
      size_t e = per_span ? (size_t) fe->a[fi] : tlen;
      int ftype = (fty && per_span) ? (int) fty->a[fi] : -1;
      size_t w = tk_render_row(t, rowbuf, maskbuf, text, tlen, per_span, s, e, ftype, c0, c1, cs, ce, cty, 0);

      // --- pack + map this row ---
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
// config-derived byte table and asserts it matches (catches assigner drift). The
// DATA-derived bpx merges live in the same block and are loaded, never recomputed.
static int tk_tokenizer_persist_lua (lua_State *L) {
  tk_tokenizer_t *t = tk_tokenizer_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKtk", 1, 4, fh);
  uint8_t version = 2;   // v2 added the bpx block to the config region; v1 layout differs
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
  if (version != 2) { tk_lua_fclose(L, fh);
    return luaL_error(L, "tokenizer.load: unsupported version %d (pre-bpx format; refit required)", (int) version); }
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

  // drift assert: recompute the config-derived byte table and bpx pool, compare
  // to what was persisted. Catches an assigner change not covered by the version
  // byte. The DATA-derived merges (bpx_n/lhs/rhs/sym) are excluded from the compare.
  tk_tokenizer_t chk = cfg; chk.ngram_map = NULL;
  tk_tokenizer_assign(L, &chk);
  if (memcmp(&cfg, &chk, offsetof(tk_tokenizer_t, bpx_n)) != 0)
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

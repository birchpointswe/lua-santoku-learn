#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include <sys/mman.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/svec.h>
#include <santoku/iuset.h>
#include <santoku/learn/normalize.h>

#define TK_AHO_MT "tk_aho_t"
#define TK_AHO_BUILDER_MT "tk_aho_builder_t"

typedef struct {
  int32_t *goto_base;
  int32_t *fail;
  int64_t *output_id;
  int64_t *output_pri;
  int64_t *output_len;
  int32_t *output_link;
  int64_t n_states;
  int64_t n_patterns;
  int64_t max_output_len;
  bool normalize;
  tk_svec_t *goto_svec;
  bool destroyed;
} tk_aho_t;

typedef struct {
  int32_t *goto_base;
  int32_t *fail;
  int64_t *output_id;
  int64_t *output_pri;
  int64_t *output_len;
  int32_t *output_link;
  int64_t n_states;
  int64_t n_patterns;
  int64_t max_output_len;
  bool normalize;
  bool consumed;
} tk_aho_builder_t;

static inline tk_aho_t *tk_aho_peek (lua_State *L, int i) {
  return (tk_aho_t *)luaL_checkudata(L, i, TK_AHO_MT);
}

static inline int tk_aho_gc (lua_State *L) {
  tk_aho_t *a = tk_aho_peek(L, 1);
  if (!a->destroyed) {
    if (!a->goto_svec)
      free(a->goto_base);
    free(a->fail);
    free(a->output_id);
    free(a->output_pri);
    free(a->output_len);
    free(a->output_link);
  }
  a->destroyed = true;
  return 0;
}

static inline tk_aho_builder_t *tk_aho_builder_peek (lua_State *L, int i) {
  return (tk_aho_builder_t *)luaL_checkudata(L, i, TK_AHO_BUILDER_MT);
}

static inline int tk_aho_builder_gc (lua_State *L) {
  tk_aho_builder_t *b = tk_aho_builder_peek(L, 1);
  if (!b->consumed) {
    free(b->goto_base);
    free(b->fail);
    free(b->output_id);
    free(b->output_pri);
    free(b->output_len);
    free(b->output_link);
  }
  b->consumed = true;
  return 0;
}

static luaL_Reg tk_aho_mt_fns[];
static luaL_Reg tk_aho_builder_mt_fns[];

static void tk_aho_html_escape (luaL_Buffer *B, const char *s, size_t len) {
  for (size_t i = 0; i < len; i++) {
    switch (s[i]) {
      case '&': luaL_addlstring(B, "&amp;", 5); break;
      case '<': luaL_addlstring(B, "&lt;", 4); break;
      case '>': luaL_addlstring(B, "&gt;", 4); break;
      case '"': luaL_addlstring(B, "&quot;", 6); break;
      case '\'': luaL_addlstring(B, "&#39;", 5); break;
      default: luaL_addchar(B, s[i]); break;
    }
  }
}

typedef struct {
  int64_t id;
  int64_t start;
  int64_t end;
  int64_t priority;
} tk_aho_match_t;

#define tk_aho_match_lt(a, b) \
  ((a).start != (b).start ? (a).start < (b).start : \
   ((a).end - (a).start) != ((b).end - (b).start) ? \
     ((a).end - (a).start) > ((b).end - (b).start) : \
   (a).priority < (b).priority)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
KSORT_INIT(tk_aho_match, tk_aho_match_t, tk_aho_match_lt)
#pragma GCC diagnostic pop

typedef struct {
  tk_aho_match_t *matches;
  int64_t *text_offsets;
  int64_t total;
} tk_aho_scan_result_t;

static tk_aho_scan_result_t tk_aho_scan (
  tk_aho_t *a, const char **texts, size_t *tlens,
  int n_texts, bool longest, tk_pvec_t *exc, tk_iuset_t *inc,
  const uint8_t *wbound
) {
  tk_aho_match_t **pt_matches = (tk_aho_match_t **)calloc((uint64_t)n_texts, sizeof(tk_aho_match_t *));
  int64_t *pt_counts = (int64_t *)calloc((uint64_t)n_texts, sizeof(int64_t));

  #pragma omp parallel
  {
    int64_t ring_size = a->max_output_len > 0 ? a->max_output_len : 1;
    int64_t *ring = a->normalize ? (int64_t *)malloc((uint64_t)ring_size * sizeof(int64_t)) : NULL;
    int64_t local_cap = 64;
    tk_aho_match_t *local_m = (tk_aho_match_t *)malloc((uint64_t)local_cap * sizeof(tk_aho_match_t));

    #pragma omp for schedule(dynamic)
    for (int ti = 0; ti < n_texts; ti++) {
      size_t tlen = tlens[ti];
      int64_t m_n = 0;
      int32_t state = 0;

      if (a->normalize) {
        int64_t npos = 0;
        size_t i = 0;
        while (i < tlen) {
          tk_norm_result_t nr = tk_text_normalize_next(texts[ti], i, tlen);
          for (int bi = 0; bi < nr.n_out; bi++) {
            ring[npos % ring_size] = (nr.n_in == 1) ? (int64_t)(i + (size_t)bi) : (int64_t)i;
            state = a->goto_base[(int64_t)state * 256 + nr.bytes[bi]];
            int32_t tmp = state;
            while (tmp > 0) {
              if (a->output_id[tmp] >= 0) {
                int64_t os = ring[(npos - a->output_len[tmp] + 1) % ring_size];
                int64_t oe = (nr.n_in == 1) ? (int64_t)(i + (size_t)bi + 1) : (int64_t)(i + (size_t)nr.n_in);
                while ((size_t)oe + 1 < tlen) {
                  uint8_t cb = (uint8_t)texts[ti][oe];
                  if (cb == 0xCC && ((uint8_t)texts[ti][oe + 1] & 0xC0) == 0x80)
                    oe += 2;
                  else if (cb == 0xCD && (uint8_t)texts[ti][oe + 1] >= 0x80 && (uint8_t)texts[ti][oe + 1] <= 0xAF)
                    oe += 2;
                  else
                    break;
                }
                int skip = 0;
                if (wbound) {
                  if (os > 0 && wbound[(uint8_t)texts[ti][os - 1]])
                    skip = 1;
                  if (!skip && (size_t)oe < tlen && wbound[(uint8_t)texts[ti][oe]])
                    skip = 1;
                }
                if (!skip) {
                  if (m_n >= local_cap) {
                    local_cap *= 2;
                    local_m = (tk_aho_match_t *)realloc(local_m, (uint64_t)local_cap * sizeof(tk_aho_match_t));
                  }
                  local_m[m_n].id = a->output_id[tmp];
                  local_m[m_n].start = os;
                  local_m[m_n].end = oe;
                  local_m[m_n].priority = a->output_pri[tmp];
                  m_n++;
                }
              }
              tmp = a->output_link[tmp];
            }
            npos++;
          }
          i += (size_t)nr.n_in;
        }
      } else {
        for (size_t i = 0; i < tlen; i++) {
          state = a->goto_base[(int64_t)state * 256 + (uint8_t)texts[ti][i]];
          int32_t tmp = state;
          while (tmp > 0) {
            if (a->output_id[tmp] >= 0) {
              int64_t os = (int64_t)i - a->output_len[tmp] + 1;
              int64_t oe = (int64_t)(i + 1);
              int skip = 0;
              if (wbound) {
                if (os > 0 && wbound[(uint8_t)texts[ti][os - 1]])
                  skip = 1;
                if (!skip && (size_t)oe < tlen && wbound[(uint8_t)texts[ti][oe]])
                  skip = 1;
              }
              if (!skip) {
                if (m_n >= local_cap) {
                  local_cap *= 2;
                  local_m = (tk_aho_match_t *)realloc(local_m, (uint64_t)local_cap * sizeof(tk_aho_match_t));
                }
                local_m[m_n].id = a->output_id[tmp];
                local_m[m_n].start = os;
                local_m[m_n].end = oe;
                local_m[m_n].priority = a->output_pri[tmp];
                m_n++;
              }
            }
            tmp = a->output_link[tmp];
          }
        }
      }

      if (inc && m_n > 0) {
        int64_t write = 0;
        for (int64_t j = 0; j < m_n; j++) {
          if (tk_iuset_contains(inc, local_m[j].id))
            local_m[write++] = local_m[j];
        }
        m_n = write;
      }

      if (exc && exc->n > 0 && m_n > 0) {
        int64_t write = 0;
        for (int64_t j = 0; j < m_n; j++) {
          bool excluded = false;
          for (uint64_t k = 0; k < exc->n; k++) {
            if (local_m[j].start < exc->a[k].p && local_m[j].end > exc->a[k].i) {
              excluded = true; break;
            }
          }
          if (!excluded) local_m[write++] = local_m[j];
        }
        m_n = write;
      }

      if (longest && m_n > 0) {
        ks_introsort(tk_aho_match, (size_t)m_n, local_m);
        int64_t write = 0;
        int64_t last_end = -1;
        for (int64_t j = 0; j < m_n; j++) {
          if (local_m[j].start >= last_end) {
            local_m[write++] = local_m[j];
            last_end = local_m[j].end;
          }
        }
        m_n = write;
      }

      if (m_n > 0) {
        pt_matches[ti] = (tk_aho_match_t *)malloc((uint64_t)m_n * sizeof(tk_aho_match_t));
        memcpy(pt_matches[ti], local_m, (uint64_t)m_n * sizeof(tk_aho_match_t));
      }
      pt_counts[ti] = m_n;
    }

    free(ring);
    free(local_m);
  }

  int64_t *text_offsets = (int64_t *)malloc((uint64_t)(n_texts + 1) * sizeof(int64_t));
  text_offsets[0] = 0;
  for (int ti = 0; ti < n_texts; ti++)
    text_offsets[ti + 1] = text_offsets[ti] + pt_counts[ti];
  int64_t total = text_offsets[n_texts];

  tk_aho_match_t *flat = NULL;
  if (total > 0) {
    flat = (tk_aho_match_t *)malloc((uint64_t)total * sizeof(tk_aho_match_t));
    int64_t pos = 0;
    for (int ti = 0; ti < n_texts; ti++) {
      int64_t cnt = pt_counts[ti];
      if (cnt > 0) {
        memcpy(flat + pos, pt_matches[ti], (uint64_t)cnt * sizeof(tk_aho_match_t));
        free(pt_matches[ti]);
      }
      pos += cnt;
    }
  } else {
    for (int ti = 0; ti < n_texts; ti++)
      free(pt_matches[ti]);
  }

  free(pt_matches);
  free(pt_counts);

  tk_aho_scan_result_t r;
  r.matches = flat;
  r.text_offsets = text_offsets;
  r.total = total;
  return r;
}

static void tk_aho_scan_free (tk_aho_scan_result_t *r) {
  free(r->matches);
  free(r->text_offsets);
}

static void tk_aho_build_trie (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  bool do_normalize = tk_lua_foptboolean(L, 1, "aho.prepare", "normalize", false);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peek(L, -1, "ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "priorities");
  tk_ivec_t *priorities = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "patterns");
  luaL_checktype(L, -1, LUA_TTABLE);
  int ptab = lua_gettop(L);

  int64_t n_patterns = (int64_t)ids->n;

  int64_t total_chars = 0;
  for (int64_t i = 0; i < n_patterns; i++) {
    lua_rawgeti(L, ptab, (int)(i + 1));
    size_t len;
    luaL_checklstring(L, -1, &len);
    total_chars += (int64_t)len;
    lua_pop(L, 1);
  }

  int64_t cap = total_chars + 1;
  if (cap < 2) cap = 2;
  int64_t n_states = 1;

  int32_t *goto_base = (int32_t *)malloc((uint64_t)cap * 256 * sizeof(int32_t));
  int32_t *fail = (int32_t *)calloc((uint64_t)cap, sizeof(int32_t));
  int64_t *output_id = (int64_t *)malloc((uint64_t)cap * sizeof(int64_t));
  int64_t *output_pri = (int64_t *)malloc((uint64_t)cap * sizeof(int64_t));
  int64_t *output_len = (int64_t *)calloc((uint64_t)cap, sizeof(int64_t));
  int32_t *output_link = (int32_t *)malloc((uint64_t)cap * sizeof(int32_t));

  memset(goto_base, 0, 256 * sizeof(int32_t));
  for (int64_t i = 0; i < cap; i++) {
    output_id[i] = -1;
    output_pri[i] = INT64_MAX;
    output_link[i] = -1;
  }

  for (int64_t p = 0; p < n_patterns; p++) {
    lua_rawgeti(L, ptab, (int)(p + 1));
    size_t len;
    const char *pat = luaL_checklstring(L, -1, &len);
    int32_t cur = 0;
    int64_t nlen = 0;
    size_t pi = 0;
    while (pi < len) {
      uint8_t buf[4];
      int n_out;
      if (do_normalize) {
        tk_norm_result_t nr = tk_text_normalize_next(pat, pi, len);
        memcpy(buf, nr.bytes, (size_t)nr.n_out);
        n_out = nr.n_out;
        pi += (size_t)nr.n_in;
      } else {
        buf[0] = (uint8_t)pat[pi];
        n_out = 1;
        pi++;
      }
      for (int bi = 0; bi < n_out; bi++) {
        uint8_t c = buf[bi];
        int32_t nxt = goto_base[(int64_t)cur * 256 + c];
        if (nxt == 0 && (cur != 0 || c != 0)) {
          nxt = (int32_t)n_states;
          goto_base[(int64_t)cur * 256 + c] = nxt;
          if (n_states >= cap) {
            cap *= 2;
            goto_base = (int32_t *)realloc(goto_base, (uint64_t)cap * 256 * sizeof(int32_t));
            fail = (int32_t *)realloc(fail, (uint64_t)cap * sizeof(int32_t));
            output_id = (int64_t *)realloc(output_id, (uint64_t)cap * sizeof(int64_t));
            output_pri = (int64_t *)realloc(output_pri, (uint64_t)cap * sizeof(int64_t));
            output_len = (int64_t *)realloc(output_len, (uint64_t)cap * sizeof(int64_t));
            output_link = (int32_t *)realloc(output_link, (uint64_t)cap * sizeof(int32_t));
          }
          memset(goto_base + (int64_t)n_states * 256, 0, 256 * sizeof(int32_t));
          fail[n_states] = 0;
          output_id[n_states] = -1;
          output_pri[n_states] = INT64_MAX;
          output_len[n_states] = 0;
          output_link[n_states] = -1;
          n_states++;
        }
        cur = nxt;
        nlen++;
      }
    }
    int64_t new_pri = priorities ? priorities->a[p] : 0;
    if (output_id[cur] < 0 || new_pri < output_pri[cur]) {
      output_id[cur] = ids->a[p];
      output_pri[cur] = new_pri;
      output_len[cur] = nlen;
    }
    lua_pop(L, 1);
  }

  int32_t *queue = (int32_t *)malloc((uint64_t)n_states * sizeof(int32_t));
  int64_t qh = 0, qt = 0;
  for (int c = 0; c < 256; c++) {
    int32_t s = goto_base[c];
    if (s != 0) {
      fail[s] = 0;
      queue[qt++] = s;
    }
  }
  while (qh < qt) {
    int32_t r = queue[qh++];
    for (int c = 0; c < 256; c++) {
      int32_t s = goto_base[(int64_t)r * 256 + c];
      if (s != 0) {
        queue[qt++] = s;
        int32_t state = fail[r];
        while (state != 0 && goto_base[(int64_t)state * 256 + c] == 0)
          state = fail[state];
        fail[s] = goto_base[(int64_t)state * 256 + c];
        if (fail[s] == s) fail[s] = 0;
        output_link[s] = (output_id[fail[s]] >= 0) ? fail[s] : output_link[fail[s]];
      } else {
        int32_t state = fail[r];
        while (state != 0 && goto_base[(int64_t)state * 256 + c] == 0)
          state = fail[state];
        goto_base[(int64_t)r * 256 + c] = goto_base[(int64_t)state * 256 + c];
      }
    }
  }
  free(queue);

  lua_pop(L, 1);

  int64_t max_ol = 0;
  for (int64_t s = 0; s < n_states; s++)
    if (output_len[s] > max_ol) max_ol = output_len[s];

  tk_aho_builder_t *b = tk_lua_newuserdata(L, tk_aho_builder_t,
    TK_AHO_BUILDER_MT, tk_aho_builder_mt_fns, tk_aho_builder_gc);
  int bi = lua_gettop(L);
  b->goto_base = goto_base;
  b->fail = fail;
  b->output_id = output_id;
  b->output_pri = output_pri;
  b->output_len = output_len;
  b->output_link = output_link;
  b->n_states = n_states;
  b->n_patterns = n_patterns;
  b->max_output_len = max_ol;
  b->normalize = do_normalize;
  b->consumed = false;

  lua_newtable(L);
  lua_getfield(L, 1, "names");
  if (lua_type(L, -1) == LUA_TTABLE) {
    int ntab = lua_gettop(L);
    lua_newtable(L);
    for (int64_t i = 0; i < n_patterns; i++) {
      lua_pushinteger(L, (lua_Integer)ids->a[i]);
      lua_rawgeti(L, ntab, (int)(i + 1));
      lua_settable(L, -3);
    }
    lua_setfield(L, bi + 1, "names");
  }
  lua_pop(L, 1);
  lua_setfenv(L, bi);
}

static int tk_aho_finalize (lua_State *L, int builder_idx, int svec_idx)
{
  tk_aho_builder_t *b = tk_aho_builder_peek(L, builder_idx);
  if (b->consumed)
    return luaL_error(L, "builder already consumed");

  tk_svec_t *sv = svec_idx ? tk_svec_peek(L, svec_idx, "goto_buf") : NULL;
  if (sv) {
    uint64_t need = (uint64_t)b->n_states * 256;
    if (sv->n < need)
      return luaL_error(L, "build: svec too small (%d < %d)", (int)sv->n, (int)need);
    memcpy(sv->a, b->goto_base, need * sizeof(int32_t));
    free(b->goto_base);
  }

  tk_aho_t *a = tk_lua_newuserdata(L, tk_aho_t,
    TK_AHO_MT, tk_aho_mt_fns, tk_aho_gc);
  int gi = lua_gettop(L);
  a->goto_base = sv ? sv->a : b->goto_base;
  a->fail = b->fail;
  a->output_id = b->output_id;
  a->output_pri = b->output_pri;
  a->output_len = b->output_len;
  a->output_link = b->output_link;
  a->n_states = b->n_states;
  a->n_patterns = b->n_patterns;
  a->max_output_len = b->max_output_len;
  a->normalize = b->normalize;
  a->goto_svec = sv;
  a->destroyed = false;

  lua_newtable(L);
  if (svec_idx) {
    lua_pushvalue(L, svec_idx);
    lua_setfield(L, -2, "goto_buf");
  }
  lua_getfenv(L, builder_idx);
  lua_getfield(L, -1, "names");
  if (lua_type(L, -1) == LUA_TTABLE) {
    lua_setfield(L, gi + 1, "names");
  } else {
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  lua_setfenv(L, gi);

  b->goto_base = NULL;
  b->fail = NULL;
  b->output_id = NULL;
  b->output_pri = NULL;
  b->output_len = NULL;
  b->output_link = NULL;
  b->consumed = true;

  return 1;
}

static int tk_aho_builder_n_states_lua (lua_State *L)
{
  tk_aho_builder_t *b = tk_aho_builder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)b->n_states);
  return 1;
}

static int tk_aho_builder_build_lua (lua_State *L)
{
  int svec_idx = (lua_gettop(L) >= 2 && !lua_isnil(L, 2)) ? 2 : 0;
  return tk_aho_finalize(L, 1, svec_idx);
}

static int tk_aho_prepare_lua (lua_State *L)
{
  lua_settop(L, 1);
  tk_aho_build_trie(L);
  return 1;
}

static int tk_aho_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  tk_aho_build_trie(L);
  return tk_aho_finalize(L, 2, 0);
}

static int tk_aho_predict_lua (lua_State *L)
{
  tk_aho_t *a = tk_aho_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  lua_getfield(L, 2, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int ttab = lua_gettop(L);
  int n_texts = (int)lua_objlen(L, ttab);

  bool longest = tk_lua_foptboolean(L, 2, "aho.predict", "longest", false);

  tk_pvec_t *exc = NULL;
  lua_getfield(L, 2, "exclude");
  if (lua_type(L, -1) == LUA_TUSERDATA)
    exc = tk_pvec_peek(L, -1, "exclude");
  lua_pop(L, 1);

  tk_iuset_t *inc = NULL;
  int inc_allocated = 0;
  lua_getfield(L, 2, "include");
  if (lua_type(L, -1) == LUA_TUSERDATA) {
    inc = tk_iuset_peekopt(L, -1);
    if (!inc) {
      tk_ivec_t *iv = tk_ivec_peekopt(L, -1);
      if (iv) {
        inc = tk_iuset_create(NULL, (uint32_t)iv->n);
        inc_allocated = 1;
        for (uint64_t i = 0; i < iv->n; i++) {
          int absent;
          tk_iuset_put(inc, iv->a[i], &absent);
        }
      }
    }
  }
  lua_pop(L, 1);

  uint8_t *wbound = NULL;
  lua_getfield(L, 2, "boundary");
  if (lua_type(L, -1) == LUA_TSTRING) {
    size_t blen;
    const char *bs = lua_tolstring(L, -1, &blen);
    wbound = (uint8_t *)calloc(256, 1);
    for (size_t i = 0; i < blen; i++)
      wbound[(uint8_t)bs[i]] = 1;
  }
  lua_pop(L, 1);

  const char **texts = (const char **)malloc((uint64_t)n_texts * sizeof(const char *));
  size_t *tlens = (size_t *)malloc((uint64_t)n_texts * sizeof(size_t));
  for (int ti = 0; ti < n_texts; ti++) {
    lua_rawgeti(L, ttab, ti + 1);
    texts[ti] = luaL_checklstring(L, -1, &tlens[ti]);
    lua_pop(L, 1);
  }

  tk_aho_scan_result_t res = tk_aho_scan(a, texts, tlens, n_texts, longest, exc, inc, wbound);
  free(texts);
  free(tlens);
  free(wbound);
  if (inc_allocated) { tk_iuset_destroy(inc); }

  uint64_t mn = res.total > 0 ? (uint64_t)res.total : 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_texts + 1));
  memcpy(offsets->a, res.text_offsets, (uint64_t)(n_texts + 1) * sizeof(int64_t));

  tk_ivec_t *out_ids = tk_ivec_create(L, mn);
  tk_ivec_t *out_starts = tk_ivec_create(L, mn);
  tk_ivec_t *out_ends = tk_ivec_create(L, mn);
  if (res.total == 0) {
    out_ids->n = 0;
    out_starts->n = 0;
    out_ends->n = 0;
  }

  for (int64_t i = 0; i < res.total; i++) {
    out_ids->a[i] = res.matches[i].id;
    out_starts->a[i] = res.matches[i].start;
    out_ends->a[i] = res.matches[i].end;
  }

  tk_aho_scan_free(&res);
  return 4;
}

static int tk_aho_tag_lua (lua_State *L)
{
  tk_aho_t *a = tk_aho_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  lua_getfield(L, 2, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int ttab = lua_gettop(L);
  int n_texts = (int)lua_objlen(L, ttab);

  lua_getfield(L, 2, "fmt");
  size_t fmt_len;
  const char *fmt = luaL_checklstring(L, -1, &fmt_len);
  lua_pop(L, 1);

  bool longest = tk_lua_foptboolean(L, 2, "aho.tag", "longest", false);

  tk_pvec_t *exc = NULL;
  lua_getfield(L, 2, "exclude");
  if (lua_type(L, -1) == LUA_TUSERDATA)
    exc = tk_pvec_peek(L, -1, "exclude");
  lua_pop(L, 1);

  tk_iuset_t *inc = NULL;
  int inc_allocated = 0;
  lua_getfield(L, 2, "include");
  if (lua_type(L, -1) == LUA_TUSERDATA) {
    inc = tk_iuset_peekopt(L, -1);
    if (!inc) {
      tk_ivec_t *iv = tk_ivec_peekopt(L, -1);
      if (iv) {
        inc = tk_iuset_create(NULL, (uint32_t)iv->n);
        inc_allocated = 1;
        for (uint64_t i = 0; i < iv->n; i++) {
          int absent;
          tk_iuset_put(inc, iv->a[i], &absent);
        }
      }
    }
  }
  lua_pop(L, 1);

  uint8_t *wbound = NULL;
  lua_getfield(L, 2, "boundary");
  if (lua_type(L, -1) == LUA_TSTRING) {
    size_t blen;
    const char *bs = lua_tolstring(L, -1, &blen);
    wbound = (uint8_t *)calloc(256, 1);
    for (size_t i = 0; i < blen; i++)
      wbound[(uint8_t)bs[i]] = 1;
  }
  lua_pop(L, 1);

  lua_getfenv(L, 1);
  lua_getfield(L, -1, "names");
  int names_idx = lua_gettop(L);
  bool has_names = lua_type(L, names_idx) == LUA_TTABLE;

  const char **texts = (const char **)malloc((uint64_t)n_texts * sizeof(const char *));
  size_t *tlens = (size_t *)malloc((uint64_t)n_texts * sizeof(size_t));
  for (int ti = 0; ti < n_texts; ti++) {
    lua_rawgeti(L, ttab, ti + 1);
    texts[ti] = luaL_checklstring(L, -1, &tlens[ti]);
    lua_pop(L, 1);
  }

  tk_aho_scan_result_t res = tk_aho_scan(a, texts, tlens, n_texts, longest, exc, inc, wbound);

  lua_newtable(L);
  int result_idx = lua_gettop(L);

  for (int ti = 0; ti < n_texts; ti++) {
    const char *text = texts[ti];
    size_t tlen = tlens[ti];
    int64_t off_s = res.text_offsets[ti];
    int64_t off_e = res.text_offsets[ti + 1];
    int64_t nm = off_e - off_s;

    const char **mnames = (const char **)malloc((uint64_t)(nm > 0 ? nm : 1) * sizeof(const char *));
    size_t *mname_lens = (size_t *)calloc((uint64_t)(nm > 0 ? nm : 1), sizeof(size_t));
    for (int64_t mi = 0; mi < nm; mi++) {
      if (has_names) {
        lua_pushinteger(L, (lua_Integer)res.matches[off_s + mi].id);
        lua_gettable(L, names_idx);
        if (lua_type(L, -1) == LUA_TSTRING)
          mnames[mi] = lua_tolstring(L, -1, &mname_lens[mi]);
        else
          mnames[mi] = "";
        lua_pop(L, 1);
      } else {
        mnames[mi] = "";
      }
    }

    luaL_Buffer B;
    luaL_buffinit(L, &B);
    int64_t prev_end = 0;

    for (int64_t mi = 0; mi < nm; mi++) {
      tk_aho_match_t *m = &res.matches[off_s + mi];
      if (m->start > prev_end)
        luaL_addlstring(&B, text + prev_end, (size_t)(m->start - prev_end));
      const char *match_str = text + m->start;
      size_t match_len = (size_t)(m->end - m->start);
      for (size_t fi = 0; fi < fmt_len; ) {
        if (fmt[fi] == '%') {
          if (fi + 2 < fmt_len && fmt[fi + 1] == '%') {
            luaL_addchar(&B, '%');
            fi += 2;
          } else if (fi + 7 <= fmt_len && memcmp(fmt + fi, "%hmatch", 7) == 0) {
            tk_aho_html_escape(&B, match_str, match_len);
            fi += 7;
          } else if (fi + 6 <= fmt_len && memcmp(fmt + fi, "%hname", 6) == 0) {
            tk_aho_html_escape(&B, mnames[mi], mname_lens[mi]);
            fi += 6;
          } else if (fi + 4 <= fmt_len && memcmp(fmt + fi, "%hid", 4) == 0) {
            char id_buf[32];
            int id_len = snprintf(id_buf, sizeof(id_buf), "%lld", (long long)m->id);
            tk_aho_html_escape(&B, id_buf, (size_t)id_len);
            fi += 4;
          } else if (fi + 6 <= fmt_len && memcmp(fmt + fi, "%match", 6) == 0) {
            luaL_addlstring(&B, match_str, match_len);
            fi += 6;
          } else if (fi + 5 <= fmt_len && memcmp(fmt + fi, "%name", 5) == 0) {
            luaL_addlstring(&B, mnames[mi], mname_lens[mi]);
            fi += 5;
          } else if (fi + 3 <= fmt_len && memcmp(fmt + fi, "%id", 3) == 0) {
            char id_buf[32];
            int id_len = snprintf(id_buf, sizeof(id_buf), "%lld", (long long)m->id);
            luaL_addlstring(&B, id_buf, (size_t)id_len);
            fi += 3;
          } else {
            luaL_addchar(&B, fmt[fi]);
            fi++;
          }
        } else {
          luaL_addchar(&B, fmt[fi]);
          fi++;
        }
      }
      prev_end = m->end;
    }

    if ((size_t)prev_end < tlen)
      luaL_addlstring(&B, text + prev_end, tlen - (size_t)prev_end);

    luaL_pushresult(&B);
    lua_rawseti(L, result_idx, ti + 1);
    free(mnames);
    free(mname_lens);
  }

  free(texts);
  free(tlens);
  free(wbound);
  if (inc_allocated) { tk_iuset_destroy(inc); }
  tk_aho_scan_free(&res);

  lua_pushvalue(L, result_idx);
  return 1;
}

static int tk_aho_persist_lua (lua_State *L) {
  tk_aho_t *a = tk_aho_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKac", 1, 4, fh);
  uint8_t version = 4;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &a->n_states, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &a->n_patterns, sizeof(int64_t), 1, fh);
  uint8_t norm_byte = a->normalize ? 1 : 0;
  tk_lua_fwrite(L, &norm_byte, sizeof(uint8_t), 1, fh);
  uint8_t goto_ext_byte = (a->goto_svec && a->goto_svec->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &goto_ext_byte, sizeof(uint8_t), 1, fh);
  if (goto_ext_byte) {
    msync(a->goto_base, (size_t)(a->n_states * 256) * sizeof(int32_t), MS_SYNC);
  } else {
    tk_lua_fwrite(L, a->goto_base, sizeof(int32_t), (size_t)(a->n_states * 256), fh);
  }
  tk_lua_fwrite(L, a->fail, sizeof(int32_t), (size_t)a->n_states, fh);
  tk_lua_fwrite(L, a->output_id, sizeof(int64_t), (size_t)a->n_states, fh);
  tk_lua_fwrite(L, a->output_pri, sizeof(int64_t), (size_t)a->n_states, fh);
  tk_lua_fwrite(L, a->output_len, sizeof(int64_t), (size_t)a->n_states, fh);
  tk_lua_fwrite(L, a->output_link, sizeof(int32_t), (size_t)a->n_states, fh);

  lua_getfenv(L, 1);
  lua_getfield(L, -1, "names");
  if (lua_type(L, -1) == LUA_TTABLE) {
    int ntab = lua_gettop(L);
    int64_t n_names = 0;
    lua_pushnil(L);
    while (lua_next(L, ntab) != 0) { n_names++; lua_pop(L, 1); }
    tk_lua_fwrite(L, &n_names, sizeof(int64_t), 1, fh);
    lua_pushnil(L);
    while (lua_next(L, ntab) != 0) {
      int64_t id = (int64_t)lua_tointeger(L, -2);
      size_t nlen;
      const char *name = lua_tolstring(L, -1, &nlen);
      int64_t nlen64 = (int64_t)nlen;
      tk_lua_fwrite(L, &id, sizeof(int64_t), 1, fh);
      tk_lua_fwrite(L, &nlen64, sizeof(int64_t), 1, fh);
      tk_lua_fwrite(L, (void *)name, 1, nlen, fh);
      lua_pop(L, 1);
    }
  } else {
    int64_t zero = 0;
    tk_lua_fwrite(L, &zero, sizeof(int64_t), 1, fh);
  }
  lua_pop(L, 2);

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

static int tk_aho_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  tk_svec_t *goto_buf = (lua_gettop(L) >= 3 && !lua_isnil(L, 3))
    ? tk_svec_peek(L, 3, "goto_buf") : NULL;
  int goto_buf_idx = goto_buf ? 3 : 0;
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKac", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid aho file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 3 && version != 4) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported aho version %d", (int)version);
  }
  int64_t n_states, n_patterns;
  tk_lua_fread(L, &n_states, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_patterns, sizeof(int64_t), 1, fh);
  uint8_t norm_byte;
  tk_lua_fread(L, &norm_byte, sizeof(uint8_t), 1, fh);
  bool do_normalize = norm_byte != 0;

  bool goto_external = false;
  if (version >= 4) {
    uint8_t goto_ext_byte;
    tk_lua_fread(L, &goto_ext_byte, sizeof(uint8_t), 1, fh);
    goto_external = goto_ext_byte != 0;
  }

  int32_t *goto_base;
  if (goto_external) {
    if (!goto_buf) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "aho load: external goto_base requires svec arg 3");
    }
    uint64_t need = (uint64_t)n_states * 256;
    if (goto_buf->n < need) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "aho load: svec too small (%d < %d)", (int)goto_buf->n, (int)need);
    }
    goto_base = goto_buf->a;
  } else if (goto_buf) {
    uint64_t need = (uint64_t)n_states * 256;
    if (goto_buf->n < need) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "aho load: svec too small (%d < %d)", (int)goto_buf->n, (int)need);
    }
    tk_lua_fread(L, goto_buf->a, sizeof(int32_t), (size_t)(n_states * 256), fh);
    goto_base = goto_buf->a;
  } else {
    goto_base = (int32_t *)malloc((uint64_t)n_states * 256 * sizeof(int32_t));
    tk_lua_fread(L, goto_base, sizeof(int32_t), (size_t)(n_states * 256), fh);
  }

  int32_t *fail = (int32_t *)malloc((uint64_t)n_states * sizeof(int32_t));
  int64_t *output_id = (int64_t *)malloc((uint64_t)n_states * sizeof(int64_t));
  int64_t *output_pri = (int64_t *)malloc((uint64_t)n_states * sizeof(int64_t));
  int64_t *output_len = (int64_t *)malloc((uint64_t)n_states * sizeof(int64_t));
  int32_t *output_link = (int32_t *)malloc((uint64_t)n_states * sizeof(int32_t));

  tk_lua_fread(L, fail, sizeof(int32_t), (size_t)n_states, fh);
  tk_lua_fread(L, output_id, sizeof(int64_t), (size_t)n_states, fh);
  tk_lua_fread(L, output_pri, sizeof(int64_t), (size_t)n_states, fh);
  tk_lua_fread(L, output_len, sizeof(int64_t), (size_t)n_states, fh);
  tk_lua_fread(L, output_link, sizeof(int32_t), (size_t)n_states, fh);

  int64_t n_names;
  tk_lua_fread(L, &n_names, sizeof(int64_t), 1, fh);
  int64_t *name_ids = NULL;
  char **name_strs = NULL;
  int64_t *name_lens = NULL;
  if (n_names > 0) {
    name_ids = (int64_t *)malloc((uint64_t)n_names * sizeof(int64_t));
    name_strs = (char **)malloc((uint64_t)n_names * sizeof(char *));
    name_lens = (int64_t *)malloc((uint64_t)n_names * sizeof(int64_t));
    for (int64_t i = 0; i < n_names; i++) {
      tk_lua_fread(L, &name_ids[i], sizeof(int64_t), 1, fh);
      tk_lua_fread(L, &name_lens[i], sizeof(int64_t), 1, fh);
      name_strs[i] = (char *)malloc((uint64_t)name_lens[i]);
      tk_lua_fread(L, name_strs[i], 1, (size_t)name_lens[i], fh);
    }
  }
  tk_lua_fclose(L, fh);

  tk_aho_t *a = tk_lua_newuserdata(L, tk_aho_t,
    TK_AHO_MT, tk_aho_mt_fns, tk_aho_gc);
  int gi = lua_gettop(L);
  a->goto_base = goto_base;
  a->fail = fail;
  a->output_id = output_id;
  a->output_pri = output_pri;
  a->output_len = output_len;
  a->output_link = output_link;
  a->n_states = n_states;
  a->n_patterns = n_patterns;
  int64_t max_ol = 0;
  for (int64_t s = 0; s < n_states; s++)
    if (output_len[s] > max_ol) max_ol = output_len[s];
  a->max_output_len = max_ol;
  a->normalize = do_normalize;
  a->goto_svec = goto_buf;
  a->destroyed = false;

  lua_newtable(L);
  if (goto_buf && goto_buf_idx) {
    lua_pushvalue(L, goto_buf_idx);
    lua_setfield(L, -2, "goto_buf");
  }
  if (n_names > 0) {
    lua_newtable(L);
    for (int64_t i = 0; i < n_names; i++) {
      lua_pushinteger(L, (lua_Integer)name_ids[i]);
      lua_pushlstring(L, name_strs[i], (size_t)name_lens[i]);
      lua_settable(L, -3);
      free(name_strs[i]);
    }
    lua_setfield(L, -2, "names");
    free(name_ids);
    free(name_strs);
    free(name_lens);
  }
  lua_setfenv(L, gi);

  return 1;
}

static luaL_Reg tk_aho_mt_fns[] = {
  { "predict", tk_aho_predict_lua },
  { "tag", tk_aho_tag_lua },
  { "persist", tk_aho_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_aho_builder_mt_fns[] = {
  { "n_states", tk_aho_builder_n_states_lua },
  { "build", tk_aho_builder_build_lua },
  { NULL, NULL }
};

static luaL_Reg tk_aho_fns[] = {
  { "create", tk_aho_create_lua },
  { "prepare", tk_aho_prepare_lua },
  { "load", tk_aho_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_aho (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_aho_fns, 0);
  return 1;
}

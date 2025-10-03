#include <santoku/lua/utils.h>
#include <santoku/klib.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/cumap.h>
#include <santoku/zumap.h>
#include <assert.h>
#include <ctype.h>

#define TK_TOKENIZER_MT "tk_tokenizer_t"
#define TK_TOKENIZER_EPH "tk_tokenizer_eph"

typedef struct {
  int max_len;
  int min_len;
  int max_run;
  int ngrams;
  int cgrams_min;
  int cgrams_max;
  int skips;
  int negations;
  int align;
  int next_id;
  tk_zumap_t *ids;
  tk_cumap_t *strs;
  tk_ivec_t *tokens;
  tk_ivec_t *window;
  tk_cvec_t *tmp_token;
  tk_cvec_t *tmp_skipgram;
  tk_cvec_t *tmp_append;
  int window_size;
  bool collected;
  bool finalized;
} tb_tokenizer_t;

static tb_tokenizer_t *peek_tokenizer (lua_State *L, int i)
{
  return (tb_tokenizer_t *) luaL_checkudata(L, i, TK_TOKENIZER_MT);
}

static inline int tb_tokenizer_gc (lua_State *L)
{
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  if (tokenizer->collected)
    return 0;
  tokenizer->collected = true;
  return 0;
}

static inline int tb_tokenizer_destroy (lua_State *L)
{
  return tb_tokenizer_gc(L);
}

static inline int tb_tokenizer_persist (lua_State *L)
{
  lua_settop(L, 2);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh;
  if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, (char *) &tokenizer->finalized, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->max_len, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->min_len, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->max_run, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->ngrams, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->cgrams_min, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->cgrams_max, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->skips, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->negations, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->align, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->next_id, sizeof(int), 1, fh);
  size_t nids = tk_zumap_size(tokenizer->ids);
  tk_lua_fwrite(L, (char *) &nids, sizeof(size_t), 1, fh);
  const char *tok;
  int id;
  tk_umap_foreach(tokenizer->ids, tok, id, ({
    size_t len = strlen(tok);
    tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, (char *) tok, len, 1, fh);
    tk_lua_fwrite(L, (char *) &id, sizeof(int), 1, fh);
  }))
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      tk_lua_fclose(L, fh);
      lua_pushlstring(L, data, len);
      free(data);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
  lua_settop(L, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
}

static inline char *tb_tokenizer_id_str (
  tb_tokenizer_t *tokenizer,
  int id
) {
  uint32_t k = tk_cumap_get(tokenizer->strs, id);
  assert(k != tk_cumap_end(tokenizer->strs));
  return (char *) tk_cumap_val(tokenizer->strs, k);
}

static inline int tb_tokenizer_new_token (
  lua_State *L,
  int Ti,
  tb_tokenizer_t *tokenizer,
  char **tokp,
  size_t len
) {
  char *tok = *tokp;
  int id, absent;
  uint32_t k = tk_zumap_get(tokenizer->ids, tok);
  if (k != tk_zumap_end(tokenizer->ids)) {
    id = tk_zumap_val(tokenizer->ids, k);
    *tokp = (char *) tk_zumap_key(tokenizer->ids, k);
  } else if (tokenizer->finalized) {
    return -1;
  } else {
    size_t str_len = strlen(tok);
    char *tmp = (char *) lua_newuserdata(L, str_len + 1);
    memcpy(tmp, tok, str_len + 1);
    id = tokenizer->next_id ++;
    k = tk_zumap_put(tokenizer->ids, tmp, &absent);
    tk_zumap_setval(tokenizer->ids, k, id);
    k = tk_cumap_put(tokenizer->strs, id, &absent);
    tk_cumap_setval(tokenizer->strs, k, tmp);
    tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
    lua_pop(L, 1);
    *tokp = tmp;
  }
  return id;
}

static inline void tb_tokenizer_append_cgrams (
  lua_State *L,
  int Ti,
  tb_tokenizer_t *tokenizer,
  char *tok
) {
  int toklen = (int) strlen(tok);
  int cmin = tokenizer->cgrams_min;
  int cmax = tokenizer->cgrams_max;
  if (cmax == 0 || toklen < cmin) return;
  if (cmax > toklen) cmax = toklen;
  for (int k = cmin; k <= cmax; k ++) {
    for (int s = 0; s + k <= toklen; s ++) {
      char buf[k + 1];
      memcpy(buf, tok + s, (size_t) k);
      buf[k] = '\0';
      char *bufp = buf;
      int id = tb_tokenizer_new_token(L, Ti, tokenizer, &bufp, (khint_t) k + 1);
      if (id != -1)
        tk_ivec_push(tokenizer->tokens, id);
    }
  }
}

static inline bool tb_tokenizer_is_delim_char (char c)
{
  return isspace((unsigned char)c) || (c != '#' && (
  c == '.' || c == ',' || c == ';' || c == ':'
  || c == '?' || c == '!' || c == '(' || c == ')'
  || c == '[' || c == ']' || c == '{' || c == '}'
  || c == '<' || c == '>' || c == '/' || c == '\\' || c == '\"'));
}

static inline int tb_have_bytes (const char *in, size_t i, size_t n)
{
  for (size_t k = 0; k < n; k++) if (in[i + k] == '\0') return 0;
  return 1;
}

static inline tk_cvec_t *tb_tokenizer_normalize (char *in, size_t len, int max_run)
{
  tk_cvec_t *out = tk_cvec_create(0, len, 0, 0);
  out->n = 0;
  char last = 0;
  int run = 0;

  for (size_t i = 0; i < len;) {
    if (last && ((isalpha((unsigned char)last) && isdigit((unsigned char)in[i])) ||
      (isdigit((unsigned char)last) && isalpha((unsigned char)in[i])))) {
      tk_cvec_push(out, ' ');
      last = ' ';
    }

    if (last && isalpha((unsigned char)last) && last == in[i])
      run ++;
    else
      run = 0;

    if (run >= max_run) { last = in[i]; i++; continue; }
    else { last = in[i]; }

    if (in[i] == '#') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#hash");
      tk_cvec_push(out, ' ');
      i ++;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) && in[i] == ':' && in[i + 1] == ')') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#smiley-positive");
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) && in[i] == ':' && tolower((unsigned char)in[i + 1]) == 'd') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#smiley-positive");
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) && in[i] == ':' && in[i + 1] == '(') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#smiley-negative");
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) && in[i] == ';' && in[i + 1] == ')') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#smiley-wink");
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    } else if (in[i] == '!') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#exclamation");
      tk_cvec_push(out, ' ');
      while (in[i] == '!') i ++;
      last = ' ';
      run = 0;
      continue;
    } else if (in[i] == '?') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#question");
      tk_cvec_push(out, ' ');
      while (in[i] == '?') i ++;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 3) && in[i] == '.' && in[i + 1] == '.' && in[i + 2] == '.') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#ellipsis");
      tk_cvec_push(out, ' ');
      i += 3;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) && in[i] == '\'' && in[i + 1] == 's' &&
      (!tb_have_bytes(in, i, 3) || in[i + 2] == '\0' || tb_tokenizer_is_delim_char(in[i + 2]))) {
      tk_cvec_push(out, ' ');
      tk_cvec_push(out, '\'');
      tk_cvec_push(out, 's');
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    }

    unsigned char c = (unsigned char) in[i];

    if (c < 0x80) {
      tk_cvec_push(out, tolower(c));
      i ++;
      continue;
    }

    if (tb_have_bytes(in, i, 2) && c == 0xC2 && (unsigned char) in[i + 1] == 0xA0) {
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    }

    if (tb_have_bytes(in, i, 3) && c == 0xE2 && (unsigned char) in[i + 1] == 0x80) {
      unsigned char b3 = (unsigned char) in[i + 2];
      switch (b3) {
        case 0x93: // en dash
        case 0x94: // em dash
        case 0x90: // hyphen
          tk_cvec_push(out, '-');
          last = '-';
          break;
        case 0x98: // left single quote
        case 0x99: // right single quote
          tk_cvec_push(out, '\'');     // don't drop; avoid "dont" vs "don't"
          last = '\'';
          break;
        case 0x9C: // left double quote
        case 0x9D: // right double quote
          tk_cvec_push(out, ' '); // prevent token concatenation
          last = ' ';
          break;
        case 0xA6: // ellipsis
          tk_cvec_push(out, ' ');
          tk_cvec_push_str(out, "#ellipsis");
          tk_cvec_push(out, ' ');
          last = ' ';
          break;
        default:
          last = ' ';
          break;
      }
      i += 3;
      run = 0;
      continue;
    }

    if (tb_have_bytes(in, i, 3) &&
      (unsigned char) c == '.' &&
      (unsigned char) in[i + 1] == '.' &&
      (unsigned char) in[i + 2] == '.') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#ellipsis");
      tk_cvec_push(out, ' ');
      i += 3;
      last = ' ';
      run = 0;
      continue;
    } else if (tb_have_bytes(in, i, 2) &&
      (unsigned char) c == '.' &&
      (unsigned char) in[i + 1] == '.') {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#ellipsis");
      tk_cvec_push(out, ' ');
      i += 2;
      last = ' ';
      run = 0;
      continue;
    }

    if (tb_have_bytes(in, i, 4) &&
      (unsigned char) c == 0xF0 &&
      (unsigned char) in[i + 1] == 0x9F &&
      (unsigned char) in[i + 2] == 0x98) {
      unsigned char b4 = (unsigned char) in[i + 3];
      if ((b4 >= 0x80 && b4 <= 0x8B) || (b4 >= 0x8F && b4 <= 0x90) || (b4 == 0x99)) {
        tk_cvec_push(out, ' ');
        tk_cvec_push_str(out, "#emoji-positive");
        tk_cvec_push(out, ' ');
      } else if ((b4 >= 0x9E && b4 <= 0xA2) || (b4 >= 0xA3 && b4 <= 0xA6)) {
        tk_cvec_push(out, ' ');
        tk_cvec_push_str(out, "#emoji-negative");
        tk_cvec_push(out, ' ');
      } else {
        tk_cvec_push(out, ' ');
        tk_cvec_push_str(out, "#emoji-other");
        tk_cvec_push(out, ' ');
      }
      i += 4;
      last = ' ';
      run = 0;
      continue;
    }

    if (tb_have_bytes(in, i, 4) &&
      (unsigned char)c == 0xF0 &&
      (unsigned char) in[i + 1] == 0x9F &&
      (unsigned char) in[i + 2] == 0x91 &&
      ((unsigned char) in[i + 3] == 0x8D || (unsigned char) in[i + 3] == 0x8E)) {
      if ((unsigned char) in[i + 3] == 0x8D) {
        tk_cvec_push(out, ' ');
        tk_cvec_push_str(out, "#emoji-positive");
        tk_cvec_push(out, ' ');
      } else {
        tk_cvec_push(out, ' ');
        tk_cvec_push_str(out, "#emoji-negative");
        tk_cvec_push(out, ' ');
      }
      i += 4;
      last = ' ';
      run = 0;
      continue;
    }

    if (tb_have_bytes(in, i, 3) &&
      (unsigned char)c == 0xE2 &&
      (unsigned char) in[i + 1] == 0x9D &&
      (unsigned char) in[i + 2] == 0xA4) {
      tk_cvec_push(out, ' ');
      tk_cvec_push_str(out, "#emoji-positive");
      tk_cvec_push(out, ' ');
      if (tb_have_bytes(in, i, 6) &&
        (unsigned char)in[i + 3] == 0xEF &&
        (unsigned char)in[i + 4] == 0xB8 &&
        (unsigned char)in[i + 5] == 0x8F) {
        i += 6; // ❤︎ (with VS16)
      } else {
        i += 3; // ❤
      }
      last = ' ';
      run = 0;
      continue;
    }

    if ((c & 0xE0) == 0xC0 && tb_have_bytes(in, i, 2)) i += 2;
    else if ((c & 0xF0) == 0xE0 && tb_have_bytes(in, i, 3)) i += 3;
    else if ((c & 0xF8) == 0xF0 && tb_have_bytes(in, i, 4)) i += 4;
    else i ++;
    last = ' ';
    run = 0;
  }

  tk_cvec_push(out, '\0');
  return out;
}

static const struct { char *from, *to1, *to2, *to3; } contractions[] = {
  { "aint"     , "are"    , "not"   , NULL } ,
  { "arent"    , "are"    , "not"   , NULL } ,
  { "cant"     , "can"    , "not"   , NULL } ,
  { "cannot"   , "can"    , "not"   , NULL } ,
  { "couldnt"  , "could"  , "not"   , NULL } ,
  { "couldve"  , "could"  , "have"  , NULL } ,
  { "didnt"    , "did"    , "not"   , NULL } ,
  { "dont"     , "do"     , "not"   , NULL } ,
  { "em"       , "them"   , NULL    , NULL } ,
  { "hadnt"    , "had"    , "not"   , NULL } ,
  { "havent"   , "have"   , "not"   , NULL } ,
  { "hed"      , "he"     , "would" , NULL } ,
  { "hedve"    , "he"     , "would" , "have" } ,
  { "hell"     , "he"     , "will"  , NULL } ,
  { "hes"      , "he"     , "is"    , NULL } ,
  { "hows"     , "how"    , "is"    , NULL } ,
  { "id"       , "i"      , "would" , NULL } ,
  { "idve"     , "i"      , "would" , "have" } ,
  { "ill"      , "i"      , "will"  , NULL } ,
  { "im"       , "i"      , "am"    , NULL } ,
  { "isnt"     , "is"     , "not"   , NULL } ,
  { "its"      , "it"     , "is"    , NULL } ,
  { "ive"      , "i"      , "have"  , NULL } ,
  { "lets"     , "let"    , "us"    , NULL } ,
  { "mightnt"  , "might"  , "not"   , NULL } ,
  { "mightve"  , "might"  , "have"  , NULL } ,
  { "mustnt"   , "must"   , "not"   , NULL } ,
  { "mustve"   , "must"   , "have"  , NULL } ,
  { "neednt"   , "need"   , "not"   , NULL } ,
  { "shant"    , "shall"  , "not"   , NULL } ,
  { "shed"     , "she"    , "would" , NULL } ,
  { "shedve"   , "she"    , "would" , "have" } ,
  { "shell"    , "she"    , "will"  , NULL } ,
  { "shes"     , "she"    , "is"    , NULL } ,
  { "shouldnt" , "should" , "not"   , NULL } ,
  { "shouldve" , "should" , "have"  , NULL } ,
  { "thats"    , "that"   , "is"    , NULL } ,
  { "theyd"    , "they"   , "would" , NULL } ,
  { "theydve"  , "they"   , "would" , "have" } ,
  { "theyll"   , "they"   , "will"  , NULL } ,
  { "theyre"   , "they"   , "are"   , NULL } ,
  { "theyve"   , "they"   , "have"  , NULL } ,
  { "wasnt"    , "was"    , "not"   , NULL } ,
  { "wed"      , "we"     , "would" , NULL } ,
  { "wedve"    , "we"     , "would" , "have" } ,
  { "well"     , "we"     , "will"  , NULL } ,
  { "were"     , "we"     , "are"   , NULL } ,
  { "werent"   , "were"   , "not"   , NULL } ,
  { "weve"     , "we"     , "have"  , NULL } ,
  { "whats"    , "what"   , "is"    , NULL } ,
  { "whens"    , "when"   , "is"    , NULL } ,
  { "wheres"   , "where"  , "is"    , NULL } ,
  { "whos"     , "who"    , "is"    , NULL } ,
  { "whove"    , "who"    , "have"  , NULL } ,
  { "whys"     , "why"    , "is"    , NULL } ,
  { "wont"     , "will"   , "not"   , NULL } ,
  { "wouldnt"  , "would"  , "not"   , NULL } ,
  { "wouldve"  , "would"  , "have"  , NULL } ,
  { "yall"     , "you"    , "all"   , NULL } ,
  { "youd"     , "you"    , "would" , NULL } ,
  { "youdve"   , "you"    , "would" , "have" } ,
  { "youll"    , "you"    , "will"  , NULL } ,
  { "youre"    , "you"    , "are"   , NULL } ,
  { "youve"    , "you"    , "have"  , NULL } ,
};

static const size_t n_contractions = sizeof(contractions)/sizeof(contractions[0]);

static inline bool tb_tokenizer_is_number_token (char *tok)
{
  if (!tok || !*tok) return false;
  for (char *p = tok; *p; p ++)
    if (!isdigit((unsigned char)*p))
      return false;
  return true;
}

static inline bool tb_tokenizer_is_emoji_token (char *tok)
{
  const char *pat = "#emoji-";
  size_t patlen = strlen(pat);
  size_t toklen = strlen(tok);
  return toklen > patlen && strncmp(pat, tok, patlen) == 0;
}

static inline bool tb_tokenizer_is_negation_token (char *tok)
{
  size_t len = strlen(tok);
  if (len >= 4 && !strcmp(tok + len - 4, "less"))
    return true;
  return
    !strcmp(tok, "not") ||
    !strcmp(tok, "no") ||
    !strcmp(tok, "none") ||
    !strcmp(tok, "nobody") ||
    !strcmp(tok, "nothing") ||
    !strcmp(tok, "nowhere") ||
    !strcmp(tok, "neither") ||
    !strcmp(tok, "nor") ||
    !strcmp(tok, "never") ||
    !strcmp(tok, "rarely") ||
    !strcmp(tok, "seldom") ||
    !strcmp(tok, "hardly") ||
    !strcmp(tok, "scarcely")||
    !strcmp(tok, "barely") ||
    !strcmp(tok, "without") ||
    !strcmp(tok, "nope") ||
    !strcmp(tok, "nah");
}

static inline void tb_copy_without_char (const char *src, char *dst, char ch)
{
  while (*src) { if (*src != ch) *dst++ = *src; src++; }
  *dst = '\0';
}

static inline bool tb_tokenizer_is_negation_boundary_char (char c)
{
  return c == '.' || c == '?' || c == '!';
}

static inline bool tb_tokenizer_is_negation_boundary_token (char *tok)
{
  return tok != NULL && (tb_tokenizer_is_negation_boundary_char(tok[0])
    || tb_tokenizer_is_emoji_token(tok));
}

static void tb_tokenizer_append_skipgram (
  lua_State *L,
  int Ti,
  tb_tokenizer_t *tok,
  int skipgram[],
  int to_pick,
  int max_skips,
  int rfirst,
  int start_idx,
  int winlen
) {
  if (to_pick == 0) {
    tok->tmp_skipgram->n = 0;
    for (int i = 0; i < rfirst; i ++) {
      int win_idx = skipgram[i];
      char *s = tb_tokenizer_id_str(tok, tok->window->a[win_idx]);
      tk_cvec_push_str(tok->tmp_skipgram, s);
      if (i + 1 < rfirst)
        tk_cvec_push(tok->tmp_skipgram, ' ');
    }
    tk_cvec_push(tok->tmp_skipgram, '\0');
    char *bufp = tok->tmp_skipgram->a;
    int nid = tb_tokenizer_new_token(L, Ti, tok, &bufp, tok->tmp_skipgram->n);
    if (nid != -1)
      tk_ivec_push(tok->tokens, nid);
    return;
  }
  int picked = (int) rfirst - (int) to_pick;
  int prev_idx = (picked == 0 ? -1 : skipgram[picked - 1]);
  for (int i = start_idx; i < (int) winlen; i ++) {
    if (prev_idx >= 0 && (i - prev_idx - 1) > max_skips)
      continue;
    skipgram[picked] = i;
    tb_tokenizer_append_skipgram(L, Ti, tok, skipgram, to_pick - 1, max_skips, rfirst, i + 1, winlen);
  }
}

static inline void tb_tokenizer_append_token (
  lua_State *L,
  int Ti,
  tb_tokenizer_t *tokenizer,
  char *word,
  int *negation
) {
  if (word == NULL)
    return;
  if (tb_tokenizer_is_negation_boundary_token(word))
    *negation = 0;
  bool is_negated = *negation > 0;
  if (*negation)
    (*negation) --;
  if (!word || word[0] == '\0')
    return;
  tokenizer->tmp_append->n = 0;
  if (tb_tokenizer_is_number_token(word)) {
    if (is_negated) {
      tk_cvec_push_str(tokenizer->tmp_append, "-#number");
    } else {
      tk_cvec_push_str(tokenizer->tmp_append, "#number");
    }
  } else if (is_negated && !tb_tokenizer_is_negation_token(word) && isalnum(word[0])) {
    tk_cvec_push(tokenizer->tmp_append, '-');
    tk_cvec_push_str(tokenizer->tmp_append, word);
  } else {
    tk_cvec_push_str(tokenizer->tmp_append, word);
  }
  if (tokenizer->ngrams == 0) {
    if (tb_tokenizer_is_negation_token(word)) {
      *negation = tokenizer->negations;
      return;
    }
    if (word[0] != '#' && strcmp(word, "'s") != 0)
      tb_tokenizer_append_cgrams(L, Ti, tokenizer, word);
    return;
  }

  // Normal mode: process unigrams and n-grams
  tk_cvec_push(tokenizer->tmp_append, '\0');
  char *bufp0 = tokenizer->tmp_append->a;
  int id0 = tb_tokenizer_new_token(L, Ti, tokenizer, &bufp0, tokenizer->tmp_append->n);
  if (id0 == -1) return;
  tk_ivec_push(tokenizer->tokens, id0);

  if (tb_tokenizer_is_negation_token(word)) {
    *negation = tokenizer->negations;
    return;
  }

  // Generate char-grams
  if (word[0] != '#' && strcmp(word, "'s") != 0)
    tb_tokenizer_append_cgrams(L, Ti, tokenizer, word);

  if ((int) tokenizer->window->n == tokenizer->window_size) {
    size_t n = tokenizer->window->n;
    if (n > 1)
      memmove(tokenizer->window->a, tokenizer->window->a + 1, sizeof *tokenizer->window->a * (n - 1));
    tokenizer->window->n = (n > 0 ? n - 1 : 0);
  }
  tk_ivec_push(tokenizer->window, id0);
  int winlen = tokenizer->window->n;

  if (tokenizer->ngrams > 0) {
    int skipgram[tokenizer->ngrams];
    for (int r = 2; r <= tokenizer->ngrams; r ++)
      tb_tokenizer_append_skipgram(L, Ti, tokenizer, skipgram, r, tokenizer->skips, r, 0, winlen);
  }
}

static inline void tb_tokenizer_populate_tokens (
  lua_State *L,
  int Ti,
  tb_tokenizer_t *tokenizer,
  const char *doc,
  size_t len
) {
  bool in_tag = false;
  bool skipping = false;
  int negation = 0;
  tokenizer->tmp_token->n = 0;
  tokenizer->tokens->n = 0;
  tokenizer->window->n = 0;
  for (size_t e = 0; e <= len; e ++) {
    char c = (e < len ? doc[e] : ' ');
    if (!in_tag && c == '<' && e + 1 < len && (isalpha((unsigned char) doc[e + 1]) || doc[e + 1] == '/')) {
      in_tag = true;
      continue;
    }
    if (in_tag) {
      if (c == '>') {
        in_tag = false;
        continue;
      }
      if (tb_tokenizer_is_delim_char(c))
        in_tag = false;
      else
        continue;
    }
    if (skipping && !tb_tokenizer_is_delim_char(c))
      continue;
    if (skipping && tb_tokenizer_is_delim_char(c))
      skipping = false;
    if (tb_tokenizer_is_negation_boundary_char(c))
      negation = 0;
    if (e < len && !tb_tokenizer_is_delim_char(c)) {
      tk_cvec_push(tokenizer->tmp_token, c);
      if ((int) tokenizer->tmp_token->n > tokenizer->max_len
        && tokenizer->tmp_token->a[0] != '#') {
        tokenizer->tmp_token->n = 0;
        skipping = true;
      }
    } else if ((int) tokenizer->tmp_token->n >= tokenizer->min_len) {
      tk_cvec_push(tokenizer->tmp_token, '\0');
      char *tok = tokenizer->tmp_token->a;
      bool was_contraction = false;
      char canon_buf[tokenizer->max_len + 1];
      tb_copy_without_char(tok, canon_buf, '\'');
      const char *key = canon_buf;
      for (size_t i = 0; i < n_contractions; i ++) {
        if (strcmp(key, contractions[i].from) == 0) {
          tb_tokenizer_append_token(L, Ti, tokenizer, contractions[i].to1, &negation);
          tb_tokenizer_append_token(L, Ti, tokenizer, contractions[i].to2, &negation);
          tb_tokenizer_append_token(L, Ti, tokenizer, contractions[i].to3, &negation);
          was_contraction = true;
          break;
        }
      }
      if (!was_contraction)
        tb_tokenizer_append_token(L, Ti, tokenizer, tok, &negation);
      tokenizer->tmp_token->n = 0;
    } else {
      tokenizer->tmp_token->n = 0;
    }
  }
}

static inline void _tb_tokenizer_parse (lua_State *L, int Ti, tb_tokenizer_t *tokenizer)
{
  size_t len;
  char *str = (char *) luaL_checklstring(L, 1, &len);
  tk_cvec_t *doc = tb_tokenizer_normalize(str, len, tokenizer->max_run);
  tb_tokenizer_populate_tokens(L, Ti, tokenizer, doc->a, doc->n);
  tk_cvec_destroy(doc);
  lua_Integer n = 1;
  lua_newtable(L);
  for (khint_t i = 0; i < tokenizer->tokens->n; i ++) {
    lua_pushinteger(L, n ++);
    int t = tokenizer->tokens->a[i];
    lua_pushstring(L, tb_tokenizer_id_str(tokenizer, t));
    lua_settable(L, -3);
  }
}

static inline int tb_tokenizer_features_aligned (tb_tokenizer_t *tokenizer)
{
  int features = (int) tokenizer->next_id;
  int align = tokenizer->align;
  return ((features + align - 1) / align) * align;
}

static inline int tb_tokenizer_parse (lua_State *L)
{
  lua_settop(L, 2);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  if (lua_type(L, 2) != LUA_TTABLE) {
    _tb_tokenizer_parse(L, 1, tokenizer);
  } else {
    for (size_t i = 1; i <= (size_t) lua_objlen(L, 2); i ++) {
      lua_pushinteger(L, (int64_t) i); // t n
      lua_gettable(L, -2); // t s
      _tb_tokenizer_parse(L, 1, tokenizer); // t s tt
      lua_pushinteger(L, (int64_t) i); // t s tt n
      lua_replace(L, -3); // t n tt
      lua_settable(L, -3); // t
    }
  }
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tb_tokenizer_tokenize (lua_State *L)
{
  lua_settop(L, 3); // tkz, t, out
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);

  tk_ivec_t *out = tk_ivec_peekopt(L, 3);
  if (out == NULL) {
    lua_pop(L, 1); // tks t
    out = tk_ivec_create(L, 0, 0, 0); // tkz t out
  } else {
    tk_ivec_clear(out); // tkz t out
  }

  int kha;
  uint64_t khi;
  tk_iuset_t *seen = tk_iuset_create(0, 0);
  uint64_t n_features = (uint64_t) tb_tokenizer_features_aligned(tokenizer);

  if (lua_type(L, 2) != LUA_TTABLE) { // tkz t out

    size_t doclen;
    char *odoc = (char *) luaL_checklstring(L, 2, &doclen); // tkz t out
    tk_cvec_t *ndoc = tb_tokenizer_normalize(odoc, doclen, tokenizer->max_run); // tkz t out
    tb_tokenizer_populate_tokens(L, 1, tokenizer, ndoc->a, ndoc->n);
    tk_cvec_destroy(ndoc);

    for (khint_t j = 0; j < tokenizer->tokens->n; j ++) {
      int id = tokenizer->tokens->a[j];
      if (id < 0)
        continue;
      khi = tk_iuset_put(seen, id, &kha);
      if (!kha)
        continue;
      tk_ivec_push(out, (int64_t) id);
    }

    // tkz t out

  } else {

    uint64_t n = (uint64_t) lua_objlen(L, 2); // tkz t out
    for (size_t i = 1; i <= n; i ++) {
      lua_rawgeti(L, 2, i); // tkz t out s
      size_t doclen;
      char *odoc = (char *) luaL_checklstring(L, -1, &doclen);
      tk_cvec_t *ndoc = tb_tokenizer_normalize(odoc, doclen, tokenizer->max_run); // t out
      tb_tokenizer_populate_tokens(L, 1, tokenizer, ndoc->a, ndoc->n);
      tk_cvec_destroy(ndoc);
      tk_iuset_clear(seen);
      for (khint_t j = 0; j < tokenizer->tokens->n; j ++) {
        int id = tokenizer->tokens->a[j];
        if (id < 0)
          continue;
        khi = tk_iuset_put(seen, id, &kha);
        if (!kha)
          continue;
        tk_ivec_push(out, (int64_t) id + ((int64_t) i - 1) * (int64_t) n_features);
      }
      lua_pop(L, 1); // tkz t out
    }

    // tkz t out

  }

  // tkz t out
  tk_iuset_destroy(seen);
  tk_ivec_shrink(out);
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tb_tokenizer_restrict (lua_State *L)
{
  lua_settop(L, 2);

  int Ti = tk_lua_absindex(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, Ti);
  tk_ivec_t *top_v = tk_ivec_peek(L, 2, "top_v");

  // Strings are userdata with ephemerons, so we can reuse them without strdup
  char *tok;
  int id, id0;
  int absent;
  khint_t k;
  tk_zumap_t *ids0 = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tk_cumap_t *strs0 = tk_cumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->next_id = 0;
  for (lua_Integer i = 0; i < (int64_t) top_v->n; i ++) {
    id = top_v->a[i];
    k = tk_cumap_get(tokenizer->strs, id);
    if (k == tk_cumap_end(tokenizer->strs))
      continue;
    tok = (char *) tk_cumap_val(tokenizer->strs, k);  // Reuse existing userdata
    id0 = tokenizer->next_id ++;
    k = tk_zumap_put(ids0, tok, &absent);
    tk_zumap_setval(ids0, k, id0);
    k = tk_cumap_put(strs0, id0, &absent);
    tk_cumap_setval(strs0, k, tok);
  }
  // String keys are now userdata managed by ephemerons, no manual free needed
  tk_lua_del_ephemeron(L, TK_TOKENIZER_EPH, Ti, tokenizer->ids);
  tk_zumap_destroy(tokenizer->ids);
  tokenizer->ids = ids0;
  tk_lua_del_ephemeron(L, TK_TOKENIZER_EPH, Ti, tokenizer->strs);
  tk_cumap_destroy(tokenizer->strs);
  tokenizer->strs = strs0;
  lua_settop(L, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}

static inline int tb_tokenizer_finalize (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, -1);
  if (tokenizer->finalized)
    return luaL_error(L, "already finalized");
  tokenizer->finalized = true;
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}

static inline int tb_tokenizer_index (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  lua_newtable(L);
  khint_t k;
  int id;
  char *tok;
  for (id = 0; id < tokenizer->next_id; id ++) {
    k = tk_cumap_get(tokenizer->strs, (uint32_t) id);
    tok = (char *) tk_cumap_val(tokenizer->strs, k);
    lua_pushinteger(L, id + 1); // t id
    lua_pushstring(L, tok); // t id str
    lua_settable(L, -3); // t
  }
  return 1;
}


static inline int tb_tokenizer_features (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  lua_pushinteger(L, (int64_t) tb_tokenizer_features_aligned(tokenizer));
  return 1;
}

static inline int tb_tokenizer_train (lua_State *L)
{
  lua_settop(L, 2);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  if (tokenizer->finalized)
    return tk_lua_error(L, "already finalized");
  tk_lua_fchecktype(L, 2, "train", "corpus", LUA_TTABLE);
  lua_getfield(L, 2, "corpus"); // corpus
  int n = lua_objlen(L, -1); //
  for (int i = 1; i <= n; i ++) {
    lua_pushinteger(L, (int64_t) i); // corpus i
    lua_gettable(L, -2); // corpus v
    size_t len;
    tk_cvec_t *doc = tb_tokenizer_normalize((char *) luaL_checklstring(L, -1, &len), len, tokenizer->max_run);
    tb_tokenizer_populate_tokens(L, 1, tokenizer, doc->a, doc->n);
    tk_cvec_destroy(doc);
    lua_pop(L, 1); // corpus
  }
  lua_settop(L, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}

static luaL_Reg tb_mt_fns[] =
{
  { "train", tb_tokenizer_train },
  { "tokenize", tb_tokenizer_tokenize },
  { "parse", tb_tokenizer_parse },
  { "features", tb_tokenizer_features },
  { "finalize", tb_tokenizer_finalize },
  { "restrict", tb_tokenizer_restrict },
  { "index", tb_tokenizer_index },
  { "persist", tb_tokenizer_persist },
  { "destroy", tb_tokenizer_destroy },
  { NULL, NULL }
};

static inline int tb_tokenizer_create (lua_State *L)
{
  lua_settop(L, 1);
  int max_len = (int) tk_lua_fcheckunsigned(L, 1, "create", "max_len");
  int min_len = (int) tk_lua_fcheckunsigned(L, 1, "create", "min_len");
  int max_run = (int) tk_lua_fcheckunsigned(L, 1, "create", "max_run");
  int ngrams = (int) tk_lua_fcheckunsigned(L, 1, "create", "ngrams");
  int cgrams_min = (int) tk_lua_fcheckunsigned(L, 1, "create", "cgrams_min");
  int cgrams_max = (int) tk_lua_fcheckunsigned(L, 1, "create", "cgrams_max");
  int skips = (int) tk_lua_fcheckunsigned(L, 1, "create", "skips");
  int negations = (int) tk_lua_fcheckunsigned(L, 1, "create", "negations");
  int align = (int) tk_lua_foptunsigned(L, 1, "create", "align", 1);
  // TODO: Get nthreads
  if (max_run < 1)
    luaL_error(L, "max_run must be greater than or equal to 1");
  if (min_len == 0)
    luaL_error(L, "min_len must be greater than or equal to 1");
  if (max_len < min_len)
    luaL_error(L, "max_len must be greater than or equal to min_len");
  if (!align)
    luaL_error(L, "alignment must be greater than 0 (use 128 for TM)");
  tb_tokenizer_t *tokenizer = tk_lua_newuserdata(L, tb_tokenizer_t, TK_TOKENIZER_MT, tb_mt_fns, tb_tokenizer_gc);
  int Ti = tk_lua_absindex(L, -1);
  tokenizer->tokens = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->window = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_token = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_append = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_skipgram = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->ids = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->strs = tk_cumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->next_id = 0;
  tokenizer->max_len = max_len;
  tokenizer->min_len = min_len;
  tokenizer->max_run = max_run;
  tokenizer->ngrams = ngrams;
  tokenizer->cgrams_min = cgrams_min;
  tokenizer->cgrams_max = cgrams_max;
  tokenizer->skips = skips;
  tokenizer->negations = negations;
  tokenizer->align = align;
  tokenizer->window_size = ngrams > 0 ? ngrams + (ngrams - 1) * skips : 0;
  lua_replace(L, 1);
  return 1;
}

static inline int tb_tokenizer_load (lua_State *L)
{
  // TODO: 2nd param currently ignored, will be used for n_threads
  lua_settop(L, 3);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 3) == LUA_TBOOLEAN && lua_toboolean(L, 3);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tb_tokenizer_t *tokenizer = tk_lua_newuserdata(L, tb_tokenizer_t, TK_TOKENIZER_MT, tb_mt_fns, tb_tokenizer_gc);
  int Ti = tk_lua_absindex(L, -1);
  tokenizer->tokens = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->window = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_token = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_append = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->tmp_skipgram = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->ids = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tokenizer->strs = tk_cumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);
  lua_pop(L, 1);
  tk_lua_fread(L, &tokenizer->finalized, sizeof(bool), 1, fh);
  tk_lua_fread(L, &tokenizer->max_len, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->min_len, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->max_run, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->ngrams, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->cgrams_min, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->cgrams_max, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->skips, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->negations, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->align, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->next_id, sizeof(int), 1, fh);
  size_t nids;
  uint32_t k;
  int absent;
  tk_lua_fread(L, (char *) &nids, sizeof(size_t), 1, fh);
  for (khint_t i = 0; i < nids; i ++) {
    size_t len;
    tk_lua_fread(L, &len, sizeof(size_t), 1, fh);
    char *tokn = (char *) lua_newuserdata(L, len + 1);  // GC-tracked on stack
    tk_lua_fread(L, tokn, len, 1, fh);  // Safe: if throws, userdata GC'd
    tokn[len] = 0;
    int id;
    tk_lua_fread(L, &id, sizeof(int), 1, fh);
    k = tk_zumap_put(tokenizer->ids, tokn, &absent);
    assert(absent);
    tk_zumap_setval(tokenizer->ids, k, id);
    k = tk_cumap_put(tokenizer->strs, id, &absent);
    assert(absent);
    tk_cumap_setval(tokenizer->strs, k, tokn);
    tk_lua_add_ephemeron(L, TK_TOKENIZER_EPH, Ti, -1);  // Anchor to tokenizer
    lua_pop(L, 1);  // Pop userdata (string remains via ephemeron)
  }
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tb_fns[] =
{
  { "create", tb_tokenizer_create },
  { "load", tb_tokenizer_load },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_tokenizer (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tb_fns, 0);
  return 1;
}

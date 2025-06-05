#define _GNU_SOURCE

#include <santoku/lua/utils.h>
#include <santoku/klib.h>
#include <santoku/iuset.h>
#include <santoku/ivec.h>
#include <assert.h>
#include <ctype.h>

#define MT_TOKENIZER "santoku_tokenizer"

typedef struct {
  int id;
  int df;
} tk_sort_pair_t;

#define tk_sort_pair_lt(a, b) ((a).df < (b).df)

KSORT_INIT(sort_pair, tk_sort_pair_t, tk_sort_pair_lt);
KHASH_MAP_INIT_STR(ids, int);
KHASH_MAP_INIT_INT(strs, char *);
KHASH_MAP_INIT_INT(dfs, int);
KHASH_SET_INIT_INT(seen);

typedef khash_t(ids) tb_ids_t;
typedef khash_t(strs) tb_strs_t;
typedef khash_t(dfs) tb_df_t;
typedef khash_t(seen) tb_seen_t;
typedef kvec_t(int) tb_tokens_t;
typedef kvec_t(char) tb_token_t;

typedef struct {
  double max_df;
  double min_df;
  int max_len;
  int min_len;
  int max_run;
  int ngrams;
  int cgrams_min;
  int cgrams_max;
  int skips;
  int negations;
  int align;
  int ndocs;
  int next_id;
  tb_ids_t *ids;
  tb_strs_t *strs;
  tb_df_t *dfs;
  tb_seen_t *tmp_seen;
  tb_tokens_t tokens;
  tb_tokens_t window;
  tb_token_t tmp_token;
  tb_token_t tmp_skipgram;
  tb_token_t tmp_append;
  int window_size;
  bool collected;
  bool finalized;
} tb_tokenizer_t;

static tb_tokenizer_t *peek_tokenizer (lua_State *L, int i)
{
  return (tb_tokenizer_t *) luaL_checkudata(L, i, MT_TOKENIZER);
}

static inline int tb_tokenizer_gc (lua_State *L)
{
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, 1);
  if (tokenizer->collected)
    return 0;
  tokenizer->collected = true;
  for (khint_t k = kh_begin(tokenizer->strs); k < kh_end(tokenizer->strs); k ++)
    if (kh_exist(tokenizer->strs, k))
      free((char *) kh_value(tokenizer->strs, k));
  kv_destroy(tokenizer->tokens);
  kv_destroy(tokenizer->tmp_token);
  kv_destroy(tokenizer->tmp_append);
  kv_destroy(tokenizer->tmp_skipgram);
  kv_destroy(tokenizer->window);
  kh_destroy(ids, tokenizer->ids);
  kh_destroy(strs, tokenizer->strs);
  if (tokenizer->dfs)
    kh_destroy(dfs, tokenizer->dfs);
  if (tokenizer->tmp_seen)
    kh_destroy(seen, tokenizer->tmp_seen);
  return 0;
}

static inline int tb_tokenizer_destroy (lua_State *L)
{
  lua_settop(L, 0);
  lua_pushvalue(L, lua_upvalueindex(1));
  return tb_tokenizer_gc(L);
}

static inline int tb_tokenizer_persist (lua_State *L)
{
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  lua_settop(L, 1);
  bool tostr = lua_type(L, 1) == LUA_TNIL;
  FILE *fh;
  if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    fh = tk_lua_fopen(L, luaL_checkstring(L, 1), "w");
  tk_lua_fwrite(L, (char *) &tokenizer->finalized, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->max_df, sizeof(double), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->min_df, sizeof(double), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->max_len, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->min_len, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->max_run, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->ngrams, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->cgrams_min, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->cgrams_max, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->skips, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->negations, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->align, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->ndocs, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &tokenizer->next_id, sizeof(int), 1, fh);
  tk_lua_fwrite(L, (char *) &kh_size(tokenizer->ids), sizeof(khint_t), 1, fh);
  for (khint_t i = kh_begin(tokenizer->ids); i < kh_end(tokenizer->ids); i ++)
    if (kh_exist(tokenizer->ids, i)) {
      char *tok = (char *) kh_key(tokenizer->ids, i);
      size_t len = strlen(tok);
      int id = kh_value(tokenizer->ids, i) ;
      tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
      tk_lua_fwrite(L, tok, len, 1, fh);
      tk_lua_fwrite(L, (char *) &id, sizeof(int), 1, fh);
    }
  if (!tokenizer->finalized) {
    tk_lua_fwrite(L, (char *) &kh_size(tokenizer->dfs), sizeof(khint_t), 1, fh);
    for (khint_t i = kh_begin(tokenizer->dfs); i < kh_end(tokenizer->dfs); i ++)
      if (kh_exist(tokenizer->dfs, i)) {
        int id = (int) kh_key(tokenizer->dfs, i);
        int df = (int) kh_value(tokenizer->dfs, i);
        tk_lua_fwrite(L, (char *) &id, sizeof(int), 1, fh);
        tk_lua_fwrite(L, (char *) &df, sizeof(int), 1, fh);
      }
  }
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

static inline char *tb_tokenizer_id_str (
  tb_tokenizer_t *tokenizer,
  int id
) {
  khint_t k = kh_get(strs, tokenizer->strs, (uint32_t) id);
  assert(k != kh_end(tokenizer->strs));
  return (char *) kh_value(tokenizer->strs, k);
}

static inline int tb_tokenizer_str_id (
  tb_tokenizer_t *tokenizer,
  char *str
) {
  khint_t k = kh_get(ids, tokenizer->ids, str);
  assert(k != kh_end(tokenizer->ids));
  return kh_value(tokenizer->ids, k);
}

static inline int tb_tokenizer_new_token (
  tb_tokenizer_t *tokenizer,
  char **tokp,
  bool train,
  size_t len
) {
  char *tok = *tokp;
  int id, absent;
  khint_t k = kh_get(ids, tokenizer->ids, tok);
  if (k != kh_end(tokenizer->ids)) {
    id = kh_value(tokenizer->ids, k);
    *tokp = (char *) kh_key(tokenizer->ids, k);
  } else if (!train) {
    return -1;
  } else {
    char *tmp = strdup(tok);
    id = tokenizer->next_id ++;
    k = kh_put(ids, tokenizer->ids, tmp, &absent);
    assert(absent);
    kh_value(tokenizer->ids, k) = id;
    k = kh_put(strs, tokenizer->strs, (uint32_t) id, &absent);
    assert(absent);
    kh_value(tokenizer->strs, k) = tmp;
    *tokp = tmp;
  }
  if (train) {
    k = kh_put(seen, tokenizer->tmp_seen, (uint32_t) id, &absent);
    if (absent) {
      k = kh_put(dfs, tokenizer->dfs, (uint32_t) id, &absent);
      kh_value(tokenizer->dfs, k) = (absent ? 0 : kh_value(tokenizer->dfs, k)) + 1;
    }
  }
  return id;
}

static inline void tb_tokenizer_append_cgrams (
  tb_tokenizer_t *tokenizer,
  char *tok,
  bool train
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
      int id = tb_tokenizer_new_token(tokenizer, &bufp, train, (khint_t) k + 1);
      if (id != -1)
        kv_push(int, tokenizer->tokens, id);
    }
  }
}

#define kv_push_str(type, vec, str) \
  do { \
    for (char *p__ = (str); *p__; p__ ++) \
      kv_push(type, vec, *p__); \
  } while (0)

static inline char *tb_tokenizer_normalize (char *in, size_t *len, int max_run)
{
  kvec_t(char) out;
  kv_init(out);
  char last = 0;
  int run = 0;
  for (size_t i = 0; in[i];) {
    if (last && ((isalpha(last) && isdigit(in[i])) || (isdigit(last) && isalpha(in[i])))) {
      kv_push(char, out, ' ');
      last = ' ';
    }
    if (last && isalpha(last) && last == in[i])
      run ++;
    else
      run = 0;
    if (run >= max_run) {
      last = in[i];
      i ++;
      continue;
    } else {
      last = in[i];
    }
    if (in[i] == '#') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#hash");
      kv_push(char, out, ' ');
      i ++;
    } else if (in[i] == ':' && in[i + 1] == ')') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#smiley-positive");
      kv_push(char, out, ' ');
      i += 2;
      continue;
    } else if (in[i] == ':' && in[i + 1] == 'D') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#smiley-positive");
      kv_push(char, out, ' ');
      i += 2;
      continue;
    } else if (in[i] == ':' && in[i + 1] == '(') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#smiley-negative");
      kv_push(char, out, ' ');
      i += 2;
      continue;
    } else if (in[i] == ';' && in[i + 1] == ')') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#smiley-wink");
      kv_push(char, out, ' ');
      i += 2;
      continue;
    } else if (in[i] == '!') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#exclamation");
      kv_push(char, out, ' ');
      while (in[i] == '!') i ++;
      continue;
    } else if (in[i] == '?') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#question");
      kv_push(char, out, ' ');
      while (in[i]=='?') i ++;
      continue;
    } else if (in[i] == '.' && in[i + 1] == '.' && in[i + 2] == '.') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#ellipsis");
      kv_push(char, out, ' ');
      i += 3;
      continue;
    }
    unsigned char c = (unsigned char) in[i];
    if (c < 0x80) {
      kv_push(char, out, tolower(c));
      i ++;
    } else if (c == 0xC2 && (unsigned char) in[i + 1] == 0xA0) {
      kv_push(char, out, ' ');
      i += 2;
    } else if (c == 0xE2 && (unsigned char) in[i + 1] == 0x80) {
      unsigned char b3 = (unsigned char) in[i + 2];
      switch (b3) {
        case 0x93: // en dash
        case 0x94: // em dash
        case 0x90: // hyphen
          kv_push(char, out, '-');
          break;
        case 0x98: // left single quote
        case 0x99: // right single quote
          break;
        case 0x9C: // left double quote
        case 0x9D: // right double quote
          break;
        case 0xA6: // ellipsis
          kv_push(char, out, ' ');
          kv_push_str(char, out, "#ellipsis");
          kv_push(char, out, ' ');
          break;
        default:
          // skip unknown
          break;
      }
      i += 3;
    } else if ((unsigned char) c == '.'
      && (unsigned char) in[i + 1] == '.'
      && (unsigned char) in[i + 2] == '.') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#ellipsis");
      kv_push(char, out, ' ');
      i += 3;
    } else if ((unsigned char) c == '.'
      && (unsigned char) in[i + 1] == '.') {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#ellipsis");
      kv_push(char, out, ' ');
      i += 2;
    } else if ((unsigned char) c == 0xF0
      && (unsigned char) in[i + 1] == 0x9F
      && (unsigned char) in[i + 2] == 0x98) {
      unsigned char b4 = (unsigned char) in[i + 3];
      if ((b4 >= 0x80 && b4 <= 0x8B)
        || (b4 >= 0x8F && b4 <= 0x90)
        || (b4 == 0x99)) {
        kv_push(char, out, ' ');
        kv_push_str(char, out, "#emoji-positive");
        kv_push(char, out, ' ');
      } else if ((b4 >= 0x9E && b4 <= 0xA2)
        || (b4 >= 0xA3 && b4 <= 0xA6)) {
        kv_push(char, out, ' ');
        kv_push_str(char, out, "#emoji-negative");
        kv_push(char, out, ' ');
      } else {
        kv_push(char, out, ' ');
        kv_push_str(char, out, "#emoji-other");
        kv_push(char, out, ' ');
      }
      i += 4;
      continue;
    } else if ((unsigned char)c == 0xF0
      && (unsigned char) in[i + 1] == 0x9F
      && (unsigned char) in[i + 2] == 0x91
      && ((unsigned char) in[i + 3] == 0x8D
      ||  (unsigned char) in[i + 3] == 0x8E))
    {
      if ((unsigned char) in[i + 3] == 0x8D) {
        kv_push(char, out, ' ');
        kv_push_str(char, out, "#emoji-positive");
        kv_push(char, out, ' ');
      } else {
        kv_push(char, out, ' ');
        kv_push_str(char, out, "#emoji-negative");
        kv_push(char, out, ' ');
      }
      i += 4;
      continue;
    } else if ((unsigned char)c == 0xE2
      && (unsigned char) in[i + 1] == 0x9D
      && (unsigned char) in[i + 2] == 0xA4) {
      kv_push(char, out, ' ');
      kv_push_str(char, out, "#emoji-positive");
      kv_push(char, out, ' ');
      i += (size_t) (3 + (in[i + 3] == (char)0xEF ? 2 : 0));
      continue;
    } else {
      if ((c & 0xE0) == 0xC0) i += 2;
      else if ((c & 0xF0) == 0xE0) i += 3;
      else if ((c & 0xF8) == 0xF0) i += 4;
      else i ++;
    }
  }
  kv_push(char, out, '\0');
  *len = out.n - 1;
  return out.a;
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

static inline bool tb_tokenizer_is_negation_boundary_char (char c)
{
  return c == '.' || c == '?' || c == '!';
}

static inline bool tb_tokenizer_is_negation_boundary_token (char *tok)
{
  return tok != NULL && (tb_tokenizer_is_negation_boundary_char(tok[0])
    || tb_tokenizer_is_emoji_token(tok));
}

static inline bool tb_tokenizer_is_delim_char (char c)
{
  return isspace((unsigned char)c) || (c != '#' && (
  c == '.' || c == ',' || c == ';' || c == ':'
  || c == '?' || c == '!' || c == '(' || c == ')'
  || c == '[' || c == ']' || c == '{' || c == '}'
  || c == '<' || c == '>' || c == '/' || c == '\\' || c == '\"'));
}

static inline bool tb_tokenizer_is_strip_char (char c)
{
  return c == '\'' || c == '_' || c == '`' || c == '-';
}

static void tb_tokenizer_append_skipgram (
  tb_tokenizer_t *tok,
  int skipgram[],
  int to_pick,
  int max_skips,
  int rfirst,
  int start_idx,
  int winlen,
  bool train
) {
  if (to_pick == 0) {
    char *first = tb_tokenizer_id_str(tok, tok->window.a[skipgram[0]]);
    if (strlen(first) == 1)
      return;
    tok->tmp_skipgram.n = 0;
    for (int i = 0; i < rfirst; i ++) {
      int win_idx = skipgram[i];
      char *s = tb_tokenizer_id_str(tok, tok->window.a[win_idx]);
      kv_push_str(char, tok->tmp_skipgram, s);
      if (i + 1 < rfirst)
        kv_push(char, tok->tmp_skipgram, ' ');
    }
    kv_push(char, tok->tmp_skipgram, '\0');
    char *bufp = tok->tmp_skipgram.a;
    int nid = tb_tokenizer_new_token(tok, &bufp, train, tok->tmp_skipgram.n);
    if (nid != -1)
      kv_push(int, tok->tokens, nid);
    return;
  }
  int picked = (int) rfirst - (int) to_pick;
  int prev_idx = (picked == 0 ? -1 : skipgram[picked - 1]);
  for (int i = start_idx; i < (int) winlen; i ++) {
    if (prev_idx >= 0 && (i - prev_idx - 1) > max_skips)
      continue;
    skipgram[picked] = i;
    tb_tokenizer_append_skipgram(tok, skipgram, to_pick - 1, max_skips, rfirst, i + 1, winlen, train);
  }
}

static inline void tb_tokenizer_append_token (
  tb_tokenizer_t *tokenizer,
  char *word,
  int *negation,
  bool train
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
  tokenizer->tmp_append.n = 0;
  if (tb_tokenizer_is_number_token(word)) {
    if (is_negated) {
      kv_push_str(char, tokenizer->tmp_append, "-#number");
    } else {
      kv_push_str(char, tokenizer->tmp_append, "#number");
    }
  } else if (is_negated && !tb_tokenizer_is_negation_token(word) && isalnum(word[0])) {
    kv_push(char, tokenizer->tmp_append, '-');
    kv_push_str(char, tokenizer->tmp_append, word);
  } else {
    kv_push_str(char, tokenizer->tmp_append, word);
  }
  kv_push(char, tokenizer->tmp_append, '\0');
  char *bufp0 = tokenizer->tmp_append.a;
  int id0 = tb_tokenizer_new_token(tokenizer, &bufp0, train, tokenizer->tmp_append.n);
  if (id0 == -1) return;
  kv_push(int, tokenizer->tokens, id0);
  if (tb_tokenizer_is_negation_token(word)) {
    *negation = tokenizer->negations;
    return;
  }
  if (word[0] != '#')
    tb_tokenizer_append_cgrams(tokenizer, word, train);
  if ((int) kv_size(tokenizer->window) == tokenizer->window_size) {
    size_t n = kv_size(tokenizer->window);
    if (n > 1)
      memmove(tokenizer->window.a, tokenizer->window.a + 1, sizeof *tokenizer->window.a * (n - 1));
    kv_size(tokenizer->window) = (n > 0 ? n - 1 : 0);
  }
  kv_push(int, tokenizer->window, id0);
  int winlen = kv_size(tokenizer->window);
  int skipgram[tokenizer->ngrams];
  for (int r = 2; r <= tokenizer->ngrams; r ++)
    tb_tokenizer_append_skipgram(tokenizer, skipgram, r, tokenizer->skips, r, 0, winlen, train);
}

static inline void tb_tokenizer_populate_tokens (
  tb_tokenizer_t *tokenizer,
  const char *doc,
  size_t len,
  bool train
) {
  bool in_tag = false;
  bool skipping = false;
  int negation = 0;
  kh_clear(seen, tokenizer->tmp_seen);
  kv_size(tokenizer->tmp_token) = 0;
  kv_size(tokenizer->tokens) = 0;
  kv_size(tokenizer->window) = 0;
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
      if (tb_tokenizer_is_strip_char(c) && (tokenizer->tmp_token.n == 0 || tokenizer->tmp_token.a[0] != '#'))
        continue;
      kv_push(char, tokenizer->tmp_token, c);
      if ((int) tokenizer->tmp_token.n > tokenizer->max_len
        && tokenizer->tmp_token.a[0] != '#') {
        tokenizer->tmp_token.n = 0;
        skipping = true;
      }
    } else if ((int) tokenizer->tmp_token.n >= tokenizer->min_len) {
      kv_push(char, tokenizer->tmp_token, '\0');
      char *tok = tokenizer->tmp_token.a;
      bool was_contraction = false;
      for (size_t i = 0; i < n_contractions; i ++) {
        if (strcmp(tok, contractions[i].from) == 0) {
          tb_tokenizer_append_token(tokenizer, contractions[i].to1, &negation, train);
          tb_tokenizer_append_token(tokenizer, contractions[i].to2, &negation, train);
          tb_tokenizer_append_token(tokenizer, contractions[i].to3, &negation, train);
          was_contraction = true;
          break;
        }
      }
      if (!was_contraction)
        tb_tokenizer_append_token(tokenizer, tok, &negation, train);
      tokenizer->tmp_token.n = 0;
    } else {
      tokenizer->tmp_token.n = 0;
    }
  }
}

static inline void _tb_tokenizer_parse (lua_State *L, tb_tokenizer_t *tokenizer)
{
  size_t len;
  char *doc = tb_tokenizer_normalize((char *) luaL_checklstring(L, 1, &len), &len, tokenizer->max_run);
  tb_tokenizer_populate_tokens(tokenizer, doc, len, false);
  free(doc);
  lua_Integer n = 1;
  lua_newtable(L);
  for (khint_t i = 0; i < kv_size(tokenizer->tokens); i ++) {
    lua_pushinteger(L, n ++);
    int t = kv_A(tokenizer->tokens, i);
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
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  lua_settop(L, 1);
  if (lua_type(L, 1) != LUA_TTABLE) {
    _tb_tokenizer_parse(L, tokenizer);
  } else {
    for (size_t i = 1; i <= (size_t) lua_objlen(L, 1); i ++) {
      lua_pushinteger(L, (int64_t) i); // t n
      lua_gettable(L, -2); // t s
      _tb_tokenizer_parse(L, tokenizer); // t s tt
      lua_pushinteger(L, (int64_t) i); // t s tt n
      lua_replace(L, -3); // t n tt
      lua_settable(L, -3); // t
    }
  }
  return 1;
}

static inline int tb_tokenizer_tokenize (lua_State *L)
{
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  uint64_t n = (uint64_t) lua_objlen(L, 1);
  uint64_t n_features = (uint64_t) tb_tokenizer_features_aligned(tokenizer);
  int kha;
  uint64_t khi;

  tk_iuset_t *seen = tk_iuset_create(); // seen
  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0); // seen, out

  // TODO: parallelize
  for (size_t i = 1; i <= n; i ++) {
    lua_pushinteger(L, (int64_t) i); // t n
    lua_gettable(L, 1); // t s

    size_t doclen;
    char *doc = tb_tokenizer_normalize((char *) luaL_checklstring(L, -1, &doclen), &doclen, tokenizer->max_run); // s
    tb_tokenizer_populate_tokens(tokenizer, doc, doclen, false);
    free(doc);

    tk_iuset_clear(seen);
    for (khint_t j = 0; j < tokenizer->tokens.n; j ++) {
      int id = tokenizer->tokens.a[j];
      if (id < 0)
        continue;
      khi = tk_iuset_put(seen, id, &kha);
      if (!kha)
        continue;
      tk_ivec_push(out, (int64_t) id + ((int64_t) i - 1) * (int64_t) n_features);
    }
    lua_pop(L, 1); // t
  }

  tk_iuset_destroy(seen);
  tk_ivec_shrink(L, out);
  return 1;
}

static inline int tb_tokenizer_restrict (lua_State *L)
{
  lua_settop(L, 1);

  tk_ivec_t *top_v = tk_ivec_peek(L, 1, "top_v");

  // TODO: Can we do this without strdup? Just don't free the ones we're
  // keeping?
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  char *tok;
  int64_t id, id0;
  int absent;
  khint_t k;
  tb_ids_t *ids0 = kh_init(ids);
  tb_strs_t *strs0 = kh_init(strs);
  tokenizer->next_id = 0;
  for (lua_Integer i = 0; i < (int64_t) top_v->n; i ++) {
    id = top_v->a[i];
    lua_pop(L, 1);
    k = kh_get(strs, tokenizer->strs, (uint32_t) id);
    if (k == kh_end(tokenizer->strs))
      continue;
    tok = (char *) strdup(kh_value(tokenizer->strs, k));
    id0 = tokenizer->next_id ++;
    k = kh_put(ids, ids0, tok, &absent);
    assert(absent);
    kh_value(ids0, k) = id0;
    k = kh_put(strs, strs0, (uint32_t) id0, &absent);
    assert(absent);
    kh_value(strs0, k) = tok;
  }
  for (khint_t k = kh_begin(tokenizer->strs); k < kh_end(tokenizer->strs); k ++)
    if (kh_exist(tokenizer->strs, k))
      free((char *) kh_value(tokenizer->strs, k));
  kh_destroy(ids, tokenizer->ids);
  kh_destroy(strs, tokenizer->strs);
  tokenizer->ids = ids0;
  tokenizer->strs = strs0;
  return 0;
}

static inline int tb_tokenizer_finalize (lua_State *L)
{
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));

  if (tokenizer->finalized)
    return luaL_error(L, "already finalized");;

  tokenizer->finalized = true;

  char *tok;
  double df;
  int id, id0, absent;
  khint_t i, k;
  tb_ids_t *ids0 = kh_init(ids);
  tb_strs_t *strs0 = kh_init(strs);

  kvec_t(tk_sort_pair_t) sort;
  kv_init(sort);

  // Delete tokens with df > max_df
  for (i = kh_begin(tokenizer->ids); i < kh_end(tokenizer->ids); i ++)
    if (kh_exist(tokenizer->ids, i)) {
      tok = (char *) kh_key(tokenizer->ids, i);
      id = kh_value(tokenizer->ids, i);
      k = kh_get(dfs, tokenizer->dfs, (uint32_t) id);
      assert(k != kh_end(tokenizer->dfs));
      df = (double) kh_value(tokenizer->dfs, k) / (double) tokenizer->ndocs;
      if (df > tokenizer->max_df || df < tokenizer->min_df) {
        kh_del(ids, tokenizer->ids, i);
        k = kh_get(strs, tokenizer->strs, (uint32_t) id);
        assert(k != kh_end(tokenizer->strs));
        kh_del(strs, tokenizer->strs, k);
        free(tok);
      } else {
        tk_sort_pair_t p = { .id = id, .df = df };
        kv_push(tk_sort_pair_t, sort, p);
      }
    }
  kh_destroy(dfs, tokenizer->dfs);
  tokenizer->dfs = NULL;
  kh_destroy(seen, tokenizer->tmp_seen);
  tokenizer->tmp_seen = NULL;

  // Renumber tokens
  ks_introsort(sort_pair, sort.n, sort.a);
  tokenizer->next_id = 0;
  for (khint_t i = 0; i < sort.n; i ++) {
    id = sort.a[i].id;
    k = kh_get(strs, tokenizer->strs, (uint32_t) id);
    assert(k != kh_end(tokenizer->strs));
    tok = (char *) kh_value(tokenizer->strs, k);
    id0 = tokenizer->next_id ++;
    k = kh_put(ids, ids0, tok, &absent);
    assert(absent);
    kh_value(ids0, k) = id0;
    k = kh_put(strs, strs0, (uint32_t) id0, &absent);
    assert(absent);
    kh_value(strs0, k) = tok;
  }

  kv_destroy(sort);
  kh_destroy(ids, tokenizer->ids);
  kh_destroy(strs, tokenizer->strs);
  tokenizer->ids = ids0;
  tokenizer->strs = strs0;

  return 0;
}

static inline int tb_tokenizer_index (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  lua_newtable(L);
  khint_t k;
  int id;
  char *tok;
  for (id = 0; id < tokenizer->next_id; id ++) {
    k = kh_get(strs, tokenizer->strs, (uint32_t) id);
    tok = (char *) kh_value(tokenizer->strs, k);
    lua_pushinteger(L, id + 1); // t id
    lua_pushstring(L, tok); // t id str
    lua_settable(L, -3); // t
  }
  return 1;
}

static inline int tb_tokenizer_features (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  lua_pushinteger(L, (int64_t) tb_tokenizer_features_aligned(tokenizer));
  return 1;
}

static inline int tb_tokenizer_train (lua_State *L)
{
  lua_settop(L, 1);
  tb_tokenizer_t *tokenizer = peek_tokenizer(L, lua_upvalueindex(1));
  if (tokenizer->finalized)
    return tk_lua_error(L, "already finalized");
  tk_lua_fchecktype(L, 1, "train", "corpus", LUA_TTABLE);
  lua_getfield(L, 1, "corpus");
  lua_remove(L, 1);
  int n = lua_objlen(L, 1);
  for (int i = 1; i <= n; i ++) {
    lua_pushinteger(L, (int64_t) i);
    lua_gettable(L, -2);
    size_t len;
    char *doc = tb_tokenizer_normalize((char *) luaL_checklstring(L, -1, &len), &len, tokenizer->max_run);
    tb_tokenizer_populate_tokens(tokenizer, doc, len, true);
    free(doc);
    lua_pop(L, 1);
    tokenizer->ndocs ++;
  }
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
  double max_df = tk_lua_fcheckposdouble(L, 1, "create", "max_df");
  double min_df = tk_lua_fcheckposdouble(L, 1, "create", "min_df");
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
  if (min_df < 0 || max_df > 1 || max_df < min_df)
    luaL_error(L, "min_df and max_df must be an interval between 0 and 1");
  if (min_len == 0)
    luaL_error(L, "min_len must be greater than or equal to 1");
  if (max_len < min_len)
    luaL_error(L, "max_len must be greater than or equal to min_len");
  if (ngrams == 0)
    luaL_error(L, "ngrams must be greater than or equal to 1");
  if (!align)
    luaL_error(L, "alignment must be greater than 0 (use 128 for TM)");
  tb_tokenizer_t *tokenizer = lua_newuserdata(L, sizeof(tb_tokenizer_t));
  memset(tokenizer, 0, sizeof(tb_tokenizer_t));
  luaL_getmetatable(L, MT_TOKENIZER);
  lua_setmetatable(L, -2);
  kv_init(tokenizer->tokens);
  kv_init(tokenizer->tmp_token);
  kv_init(tokenizer->tmp_append);
  kv_init(tokenizer->tmp_skipgram);
  kv_init(tokenizer->window);
  tokenizer->ids = kh_init(ids);
  tokenizer->strs = kh_init(strs);
  tokenizer->dfs = kh_init(dfs);
  tokenizer->tmp_seen = kh_init(seen);
  tokenizer->next_id = 0;
  tokenizer->ndocs = 0;
  tokenizer->max_df = max_df;
  tokenizer->min_df = min_df;
  tokenizer->max_len = max_len;
  tokenizer->min_len = min_len;
  tokenizer->max_run = max_run;
  tokenizer->ngrams = ngrams;
  tokenizer->cgrams_min = cgrams_min;
  tokenizer->cgrams_max = cgrams_max;
  tokenizer->skips = skips;
  tokenizer->negations = negations;
  tokenizer->align = align;
  tokenizer->window_size = ngrams + (ngrams - 1) * skips;
  lua_newtable(L);
  lua_pushvalue(L, -2);
  tk_lua_register(L, tb_mt_fns, 1);
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
  tb_tokenizer_t *tokenizer = lua_newuserdata(L, sizeof(tb_tokenizer_t));
  memset(tokenizer, 0, sizeof(tb_tokenizer_t));
  luaL_getmetatable(L, MT_TOKENIZER);
  lua_setmetatable(L, -2);
  kv_init(tokenizer->tokens);
  kv_init(tokenizer->tmp_token);
  kv_init(tokenizer->tmp_append);
  kv_init(tokenizer->tmp_skipgram);
  kv_init(tokenizer->window);
  tokenizer->ids = kh_init(ids);
  tokenizer->strs = kh_init(strs);
  if (!tokenizer->finalized) {
    tokenizer->dfs = kh_init(dfs);
    tokenizer->tmp_seen = kh_init(seen);
  }
  tk_lua_fread(L, &tokenizer->finalized, sizeof(bool), 1, fh);
  tk_lua_fread(L, &tokenizer->max_df, sizeof(double), 1, fh);
  tk_lua_fread(L, &tokenizer->min_df, sizeof(double), 1, fh);
  tk_lua_fread(L, &tokenizer->max_len, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->min_len, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->max_run, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->ngrams, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->cgrams_min, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->cgrams_max, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->skips, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->negations, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->align, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->ndocs, sizeof(int), 1, fh);
  tk_lua_fread(L, &tokenizer->next_id, sizeof(int), 1, fh);
  khint_t nkeys;
  khint_t k;
  int absent;
  tk_lua_fread(L, (char *) &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    size_t len;
    tk_lua_fread(L, &len, sizeof(size_t), 1, fh);
    char tok[len + 1];
    tk_lua_fread(L, tok, len, 1, fh);
    tok[len] = 0;
    int id;
    tk_lua_fread(L, &id, sizeof(int), 1, fh);
    char *tokn = strdup(tok);
    k = kh_put(ids, tokenizer->ids, tokn, &absent);
    assert(absent);
    kh_value(tokenizer->ids, k) = id;
    k = kh_put(strs, tokenizer->strs, (uint32_t) id, &absent);
    assert(absent);
    kh_value(tokenizer->strs, k) = tokn;
  }
  if (!tokenizer->finalized) {
    tk_lua_fread(L, (char *) &nkeys, sizeof(khint_t), 1, fh);
    for (khint_t i = 0; i < nkeys; i ++) {
      int id;
      int df;
      tk_lua_fread(L, &id, sizeof(int), 1, fh);
      tk_lua_fread(L, &df, sizeof(int), 1, fh);
      k = kh_put(dfs, tokenizer->dfs, (uint32_t) id, &absent);
      assert(absent);
      kh_value(tokenizer->dfs, k) = df;
    }
  }
  tk_lua_fclose(L, fh);
  lua_newtable(L);
  lua_pushvalue(L, -2);
  tk_lua_register(L, tb_mt_fns, 1);
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
  luaL_newmetatable(L, MT_TOKENIZER);
  lua_pushcfunction(L, tb_tokenizer_gc);
  lua_setfield(L, -2, "__gc");
  lua_pop(L, 1);
  return 1;
}

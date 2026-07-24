#ifndef TK_ANN_H
#define TK_ANN_H

#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/pvec.h>
#include <santoku/rvec.h>
#include <santoku/cvec.h>
#include <santoku/cvec/ext.h>
#include <santoku/csr.h>
#include <santoku/mtx.h>
#include <santoku/learn/mathlibs.h>

#define TK_ANN_SUBSTR_BITS 16
#define TK_ANN_BUCKETS (1 << TK_ANN_SUBSTR_BITS)

#define TK_ANN_MT "tk_ann_flat_t"

typedef struct {
  int64_t *sorted_sids;
  int64_t *bucket_off;
  uint64_t N, m, features;
  const char *data;
  size_t bytes_per_vec;
  const float *codes;
  uint64_t n_dims;
} tk_ann_flat_t;

static inline tk_ann_flat_t *tk_ann_flat_peek (lua_State *L, int i)
{
  return (tk_ann_flat_t *) luaL_checkudata(L, i, TK_ANN_MT);
}

static inline void tk_ann_sign_pack (
  const float *codes, uint64_t n, uint64_t n_dims, uint64_t features,
  char *bits, size_t bpv
) {
  memset(bits, 0, n * bpv);
  for (uint64_t s = 0; s < n; s++) {
    const float *row = codes + s * n_dims;
    unsigned char *bvec = (unsigned char *) (bits + s * bpv);
    for (uint64_t i = 0; i < n_dims && i < features; i++)
      if (row[i] >= 0.0f) bvec[i >> 3] |= (unsigned char) (1u << (i & 7));
  }
}

static inline uint32_t tk_ann_flat_substring (
  const char *vec, uint64_t features, uint64_t ti
) {
  uint64_t bit_offset = ti * TK_ANN_SUBSTR_BITS;
  uint64_t byte_offset = bit_offset / 8;
  uint32_t h = 0;
  uint64_t remaining = features - bit_offset;
  uint64_t bits_to_copy = remaining < TK_ANN_SUBSTR_BITS ? remaining : TK_ANN_SUBSTR_BITS;
  uint64_t bytes_to_copy = (bits_to_copy + 7) / 8;
  memcpy(&h, vec + byte_offset, bytes_to_copy);
  if (bits_to_copy < 32)
    h &= (1u << bits_to_copy) - 1;
  return h;
}

// sorted_sids (size m*N) and bucket_off (size m*(TK_ANN_BUCKETS+1)) must be
// pre-allocated by the caller; this only fills them via a counting sort over data.
static inline void tk_ann_flat_build (
  tk_ann_flat_t *flat, const char *data, uint64_t N, uint64_t features
) {
  uint64_t m = (features + TK_ANN_SUBSTR_BITS - 1) / TK_ANN_SUBSTR_BITS;
  if (m == 0) m = 1;
  flat->N = N;
  flat->m = m;
  flat->features = features;
  flat->data = data;
  flat->bytes_per_vec = TK_CVEC_BITS_BYTES(features);

  uint64_t stride = (uint64_t)TK_ANN_BUCKETS + 1;

  for (uint64_t ti = 0; ti < m; ti++) {
    int64_t *off = flat->bucket_off + ti * stride;
    int64_t *sids = flat->sorted_sids + ti * N;

    memset(off, 0, stride * sizeof(int64_t));

    for (uint64_t i = 0; i < N; i++) {
      uint32_t h = tk_ann_flat_substring(data + i * flat->bytes_per_vec, features, ti);
      off[h + 1]++;
    }
    for (uint64_t h = 1; h < stride; h++)
      off[h] += off[h - 1];

    for (uint64_t i = 0; i < N; i++) {
      uint32_t h = tk_ann_flat_substring(data + i * flat->bytes_per_vec, features, ti);
      sids[off[h]++] = (int64_t)i;
    }
    memmove(off + 1, off, (uint64_t)TK_ANN_BUCKETS * sizeof(int64_t));
    off[0] = 0;
  }
}

static inline void tk_ann_flat_probe (
  tk_ann_flat_t *flat,
  uint64_t ti,
  uint32_t h,
  int r,
  int64_t skip_sid,
  uint8_t *seen,
  const unsigned char *query,
  uint64_t k,
  tk_pvec_t *out
) {
  uint64_t features = flat->features;
  uint64_t sub_bits = (ti < flat->m - 1) ? TK_ANN_SUBSTR_BITS :
    (features - ti * TK_ANN_SUBSTR_BITS);
  int nbits = (int)sub_bits;
  if (r > nbits)
    return;
  uint64_t stride = (uint64_t)TK_ANN_BUCKETS + 1;
  const int64_t *bucket_off = flat->bucket_off + ti * stride;
  const int64_t *sorted_sids = flat->sorted_sids + ti * flat->N;
  const char *data = flat->data;
  size_t bpv = flat->bytes_per_vec;

  int pos[TK_ANN_SUBSTR_BITS];
  for (int i = 0; i < r; i++)
    pos[i] = i;

  while (true) {
    uint32_t mask = 0;
    for (int i = 0; i < r; i++)
      mask |= (1U << pos[i]);
    uint32_t probe_h = h ^ mask;
    int64_t lo = bucket_off[probe_h];
    int64_t hi = bucket_off[probe_h + 1];
    for (int64_t bi = lo; bi < hi; bi++) {
      int64_t sid = sorted_sids[bi];
      if (sid == skip_sid)
        continue;
      uint64_t usid = (uint64_t)sid;
      if (seen[usid >> 3] & (1u << (usid & 7)))
        continue;
      seen[usid >> 3] |= (uint8_t)(1u << (usid & 7));
      const unsigned char *vec = (const unsigned char *)(data + usid * bpv);
      uint64_t dist = tk_cvec_bits_hamming_serial(query, vec, features);
      tk_pvec_hmax(out, k, tk_pair(sid, (int64_t)dist));
    }

    if (r == 0)
      break;
    int j;
    for (j = r - 1; j >= 0; j--) {
      if (pos[j] != j + nbits - r) {
        pos[j]++;
        for (int l = j + 1; l < r; l++)
          pos[l] = pos[l - 1] + 1;
        break;
      }
    }
    if (j < 0)
      break;
  }
}

static inline void tk_ann_flat_query (
  tk_ann_flat_t *flat,
  const char *query_vec,
  int64_t skip_sid,
  uint64_t k,
  uint64_t max_radius,
  uint8_t *seen,
  tk_pvec_t *out
) {
  tk_pvec_clear(out);
  uint64_t seen_bytes = (flat->N + 7) / 8;
  memset(seen, 0, seen_bytes);

  const unsigned char *q = (const unsigned char *)query_vec;
  uint32_t hs[flat->m];
  for (uint64_t ti = 0; ti < flat->m; ti++)
    hs[ti] = tk_ann_flat_substring(query_vec, flat->features, ti);

  for (int r = 0; r <= (int)max_radius; r++) {
    for (uint64_t ti = 0; ti < flat->m; ti++)
      tk_ann_flat_probe(flat, ti, hs[ti], r, skip_sid, seen, q, k, out);
    if (out->n >= k && out->a[0].p < (int64_t)(flat->m * ((uint64_t)r + 1)))
      break;
  }
  tk_pvec_asc(out, 0, out->n);
}

static inline int tk_ann_flat_gc (lua_State *L)
{
  // sorted_sids / bucket_off / data / codes are all backed by vec (or mtx)
  // userdata anchored in this flat's fenv; they free themselves. Just detach.
  tk_ann_flat_t *flat = tk_ann_flat_peek(L, 1);
  flat->sorted_sids = NULL;
  flat->bucket_off = NULL;
  flat->data = NULL;
  flat->codes = NULL;
  return 0;
}

static inline void tk_ann_flat_query_csr (
  lua_State *L,
  tk_ann_flat_t *flat,
  const char *query_data,
  uint64_t nq,
  bool skip_self,
  uint64_t k,
  uint64_t max_radius,
  const float *query_codes,
  const float *corpus_codes,
  uint64_t n_dims,
  uint64_t oversample
) {
  uint64_t features = flat->features;
  bool rerank = (query_codes && corpus_codes && n_dims > 0);
  uint64_t fetch_k = rerank ? k * (oversample > 0 ? oversample : 1) : k;

  tk_ivec_t *off = tk_ivec_create(L, nq + 1);
  off->n = nq + 1;
  int off_idx = lua_gettop(L);
  tk_ivec_t *nbr = tk_ivec_create(L, nq * k);
  nbr->n = nq * k;
  int nbr_idx = lua_gettop(L);
  tk_dvec_t *wt = tk_dvec_create(L, nq * k);
  wt->n = nq * k;
  int wt_idx = lua_gettop(L);

  int64_t *counts = tk_malloc(L, nq * sizeof(int64_t));

  #pragma omp parallel
  {
    tk_pvec_t *heap = tk_pvec_create(NULL, fetch_k);
    tk_rvec_t *rerank_buf = rerank ? tk_rvec_create(NULL, fetch_k) : NULL;
    uint64_t seen_bytes = (flat->N + 7) / 8;
    uint8_t *seen = (uint8_t *)calloc(1, seen_bytes);

    #pragma omp for schedule(guided) nowait
    for (uint64_t i = 0; i < nq; i++) {
      const char *vec = query_data + i * flat->bytes_per_vec;
      int64_t skip = skip_self ? (int64_t)i : -1;
      tk_ann_flat_query(flat, vec, skip, fetch_k, max_radius, seen, heap);
      int64_t base = (int64_t)(i * k);

      if (rerank) {
        const float *qrow = query_codes + i * n_dims;
        rerank_buf->n = 0;
        for (uint64_t j = 0; j < heap->n; j++) {
          int64_t cand = heap->a[j].i;
          double dot = (double)cblas_sdot((int)n_dims, qrow, 1,
            corpus_codes + (uint64_t)cand * n_dims, 1);
          tk_rvec_push(rerank_buf, tk_rank(cand, dot));
        }
        tk_rvec_desc(rerank_buf, 0, rerank_buf->n);
        uint64_t cnt = rerank_buf->n < k ? rerank_buf->n : k;
        counts[i] = (int64_t)cnt;
        for (uint64_t j = 0; j < cnt; j++) {
          nbr->a[base + (int64_t)j] = rerank_buf->a[j].i;
          wt->a[base + (int64_t)j] = rerank_buf->a[j].d;
        }
      } else {
        uint64_t cnt = heap->n < k ? heap->n : k;
        counts[i] = (int64_t)cnt;
        for (uint64_t j = 0; j < cnt; j++) {
          nbr->a[base + (int64_t)j] = heap->a[j].i;
          wt->a[base + (int64_t)j] = 1.0 - (double)heap->a[j].p / (double)features;
        }
      }
    }

    free(seen);
    tk_pvec_destroy(heap);
    if (rerank_buf) tk_rvec_destroy(rerank_buf);
  }

  bool need_compact = false;
  off->a[0] = 0;
  for (uint64_t i = 0; i < nq; i++) {
    if (counts[i] < (int64_t)k)
      need_compact = true;
    off->a[i + 1] = off->a[i] + counts[i];
  }

  if (need_compact) {
    int64_t total = off->a[nq];
    int64_t write = 0;
    for (uint64_t i = 0; i < nq; i++) {
      int64_t src = (int64_t)(i * k);
      int64_t cnt = counts[i];
      if (write != src) {
        memmove(nbr->a + write, nbr->a + src, (uint64_t)cnt * sizeof(int64_t));
        memmove(wt->a + write, wt->a + src, (uint64_t)cnt * sizeof(double));
      }
      write += cnt;
    }
    nbr->n = (uint64_t)total;
    wt->n = (uint64_t)total;
  }

  free(counts);

  tk_csr_push(L, TK_TAG_F64, TK_TAG_I64, flat->N,
    off_idx, off, nbr_idx, (void *) nbr, wt_idx, wt);
}

static inline int tk_ann_flat_nbr_by_vecs_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ann_flat_t *flat = tk_ann_flat_peek(L, 1);
  tk_mtx_t *Q = tk_mtx_peek(L, 2, "query");
  uint64_t k = tk_lua_checkunsigned(L, 3, "k");
  uint64_t max_radius = tk_lua_optunsigned(L, 4, "radius", 3);
  uint64_t oversample = tk_lua_optunsigned(L, 5, "oversample", 16);
  uint64_t nq = Q->n_rows;
  uint64_t n_dims = Q->n_cols;
  const float *qcodes = ((tk_fvec_t *) Q->v)->a;
  size_t bpv = flat->bytes_per_vec;
  tk_cvec_t *qbits = tk_cvec_create(L, nq * bpv);
  qbits->n = nq * bpv;
  tk_ann_sign_pack(qcodes, nq, n_dims, flat->features, qbits->a, bpv);
  const float *corpus_codes = flat->codes;
  const float *query_codes = (corpus_codes && flat->n_dims == n_dims) ? qcodes : NULL;
  tk_ann_flat_query_csr(L, flat, qbits->a, nq, false, k, max_radius,
    query_codes, corpus_codes, flat->n_dims, oversample);
  return 1;
}

static inline int tk_ann_flat_nbr_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ann_flat_t *flat = tk_ann_flat_peek(L, 1);
  uint64_t k = tk_lua_checkunsigned(L, 2, "k");
  bool do_rerank = flat->codes != NULL;
  if (lua_isboolean(L, 3))
    do_rerank = lua_toboolean(L, 3);
  int ai = lua_isboolean(L, 3) ? 4 : 3;
  uint64_t max_radius = tk_lua_optunsigned(L, ai, "radius", 3);
  uint64_t oversample = tk_lua_optunsigned(L, ai + 1, "oversample", 16);
  const float *corpus = do_rerank ? flat->codes : NULL;
  uint64_t n_dims = do_rerank ? flat->n_dims : 0;
  tk_ann_flat_query_csr(L, flat, flat->data, flat->N, true, k, max_radius,
    corpus, corpus, n_dims, oversample);
  return 1;
}

// Manifest at `path` (dims + a per-member "external" flag). Each big member (bits/sids/buckets)
// is EMBEDDED inline here if it is a RAM vec, or left in its own caller-owned mmap file (flag =
// external, msync only) if it was mmap-backed -- exactly like ridge's W. Rerank codes are always
// external (passed back at load). No ann-written sidecars: the mid layer owns any mmap paths.
static inline int tk_ann_persist_lua (lua_State *L)
{
  tk_ann_flat_t *flat = tk_ann_flat_peek(L, 1);
  const char *path = luaL_checkstring(L, 2);
  uint64_t stride = (uint64_t)TK_ANN_BUCKETS + 1;
  uint64_t sids_bytes = flat->m * flat->N * sizeof(int64_t);
  uint64_t buckets_bytes = flat->m * stride * sizeof(int64_t);
  uint64_t bits_bytes = flat->N * flat->bytes_per_vec;

  lua_getfenv(L, 1);
  int eph = lua_gettop(L);

  FILE *fh = tk_lua_fopen(L, path, "w");
  tk_lua_fwrite(L, "TKan", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  int64_t N = (int64_t) flat->N, m = (int64_t) flat->m;
  int64_t features = (int64_t) flat->features, bpv = (int64_t) flat->bytes_per_vec;
  tk_lua_fwrite(L, &N, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &m, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &features, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &bpv, sizeof(int64_t), 1, fh);

  lua_getfield(L, eph, "data");
  tk_cvec_t *bv = tk_cvec_peek(L, -1, "data");
  uint8_t ext = (bv->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &ext, sizeof(uint8_t), 1, fh);
#if !defined(__EMSCRIPTEN__)
  if (ext) msync(bv->a, bits_bytes, MS_SYNC); else
#endif
  tk_lua_fwrite(L, bv->a, 1, bits_bytes, fh);
  lua_pop(L, 1);

  lua_getfield(L, eph, "sids");
  tk_ivec_t *sv = tk_ivec_peek(L, -1, "sids");
  ext = (sv->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &ext, sizeof(uint8_t), 1, fh);
#if !defined(__EMSCRIPTEN__)
  if (ext) msync(sv->a, sids_bytes, MS_SYNC); else
#endif
  tk_lua_fwrite(L, (char *) sv->a, 1, sids_bytes, fh);
  lua_pop(L, 1);

  lua_getfield(L, eph, "buckets");
  tk_ivec_t *kv = tk_ivec_peek(L, -1, "buckets");
  ext = (kv->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &ext, sizeof(uint8_t), 1, fh);
#if !defined(__EMSCRIPTEN__)
  if (ext) msync(kv->a, buckets_bytes, MS_SYNC); else
#endif
  tk_lua_fwrite(L, (char *) kv->a, 1, buckets_bytes, fh);
  lua_pop(L, 1);

  tk_lua_fclose(L, fh);
  return 0;
}

static luaL_Reg tk_ann_flat_mt_fns[] =
{
  { "neighborhoods_by_vecs", tk_ann_flat_nbr_by_vecs_lua },
  { "neighborhoods", tk_ann_flat_nbr_lua },
  { "persist", tk_ann_persist_lua },
  { NULL, NULL }
};

// ann.load(path, [codes_mtx], [sids], [buckets], [bits])
//   codes_mtx: external float codes for exact-dot rerank (nil => Hamming-only)
//   sids/buckets/bits: for members persist wrote as EXTERNAL, pass the caller-mmap_open'd vec
//     here (required); for members it EMBEDDED, leave nil (read from the main file into RAM).
//     ann never mmaps a path itself -- the mid layer owns that (same convention as ridge's W).
static inline int tk_ann_load_lua (lua_State *L)
{
  const char *path = luaL_checkstring(L, 1);
  lua_settop(L, 5);  // fix arg count so optional sids/buckets/bits (3/4/5) stay nil-or-value
                     // even after we push the flat userdata (else absent args read as the flat)
  tk_mtx_t *Mc = lua_isnoneornil(L, 2) ? NULL : tk_mtx_peek(L, 2, "codes");
  int codes_idx = Mc ? 2 : 0;

  FILE *fh = tk_lua_fopen(L, path, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKan", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ann file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ann version %d", (int) version);
  }
  int64_t N, m, features, bpv;
  tk_lua_fread(L, &N, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &m, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &features, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &bpv, sizeof(int64_t), 1, fh);
  // embedded members (if any) follow inline -- keep fh open until they're read.

  uint64_t stride = (uint64_t)TK_ANN_BUCKETS + 1;
  uint64_t sids_n = (uint64_t)(m * N);
  uint64_t buckets_n = (uint64_t) m * stride;
  uint64_t bits_n = (uint64_t)(N * bpv);

  tk_ann_flat_t *flat = tk_lua_newuserdata(L, tk_ann_flat_t, TK_ANN_MT, tk_ann_flat_mt_fns, tk_ann_flat_gc);
  int flat_idx = lua_gettop(L);
  flat->N = (uint64_t) N;
  flat->m = (uint64_t) m;
  flat->features = (uint64_t) features;
  flat->bytes_per_vec = (size_t) bpv;
  flat->sorted_sids = NULL;
  flat->bucket_off = NULL;
  flat->data = NULL;
  flat->codes = NULL;
  flat->n_dims = 0;

  // Per member, read the "external" flag persist wrote: external => data lives in the caller's
  // mmap file, which must be passed in (arg 5/3/4); embedded => read the bytes from this file
  // into a RAM vec (and no buffer may be passed). File order matches persist: bits, sids, buckets.
  int bits_idx, sids_idx, buckets_idx;
  uint8_t ext;

  tk_lua_fread(L, &ext, sizeof(uint8_t), 1, fh);
  tk_cvec_t *bits;
  if (ext) {
    if (lua_isnoneornil(L, 5)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: bits is external; pass its mmap as arg 5"); }
    bits = tk_cvec_peek(L, 5, "bits"); bits_idx = 5;
  } else {
    if (!lua_isnoneornil(L, 5)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: bits is embedded; do not pass arg 5"); }
    bits = tk_cvec_create(L, bits_n); bits->n = bits_n; bits_idx = lua_gettop(L);
    tk_lua_fread(L, bits->a, 1, bits_n, fh);
  }

  tk_lua_fread(L, &ext, sizeof(uint8_t), 1, fh);
  tk_ivec_t *sids;
  if (ext) {
    if (lua_isnoneornil(L, 3)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: sids is external; pass its mmap as arg 3"); }
    sids = tk_ivec_peek(L, 3, "sids"); sids_idx = 3;
  } else {
    if (!lua_isnoneornil(L, 3)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: sids is embedded; do not pass arg 3"); }
    sids = tk_ivec_create(L, sids_n); sids->n = sids_n; sids_idx = lua_gettop(L);
    tk_lua_fread(L, sids->a, sizeof(int64_t), sids_n, fh);
  }

  tk_lua_fread(L, &ext, sizeof(uint8_t), 1, fh);
  tk_ivec_t *buckets;
  if (ext) {
    if (lua_isnoneornil(L, 4)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: buckets is external; pass its mmap as arg 4"); }
    buckets = tk_ivec_peek(L, 4, "buckets"); buckets_idx = 4;
  } else {
    if (!lua_isnoneornil(L, 4)) { tk_lua_fclose(L, fh); return luaL_error(L, "ann load: buckets is embedded; do not pass arg 4"); }
    buckets = tk_ivec_create(L, buckets_n); buckets->n = buckets_n; buckets_idx = lua_gettop(L);
    tk_lua_fread(L, buckets->a, sizeof(int64_t), buckets_n, fh);
  }
  tk_lua_fclose(L, fh);

  if (sids->n != sids_n || buckets->n != buckets_n || bits->n != bits_n)
    return luaL_error(L, "ann load: index array size mismatch (corrupt, wrong dims, or wrong buffer)");
  flat->sorted_sids = sids->a;
  flat->bucket_off = buckets->a;
  flat->data = bits->a;

  if (Mc) {
    flat->codes = ((tk_fvec_t *) Mc->v)->a;
    flat->n_dims = Mc->n_cols;
  }

  lua_newtable(L);
  int eph = lua_gettop(L);
  lua_pushvalue(L, sids_idx);
  lua_setfield(L, eph, "sids");
  lua_pushvalue(L, buckets_idx);
  lua_setfield(L, eph, "buckets");
  lua_pushvalue(L, bits_idx);
  lua_setfield(L, eph, "data");
  if (codes_idx) {
    lua_pushvalue(L, codes_idx);
    lua_setfield(L, eph, "codes");
  }
  lua_pushvalue(L, eph);
  lua_setfenv(L, flat_idx);
  lua_settop(L, flat_idx);
  return 1;
}

#endif

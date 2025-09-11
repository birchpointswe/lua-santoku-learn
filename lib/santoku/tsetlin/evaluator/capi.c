#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/cluster.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>

#define TK_EVAL_EPH "tk_eval_eph"

typedef enum {
  TK_EVAL_CLASS_ACCURACY,
  TK_EVAL_CLUSTERING_ACCURACY,
  TK_EVAL_OPTIMIZE_CLUSTERING,
  TK_EVAL_ENCODING_ACCURACY,
  TK_EVAL_ENCODING_SIMILARITY,
  TK_EVAL_ENCODING_AUC,
  TK_EVAL_GRAPH_RECONSTRUCTION,
} tk_eval_stage_t;

typedef struct {
  tm_dl_t *pl;
  atomic_ulong *ERR, *TP, *FP, *FN, *hist_pos, *hist_neg;
  atomic_ulong *valid_pos, *valid_neg;
  tk_ivec_t *ids;
  tk_iumap_t *ididx;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  uint64_t min_pts;
  bool assign_noise;
  uint64_t min_margin, max_margin;
  tk_pvec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  double *f1, *precision, *recall;
  unsigned int *predicted, *expected;
  tk_bits_t *codes, *codes_predicted, *codes_expected, *mask;
  uint64_t mask_popcount;
  double *dcodes;
  tk_iumap_t *id_code, *id_assignment;
  unsigned int n_visible, n_dims, chunks;
  tk_ivec_t *assignments;
  tk_graph_t *graph;
  tk_ivec_t *offsets;
  tk_ivec_t *neighbors;
  tk_dvec_t *weights;
} tk_eval_t;

typedef struct {
  double tpr;
  double tnr;
  double bacc;
} tk_accuracy_t;

typedef struct {
  tk_thread_t *self;
  tk_eval_t *state;
  uint64_t diff;
  uint64_t *bdiff;
  uint64_t sfirst, slast;
  uint64_t pfirst, plast;
  uint64_t dfirst, dlast;
  uint64_t mfirst, mlast;
  tk_accuracy_t best, next;
  uint64_t next_n;
  uint64_t best_n;
  uint64_t next_m;
  uint64_t best_m;
  tk_ivec_t *next_assignments;
  tk_ivec_t *best_assignments;
  tk_rvec_t *rtmp;
  tk_pvec_t *ptmp;
  double recon_error;
} tk_eval_thread_t;

static inline tk_accuracy_t _tm_clustering_accuracy (
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg
);

static void tk_eval_worker (void *dp, int sig)
{
  tk_eval_thread_t *data = (tk_eval_thread_t *) dp;
  tk_eval_t *state = data->state;
  khint_t khi;
  switch ((tk_eval_stage_t) sig) {

    case TK_EVAL_CLASS_ACCURACY:
      for (unsigned int i = data->sfirst; i <= data->slast; i ++) {
        unsigned int y_pred = state->predicted[i];
        unsigned int y_true = state->expected[i];
        if (y_pred >= state->n_dims || y_true >= state->n_dims)
          continue;
        if (y_pred == y_true)
          atomic_fetch_add(state->TP + y_true, 1);
        else {
          atomic_fetch_add(state->FP + y_pred, 1);
          atomic_fetch_add(state->FN + y_true, 1);
        }
      }
      break;

    case TK_EVAL_OPTIMIZE_CLUSTERING:
      for (uint64_t m0 = data->mfirst; m0 <= data->mlast; m0 ++) {
        uint64_t m = m0 + state->min_margin;
        data->next_m = m;
        tk_ivec_clear(data->next_assignments);
        tk_cluster_dsu(state->hbi, state->ann, data->rtmp, data->ptmp, m, state->min_pts, state->assign_noise, state->ids, data->next_assignments, state->ididx, &data->next_n);
        tk_accuracy_t result = _tm_clustering_accuracy(state->ids, data->next_assignments, state->pos, state->neg);
        data->next = result;
        tk_threads_notify_parent(data->self);
        if (result.bacc > data->best.bacc) {
          data->best = result;
          data->best_n = data->next_n;
          data->best_m = m;
          tk_ivec_t *tmp = data->best_assignments;
          data->best_assignments = data->next_assignments;
          data->next_assignments = tmp;
        }
      }
      break;

    case TK_EVAL_CLUSTERING_ACCURACY:
      for (unsigned int i = data->pfirst; i <= data->plast; i ++) {
        tk_pvec_t *pairs = i < state->n_pos ? state->pos : state->neg;
        uint64_t offset = i < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(i - offset)].i;
        int64_t v = pairs->a[(i - offset)].p;
        khi = tk_iumap_get(state->id_assignment, u);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iu = tk_iumap_value(state->id_assignment, khi);
        khi = tk_iumap_get(state->id_assignment, v);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iv = tk_iumap_value(state->id_assignment, khi);
        int64_t cu = state->assignments->a[iu];
        int64_t cv = state->assignments->a[iv];
        if (i < state->n_pos) {
          atomic_fetch_add(state->valid_pos, 1);
          if (cu == cv)
            atomic_fetch_add(state->TP, 1);
          else
            atomic_fetch_add(state->FN, 1);
        } else {
          atomic_fetch_add(state->valid_neg, 1);
          if (cu == cv)
            atomic_fetch_add(state->FP, 1);
        }
      }
      break;

    case TK_EVAL_ENCODING_ACCURACY:
      for (uint64_t i = data->sfirst; i <= data->slast; i ++) {
        for (uint64_t j = 0; j < state->n_dims; j ++) {
          uint64_t word = TK_CVEC_BITS_BYTE(j);
          uint64_t bit = TK_CVEC_BITS_BIT(j);
          bool y =
            (state->codes_expected[i * state->chunks + word] & ((tk_bits_t)1 << bit)) ==
            (state->codes_predicted[i * state->chunks + word] & ((tk_bits_t)1 << bit));
          if (y)
            continue;
          data->diff ++;
          data->bdiff[j] ++;
        }
      }
      break;

    case TK_EVAL_ENCODING_SIMILARITY:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        if (state->pl[k].sim == -1)
          continue;
        int64_t s = state->n_dims - state->pl[k].sim;
        if (s < 0) s = 0;
        if (s > state->n_dims) s = state->n_dims;
        if (state->pl[k].label)
          atomic_fetch_add(state->hist_pos + s, 1);
        else
          atomic_fetch_add(state->hist_neg + s, 1);
      }
      break;

    case TK_EVAL_ENCODING_AUC:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        tk_pvec_t *pairs = k < state->n_pos ? state->pos : state->neg;
        uint64_t offset = k < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(k - offset)].i;
        int64_t v = pairs->a[(k - offset)].p;
        int64_t iu, iv;
        if (state->id_code == NULL) {
          iu = u;
          iv = v;
        } else {
          khi = tk_iumap_get(state->id_code, u);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iu = tk_iumap_value(state->id_code, khi);
          if (iu == -1)
            continue;
          khi = tk_iumap_get(state->id_code, v);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iv = tk_iumap_value(state->id_code, khi);
          if (iv == -1)
            continue;
        }
        if (state->dcodes != NULL) {
          // continuous embedding: dot-product
          double *X = state->dcodes;
          uint64_t K = state->n_dims;
          double dot = 0.0;
          double *xi = X + (uint64_t)iu * K;
          double *xj = X + (uint64_t)iv * K;
          for (uint64_t d = 0; d < K; d ++)
            dot += xi[d] * xj[d];
          // scale to retain precision in int64
          state->pl[k].sim = (int64_t)(dot * 1e3);
        } else if (state->codes != NULL) {
          if (state->mask != NULL) {
            state->pl[k].sim = (int64_t) (state->mask_popcount - tk_cvec_bits_hamming_mask(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, (const unsigned char *) state->mask, state->n_dims));
          } else {
            state->pl[k].sim = (int64_t) (state->n_dims - tk_cvec_bits_hamming(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, state->n_dims));
          }
        } else {
          state->pl[k].sim = -1;
        }
        state->pl[k].label = k < state->n_pos ? 1 : 0;
      }
      break;

    case TK_EVAL_GRAPH_RECONSTRUCTION:
      data->recon_error = 0.0;
      for (uint64_t i = data->sfirst; i <= data->slast; i++) {
        int64_t start = state->offsets->a[i];
        int64_t end = state->offsets->a[i + 1];
        for (int64_t j = start; j < end; j++) {
          int64_t neighbor = state->neighbors->a[j];
          double weight = state->weights->a[j];
          uint64_t hamming_dist = tk_cvec_bits_hamming_mask(
            (const unsigned char *) state->codes + i * state->chunks,
            (const unsigned char *) state->codes + neighbor * state->chunks,
            (const unsigned char *) state->mask,
            state->n_dims);
          double hamming_sim = 1.0 - ((double) hamming_dist / state->mask_popcount);
          double target_sim = (weight > 0) ? weight : 0.0;
          double error = fabs(weight) * (target_sim - hamming_sim) * (target_sim - hamming_sim);
          data->recon_error += error;
        }
      }
      break;

    default:
      assert(false);

  }
}

static inline int tm_class_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  unsigned int *predicted, *expected;
  tk_cvec_t *pvec = tk_cvec_peekopt(L, 1);
  tk_cvec_t *evec = tk_cvec_peekopt(L, 2);
  predicted = pvec != NULL ? (unsigned int *) pvec->a : (unsigned int *) tk_lua_checkustring(L, 1, "predicted");
  expected = evec != NULL ? (unsigned int *) evec->a : (unsigned int *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_classes");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  if (n_dims == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  tk_eval_t state;
  state.n_dims = n_dims;
  state.expected = expected;
  state.predicted = predicted;
  state.TP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FN = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.precision = tk_malloc(L, n_dims * sizeof(double));
  state.recall = tk_malloc(L, n_dims * sizeof(double));
  state.f1 = tk_malloc(L, n_dims * sizeof(double));
  for (uint64_t i = 0; i < n_dims; i ++) {
    atomic_init(state.TP + i, 0);
    atomic_init(state.FP + i, 0);
    atomic_init(state.FN + i, 0);
  }

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_CLASS_ACCURACY, 0);
  tk_threads_destroy(pool);

  // Reduce
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_dims; c ++) {
    uint64_t tp = state.TP[c], fp = state.FP[c], fn = state.FN[c];
    state.precision[c] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    state.recall[c] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    state.f1[c] = (state.precision[c] + state.recall[c]) > 0 ?
      2.0 * state.precision[c] * state.recall[c] / (state.precision[c] + state.recall[c]) : 0.0;
    precision_avg += state.precision[c];
    recall_avg += state.recall[c];
    f1_avg += state.f1[c];
  }

  // Cleanup
  free(state.TP);
  free(state.FP);
  free(state.FN);

  // Lua output
  precision_avg /= n_dims;
  recall_avg /= n_dims;
  f1_avg /= n_dims;
  lua_newtable(L);
  lua_newtable(L);
  for (uint64_t c = 0; c < n_dims; c ++) {
    lua_pushinteger(L, (int64_t) c + 1);
    lua_newtable(L);
    lua_pushnumber(L, state.precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, state.recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, state.f1[c]);
    lua_setfield(L, -2, "f1");
    lua_settable(L, -3);
  }
  lua_setfield(L, -2, "classes");
  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  // Cleanup
  free(state.precision);
  free(state.recall);
  free(state.f1);

  return 1;
}

static inline tk_accuracy_t _tm_clustering_accuracy (
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg
) {
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;
  if (n_pos + n_neg == 0)
    return (tk_accuracy_t) { 0 };

  tk_eval_t state;
  atomic_ulong TP, FP, FN, valid_pos, valid_neg;
  state.assignments = assignments;
  state.id_assignment = tk_iumap_from_ivec(ids);
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.TP = &TP;
  state.FP = &FP;
  state.FN = &FN;
  state.valid_pos = &valid_pos;
  state.valid_neg = &valid_neg;
  atomic_init(&TP, 0);
  atomic_init(&FP, 0);
  atomic_init(&FN, 0);
  atomic_init(&valid_pos, 0);
  atomic_init(&valid_neg, 0);

  tk_eval_thread_t data;
  data.state = &state;
  data.pfirst = 0;
  data.plast = n_pos + n_neg - 1;

  // Run eval directly (no threading)
  tk_eval_worker(&data, TK_EVAL_CLUSTERING_ACCURACY);

  uint64_t vpos = atomic_load(state.valid_pos);
  uint64_t vneg = atomic_load(state.valid_neg);
  uint64_t tp = atomic_load(state.TP);
  uint64_t fp = atomic_load(state.FP);
  uint64_t tn = vneg > fp ? (vneg - fp) : 0;
  double tpr = vpos > 0 ? (double) tp / vpos : 0.0;
  double tnr = vneg > 0 ? (double) tn / vneg : 0.0;
  double bacc = 0.5 * (tpr + tnr); // true balanced accuracy
  tk_accuracy_t acc;
  acc.tpr = tpr;
  acc.tnr = tnr;
  acc.bacc = bacc;

  tk_iumap_destroy(state.id_assignment);
  return acc;
}

static inline tk_accuracy_t tm_clustering_accuracy (
  lua_State *L,
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  unsigned int n_threads
) {
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  atomic_ulong TP, FP, FN, valid_pos, valid_neg;
  state.assignments = assignments;
  state.id_assignment = tk_iumap_from_ivec(ids);
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.TP = &TP;
  state.FP = &FP;
  state.FN = &FN;
  state.valid_pos = &valid_pos;
  state.valid_neg = &valid_neg;
  atomic_init(&TP, 0);
  atomic_init(&FP, 0);
  atomic_init(&FN, 0);
  atomic_init(&valid_pos, 0);
  atomic_init(&valid_neg, 0);

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_CLUSTERING_ACCURACY, 0);
  tk_threads_destroy(pool);

  uint64_t vpos = atomic_load(state.valid_pos);
  uint64_t vneg = atomic_load(state.valid_neg);
  uint64_t tp = atomic_load(state.TP);
  uint64_t fp = atomic_load(state.FP);
  uint64_t tn = vneg > fp ? (vneg - fp) : 0;
  double tpr = vpos > 0 ? (double) tp / vpos : 0.0;
  double tnr = vneg > 0 ? (double) tn / vneg : 0.0;
  double bacc = 0.5 * (tpr + tnr); // true balanced accuracy
  tk_accuracy_t acc;
  acc.tpr = tpr;
  acc.tnr = tnr;
  acc.bacc = bacc;

  tk_iumap_destroy(state.id_assignment);
  return acc;
}

static inline int tm_clustering_accuracy_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ivec_t *assignments = tk_ivec_peek(L, 1, "assignments");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  tk_accuracy_t acc = tm_clustering_accuracy(L, ids, assignments, pos, neg, n_threads);

  // Lua output
  lua_newtable(L);
  lua_pushnumber(L, acc.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, acc.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, acc.tnr);
  lua_setfield(L, -2, "tnr");
  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_bits_t *codes_predicted, *codes_expected;
  tk_cvec_t *pvec = tk_cvec_peekopt(L, 1);
  tk_cvec_t *evec = tk_cvec_peekopt(L, 2);
  codes_predicted = pvec != NULL ? (tk_bits_t *) pvec->a : (tk_bits_t *) tk_lua_checkustring(L, 1, "predicted");
  codes_expected = evec != NULL ? (tk_bits_t *) evec->a : (tk_bits_t *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);
  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = chunks;
  state.codes_expected = codes_expected;
  state.codes_predicted = codes_predicted;

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    data[i].diff = 0;
    data[i].bdiff = tk_malloc(L, n_dims * sizeof(uint64_t));
    memset(data[i].bdiff, 0, sizeof(uint64_t) * n_dims);
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_ENCODING_ACCURACY, 0);
  tk_threads_destroy(pool);

  // Reduce
  uint64_t diff_total = 0;
  uint64_t *bdiff_total = tk_malloc(L, n_dims * sizeof(uint64_t));
  memset(bdiff_total, 0, n_dims * sizeof(uint64_t));
  for (unsigned int i = 0; i < n_threads; i ++) {
    diff_total += data[i].diff;
    for (uint64_t j = 0; j < n_dims; j ++)
      bdiff_total[j] += data[i].bdiff[j];
  }

  // Lua output
  lua_newtable(L);
  lua_newtable(L); // dims
  double min_bdiff = 1.0, max_bdiff = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double t = (double) bdiff_total[j] / (double) n_samples;
    if (t < min_bdiff) min_bdiff = t;
    if (t > max_bdiff) max_bdiff = t;
    lua_pushinteger(L, (int64_t) j + 1);
    lua_pushnumber(L, t);
    lua_settable(L, -3);
  }

  double mean_bdiff = (double) diff_total / (n_samples * n_dims);
  double var = 0.0;
  for (uint64_t j = 0; j < n_dims; ++j) {
    double ber = (double) bdiff_total[j] / n_samples;
    var += (ber - mean_bdiff) * (ber - mean_bdiff);
  }
  double std_bdiff = n_dims > 1 ? sqrt(var / (n_dims - 1)) : 0.0;

  lua_setfield(L, -2, "bits");
  lua_pushnumber(L, 1.0 - mean_bdiff);
  lua_setfield(L, -2, "mean_hamming");
  lua_pushnumber(L, 1.0 - min_bdiff);
  lua_setfield(L, -2, "ber_min");
  lua_pushnumber(L, 1.0 - max_bdiff);
  lua_setfield(L, -2, "ber_max");
  lua_pushnumber(L, std_bdiff);
  lua_setfield(L, -2, "ber_std");

  free(bdiff_total);
  for (unsigned int i = 0; i < n_threads; i ++)
    free(data[i].bdiff);

  return 1;
}

static inline double _tm_auc (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool
) {
  // Return 0.0 for empty mask (no features selected)
  if (state->mask != NULL && state->mask_popcount == 0)
    return 0.0;

  tk_threads_signal(pool, TK_EVAL_ENCODING_AUC, 0);
  ks_introsort(dl, state->n_pos + state->n_neg, state->pl);
  double sum_ranks = 0.0;
  unsigned int rank = 1;
  uint64_t n_pos_valid = 0, n_neg_valid = 0;
  for (uint64_t k = 0; k < state->n_pos + state->n_neg; k ++) {
    if (state->pl[k].sim == -1)
      continue;
    if (state->pl[k].label)
      sum_ranks += rank, n_pos_valid ++;
    else
      n_neg_valid ++;
    rank ++;
  }
  if (n_pos_valid == 0 || n_neg_valid == 0)
    return 0.0;
  double auc = (sum_ranks - ((double) n_pos_valid * (n_pos_valid + 1) / 2)) / ((double) n_pos_valid * n_neg_valid);
  return auc;
}

static inline double _tm_graph_reconstruction (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool
) {
  if (state->mask != NULL && state->mask_popcount == 0)
    return -1e9;
  uint64_t n_nodes = state->offsets->n - 1;
  tk_eval_thread_t *data = (tk_eval_thread_t *)pool->threads[0].data;
  for (unsigned int i = 0; i < pool->n_threads; i++) {
    data[i].recon_error = 0.0;
    tk_thread_range(i, pool->n_threads, n_nodes, &data[i].sfirst, &data[i].slast);
  }
  tk_threads_signal(pool, TK_EVAL_GRAPH_RECONSTRUCTION, 0);
  double total_error = 0.0;
  for (unsigned int i = 0; i < pool->n_threads; i++)
    total_error += data[i].recon_error;
  double n_edges = state->weights->n;
  double avg_error = (n_edges > 0) ? total_error / n_edges : 0.0;
  return -avg_error;
}

static inline int tm_auc (lua_State *L)
{
  lua_settop(L, 7);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_dvec_t *dcodes = tk_dvec_peekopt(L, 2);
  tk_cvec_t *ccodes = dcodes == NULL ? tk_cvec_peekopt(L, 2) : NULL;
  tk_bits_t *codes = dcodes != NULL ? NULL : (ccodes != NULL ? (tk_bits_t *) ccodes->a : (tk_bits_t *) tk_lua_checkstring(L, 2, "codes"));
  if (!(dcodes != NULL || codes != NULL))
    tk_lua_verror(L, 3, "auc", "codes", "must be either a string, tk_cvec_t, or tk_dvec_t");

  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  uint64_t n_dims = tk_lua_checkunsigned(L, 5, "n_hidden");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_bits_t *mask;
  tk_cvec_t *mvec = tk_cvec_peekopt(L, 6);
  mask = mvec != NULL ? (tk_bits_t *) mvec->a :
    (lua_type(L, 6) == LUA_TSTRING ? (tk_bits_t *) luaL_checkstring(L, 6) :
     lua_type(L, 6) == LUA_TLIGHTUSERDATA ? (tk_bits_t *) lua_touserdata(L, 6) : NULL);

  unsigned int n_threads = tk_threads_getn(L, 7, "n_threads", NULL);

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = mask;
  state.id_code = tk_iumap_from_ivec(ids);
  state.codes = codes;
  state.dcodes = dcodes != NULL ? dcodes->a : NULL;
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  free(state.pl);
  tk_threads_destroy(pool);
  tk_iumap_destroy(state.id_code);

  lua_pushnumber(L, auc);
  return 1;
}

static inline int tm_optimize_clustering (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "index");
  int i_index = tk_lua_absindex(L, -1);
  tk_ann_t *ann = tk_ann_peekopt(L, i_index);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, i_index);

  if (ann == NULL && hbi == NULL)
    tk_lua_verror(L, 3, "optimize_clustering", "index", "either tk_ann_t or tk_hbi_t must be provided");

  uint64_t n_features = ann != NULL ? ann->features : hbi->features;

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  int i_ids = ids == NULL ? -1 : tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  uint64_t min_pts = tk_lua_foptunsigned(L, 1, "optimize clustering", "min_pts", 0);
  bool assign_noise = tk_lua_foptboolean(L, 1, "optimize clustering", "assign_noise", true);

  uint64_t min_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "min_margin", 0);
  uint64_t max_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "max_margin", n_features);
  uint64_t fs = ann != NULL ? ann->features : hbi != NULL ? hbi->features : 0;
  if (max_margin > fs) max_margin = fs;
  if (min_margin > fs) min_margin = fs;

  if (max_margin < min_margin)
    max_margin = min_margin;

  unsigned int n_threads = tk_threads_getn(L, 1, "optimize clustering", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_eval_t state;
  state.ann = ann;
  state.hbi = hbi;
  state.pos = pos;
  state.neg = neg;
  state.assign_noise = assign_noise;
  state.min_pts = min_pts;
  state.min_margin = min_margin;
  state.max_margin = max_margin;
  lua_newtable(L); // t -- to track gc for vectors
  int i_eph = tk_lua_absindex(L, -1);
  if (ids != NULL)
    lua_pushvalue(L, i_ids); // t ids
  state.ids =
    ids != NULL ? ids :
    hbi ? tk_iumap_keys(L, hbi->uid_sid) :
    ann ? tk_iumap_keys(L, ann->uid_sid) : tk_ivec_create(L, 0, 0, 0); // t ids
  state.ididx =
    tk_iumap_from_ivec(state.ids);

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].self = pool->threads + i;
    data[i].state = &state;
    data[i].best = (tk_accuracy_t) { .bacc = -1.0, .tpr = 0.0, .tnr = 0.0 };
    data[i].next_n = 0;
    data[i].best_n = 0;
    data[i].best_m = 0;
    data[i].next_assignments = tk_ivec_create(L, state.ids->n, 0, 0);
    data[i].best_assignments = tk_ivec_create(L, state.ids->n, 0, 0);
    data[i].ptmp = tk_pvec_create(L, 0, 0, 0);
    data[i].rtmp = tk_rvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -4);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -3);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
    lua_pop(L, 4);
    tk_thread_range(i, n_threads, max_margin - min_margin + 1, &data[i].mfirst, &data[i].mlast);
  }

  unsigned int child;
  while (tk_threads_signal(pool, TK_EVAL_OPTIMIZE_CLUSTERING, &child)) {
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, data[child].next.bacc);
      lua_pushnumber(L, data[child].next.tpr);
      lua_pushnumber(L, data[child].next.tnr);
      lua_pushinteger(L, (lua_Integer) data[child].next_m);
      lua_pushinteger(L, (lua_Integer) data[child].next_n);
      lua_pushvalue(L, i_ids);
      tk_lua_get_ephemeron(L, TK_EVAL_EPH, data[child].next_assignments);
      if (lua_pcall(L, 7, 0, 0))
      {
        luaL_callmeta(L, -1, "__tostring");
        const char *str = luaL_optstring(L, -1, NULL);
        if (str)
          fprintf(stderr, "%s\n", str);
        lua_pop(L, 1);
      }
    }
    tk_threads_acknowledge_child(pool, child);
  }

  unsigned int i_best = 0;
  for (unsigned int i = 1; i < n_threads; i ++)
    if (data[i].best.bacc > data[i_best].best.bacc)
      i_best = i;

  tk_accuracy_t best = data[i_best].best;
  uint64_t best_m = data[i_best].best_m;
  uint64_t best_n = data[i_best].best_n;
  tk_ivec_t *assignments = data[i_best].best_assignments;
  tk_lua_get_ephemeron(L, TK_EVAL_EPH, assignments); // t ids as

  // Cleanup
  tk_threads_destroy(pool);
  tk_iumap_destroy(state.ididx);

  lua_newtable(L); // t ids as score
  lua_pushinteger(L, (lua_Integer) best_m);
  lua_setfield(L, -2, "margin");
  lua_pushinteger(L, (lua_Integer) best_n);
  lua_setfield(L, -2, "n_clusters");
  lua_pushnumber(L, best.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best.tnr);
  lua_setfield(L, -2, "tnr");
  lua_insert(L, -3); // t score ids as
  lua_pushinteger(L, (int64_t) best_n); // t score ids as n
  return 4;
}

static inline int tm_optimize_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  tk_bits_t *codes;
  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peekopt(L, -1);
  codes = cvec != NULL
    ? (tk_bits_t *) cvec->a
    : (tk_bits_t *) tk_lua_checkustring(L, -1, "codes");

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_retrieval", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_retrieval", "threads");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = NULL;
  state.codes = codes;
  state.dcodes = NULL;
  state.id_code = ids == NULL ? NULL : tk_iumap_from_ivec(ids); // TODO: Pass id_code (aka uid_hood) in instead of recomputing
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.hist_pos = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  state.hist_neg = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  for (uint64_t i = 0; i < n_dims + 1; i ++) {
    atomic_init(state.hist_pos + i, 0);
    atomic_init(state.hist_neg + i, 0);
  }

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  // Calculate total for best margin calculation via pool
  tk_threads_signal(pool, TK_EVAL_ENCODING_SIMILARITY, 0);
  tk_threads_destroy(pool);

  // Calculate histogram
  uint64_t pref_tp[n_dims + 1], pref_fp[n_dims + 1];
  uint64_t running_tp = 0, running_fp = 0;
  for (uint64_t d = 0; d <= n_dims; d ++) {
    running_tp += atomic_load(state.hist_pos + d);
    running_fp += atomic_load(state.hist_neg + d);
    pref_tp[d] = running_tp;
    pref_fp[d] = running_fp;
  }

  double best_bacc = -1.0;
  double best_tpr = -1.0;
  double best_tnr = -1.0;
  uint64_t best_margin = 0;
  for (uint64_t m = 0; m <= n_dims; ++m) {
    uint64_t tp = pref_tp[m];
    uint64_t fp = pref_fp[m];
    uint64_t tn = n_neg - fp;
    double tpr = n_pos > 0 ? (double)tp / (double)n_pos : 0;
    double tnr = n_neg > 0 ? (double)tn / (double)n_neg : 0;
    double bacc = 0.5 * (tpr + tnr); // true balanced accuracy
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, bacc);
      lua_pushnumber(L, tpr);
      lua_pushnumber(L, tnr);
      lua_pushinteger(L, (lua_Integer)m);
      lua_pushnumber(L, auc);
      lua_call(L, 5, 1);
      if (lua_isboolean(L, -1) && !lua_toboolean(L, -1)) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
    if (bacc > best_bacc || (bacc == best_bacc && tpr > best_tpr)) {
      best_bacc = bacc;
      best_tpr = tpr;
      best_tnr = tnr;
      best_margin = m;
    }
  }

  // Cleanup
  free(state.pl);
  free(state.hist_pos);
  free(state.hist_neg);
  if (state.id_code != NULL)
    tk_iumap_destroy(state.id_code);

  // Output
  lua_newtable(L);
  lua_pushinteger(L, (int64_t) best_margin);
  lua_setfield(L, -2, "margin");
  lua_pushnumber(L, best_bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best_tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best_tnr);
  lua_setfield(L, -2, "tnr");
  lua_pushnumber(L, auc);
  lua_setfield(L, -2, "auc");
  return 1;
}

typedef double (*tk_optimize_bits_score_fn)(lua_State *L, tk_eval_t *state, tk_threadpool_t *pool);

static inline tk_ivec_t *tm_optimize_bits_generic (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  bool forward,
  bool use_float,
  int i_each,
  tk_optimize_bits_score_fn score_function
) {
  uint64_t n_dims = state->n_dims;
  tk_cvec_t *mask = tk_cvec_create(L, TK_CVEC_BITS_BYTES(n_dims), 0, 0); // mask
  tk_ivec_t *active = tk_ivec_create(L, 0, 0, 0); // mask active
  tk_ivec_t *removed = tk_ivec_create(L, 0, 0, 0); // mask active removed

  if (forward) {
    memset(mask->a, 0x00, TK_CVEC_BITS_BYTES(n_dims));
    mask->a[0] |= 1;  // Set bit 0
    state->mask_popcount = 1;
    tk_ivec_push(active, 0);
    tk_ivec_setn(removed, n_dims - 1);
    // Fill removed with indices 1 through n_dims-1
    for (uint64_t i = 0; i < n_dims - 1; i ++)
      removed->a[i] = (int64_t) i + 1;
  } else {
    memset(mask->a, 0xFF, TK_CVEC_BITS_BYTES(n_dims));
    state->mask_popcount = n_dims;
    tk_ivec_setn(active, n_dims);
    tk_ivec_fill_indices(active);
  }

  state->mask = (tk_bits_t *) mask->a;
  double best_score = score_function(L, state, pool);
  bool converged = false;
  while (!converged) {

    if (forward) {
      // Forward selection: Add features from removed set
      int64_t best_bit_to_add = -1;
      uint64_t best_idx_to_add = 0;
      double best_score_after_add = best_score;

      for (uint64_t i = 0; i < removed->n; i ++) {
        int64_t bit = removed->a[i];
        state->mask_popcount++;
        mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));
        double score = score_function(L, state, pool);
        state->mask_popcount--;
        mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
        if (score >= best_score_after_add) {
          best_score_after_add = score;
          best_bit_to_add = bit;
          best_idx_to_add = i;
        }
      }

      if (best_bit_to_add >= 0 && best_score_after_add >= best_score) {
        state->mask_popcount++;
        mask->a[TK_CVEC_BITS_BYTE(best_bit_to_add)] |= (1 << TK_CVEC_BITS_BIT(best_bit_to_add));
        tk_ivec_push(active, best_bit_to_add);
        removed->a[best_idx_to_add] = removed->a[--removed->n];
        double gain = best_score_after_add - best_score;
        best_score = best_score_after_add;
        if (i_each > -1) {
          lua_pushvalue(L, i_each);
          lua_pushinteger(L, best_bit_to_add);
          lua_pushnumber(L, gain);
          lua_pushnumber(L, best_score);
          lua_pushstring(L, "add");
          lua_call(L, 4, 0);
        }

        // Floating removal: Remove previously added features if it improves score
        bool float_improved = use_float; // Only run if floating is enabled
        while (float_improved && active->n > 1) {
          float_improved = false;
          int64_t best_float_bit_to_remove = -1;
          uint64_t best_float_idx_to_remove = 0;
          double best_float_score_after_remove = best_score;

          for (uint64_t i = 0; i < active->n; i++) {
            int64_t bit = active->a[i];
            state->mask_popcount--;
            mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
            double score = score_function(L, state, pool);
            state->mask_popcount++;
            mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));
            if (score > best_float_score_after_remove) {
              best_float_score_after_remove = score;
              best_float_bit_to_remove = bit;
              best_float_idx_to_remove = i;
            }
          }

          if (best_float_bit_to_remove >= 0 && best_float_score_after_remove > best_score) {
            state->mask_popcount--;
            mask->a[TK_CVEC_BITS_BYTE(best_float_bit_to_remove)] &= ~(1 << TK_CVEC_BITS_BIT(best_float_bit_to_remove));
            tk_ivec_push(removed, best_float_bit_to_remove);
            active->a[best_float_idx_to_remove] = active->a[--active->n];
            double float_gain = best_float_score_after_remove - best_score;
            best_score = best_float_score_after_remove;
            float_improved = true;
            if (i_each > -1) {
              lua_pushvalue(L, i_each);
              lua_pushinteger(L, best_float_bit_to_remove);
              lua_pushnumber(L, float_gain);
              lua_pushnumber(L, best_score);
              lua_pushstring(L, "float-remove");
              lua_call(L, 4, 0);
            }
          }
        }
      } else {
        converged = true;
      }

    } else {
      // Backward elimination: Remove features from active set
      int64_t best_bit_to_remove = -1;
      uint64_t best_idx_to_remove = 0;
      double best_score_after_remove = best_score;

      for (uint64_t i = 0; i < active->n; i ++) {
        int64_t bit = active->a[i];
        state->mask_popcount--;
        mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
        double score = score_function(L, state, pool);
        state->mask_popcount++;
        mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));
        if (score >= best_score_after_remove) {
          best_score_after_remove = score;
          best_bit_to_remove = bit;
          best_idx_to_remove = i;
        }
      }

      if (best_bit_to_remove >= 0 && best_score_after_remove >= best_score) {
        state->mask_popcount--;
        mask->a[TK_CVEC_BITS_BYTE(best_bit_to_remove)] &= ~(1 << TK_CVEC_BITS_BIT(best_bit_to_remove));
        tk_ivec_push(removed, best_bit_to_remove);
        active->a[best_idx_to_remove] = active->a[--active->n];
        double gain = best_score_after_remove - best_score;
        best_score = best_score_after_remove;
        if (i_each > -1) {
          lua_pushvalue(L, i_each);
          lua_pushinteger(L, best_bit_to_remove);
          lua_pushnumber(L, gain);
          lua_pushnumber(L, best_score);
          lua_pushstring(L, "remove");
          lua_call(L, 4, 0);
        }

        // Floating addition: Add previously removed features if it improves score
        bool float_improved = use_float; // Only run if floating is enabled
        while (float_improved && removed->n > 0) {
          float_improved = false;
          int64_t best_float_bit_to_add = -1;
          uint64_t best_float_idx_to_add = 0;
          double best_float_score_after_add = best_score;

          for (uint64_t i = 0; i < removed->n; i++) {
            int64_t bit = removed->a[i];
            state->mask_popcount++;
            mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));
            double score = score_function(L, state, pool);
            state->mask_popcount--;
            mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
            if (score > best_float_score_after_add) {
              best_float_score_after_add = score;
              best_float_bit_to_add = bit;
              best_float_idx_to_add = i;
            }
          }

          if (best_float_bit_to_add >= 0 && best_float_score_after_add > best_score) {
            state->mask_popcount++;
            mask->a[TK_CVEC_BITS_BYTE(best_float_bit_to_add)] |= (1 << TK_CVEC_BITS_BIT(best_float_bit_to_add));
            tk_ivec_push(active, best_float_bit_to_add);
            removed->a[best_float_idx_to_add] = removed->a[--removed->n];
            double float_gain = best_float_score_after_add - best_score;
            best_score = best_float_score_after_add;
            float_improved = true;
            if (i_each > -1) {
              lua_pushvalue(L, i_each);
              lua_pushinteger(L, best_float_bit_to_add);
              lua_pushnumber(L, float_gain);
              lua_pushnumber(L, best_score);
              lua_pushstring(L, "float-add");
              lua_call(L, 4, 0);
            }
          }
        }
      } else {
        converged = true;
      }
    }
  }

  tk_ivec_destroy(removed);
  tk_cvec_destroy(mask);
  lua_remove(L, -3); // active, removed
  lua_pop(L, 1); // active
  return active;
}

static inline tk_ivec_t *tm_optimize_bits_auc (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  bool forward,
  bool use_float,
  int i_each
) {
  return tm_optimize_bits_generic(L, state, pool, forward, use_float, i_each, _tm_auc);
}

static inline tk_ivec_t *tm_optimize_bits_graph (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  bool forward,
  bool use_float,
  int i_each
) {
  return tm_optimize_bits_generic(L, state, pool, forward, use_float, i_each, _tm_graph_reconstruction);
}

static inline tk_ivec_t *tm_optimize_bits_prefix_only (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  int i_each,
  tk_optimize_bits_score_fn score_function
) {
  uint64_t n_dims = state->n_dims;
  tk_cvec_t *mask = tk_cvec_create(L, TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  const double epsilon = 1e-6;

  double best_score = -INFINITY;
  uint64_t best_k = 0;

  // Test each prefix: bits 0 through k-1
  for (uint64_t k = 1; k <= n_dims; k++) {
    // Set mask for bits 0 through k-1
    memset(mask->a, 0x00, TK_CVEC_BITS_BYTES(n_dims));
    for (uint64_t i = 0; i < k; i++) {
      mask->a[TK_CVEC_BITS_BYTE(i)] |= (1 << TK_CVEC_BITS_BIT(i));
    }

    state->mask = (tk_bits_t *) mask->a;
    state->mask_popcount = k;

    // Evaluate this prefix
    double score = score_function(L, state, pool);

    // Callback for monitoring
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t) k);
      lua_pushnumber(L, 0.0);  // unused gain field
      lua_pushnumber(L, score);
      lua_pushstring(L, "add");
      lua_call(L, 4, 0);
    }

    // Track best with tie-breaking for size
    if (score > best_score + epsilon) {
      // Clear new best
      best_score = score;
      best_k = k;
    } else if (fabs(score - best_score) <= epsilon && k < best_k) {
      // Equal score but smaller size - prefer it
      best_k = k;
    }
  }

  // Return the best prefix as selected bits
  tk_ivec_t *kept_bits = tk_ivec_create(L, best_k, 0, 0);
  kept_bits->n = best_k;
  for (uint64_t i = 0; i < best_k; i++) {
    kept_bits->a[i] = (int64_t) i;
  }

  // Final callback with best result
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) best_k);
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, best_score);
    lua_pushstring(L, "final");
    lua_call(L, 4, 0);
  }

  tk_cvec_destroy(mask);
  return kept_bits;
}

static inline tk_ivec_t *tm_optimize_bits_prefix (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  int i_each,
  tk_optimize_bits_score_fn score_function
) {
  uint64_t n_dims = state->n_dims;
  tk_cvec_t *mask = tk_cvec_create(L, TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  const double epsilon = 1e-6;

  double best_score = -INFINITY;
  uint64_t best_k = 0;

  // Phase 1: Test each prefix: bits 0 through k-1
  for (uint64_t k = 1; k <= n_dims; k++) {
    // Set mask for bits 0 through k-1
    memset(mask->a, 0x00, TK_CVEC_BITS_BYTES(n_dims));
    for (uint64_t i = 0; i < k; i++) {
      mask->a[TK_CVEC_BITS_BYTE(i)] |= (1 << TK_CVEC_BITS_BIT(i));
    }

    state->mask = (tk_bits_t *) mask->a;
    state->mask_popcount = k;

    // Evaluate this prefix
    double score = score_function(L, state, pool);

    // Callback for monitoring
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t) k);
      lua_pushnumber(L, 0.0);  // unused gain field
      lua_pushnumber(L, score);
      lua_pushstring(L, "add");
      lua_call(L, 4, 0);
    }

    // Track best with tie-breaking for size
    if (score > best_score + epsilon) {
      // Clear new best
      best_score = score;
      best_k = k;
    } else if (fabs(score - best_score) <= epsilon && k < best_k) {
      // Equal score but smaller size - prefer it
      best_k = k;
    }
  }

  // Initialize with the best prefix
  tk_ivec_t *active = tk_ivec_create(L, best_k, 0, 0);
  active->n = best_k;
  for (uint64_t i = 0; i < best_k; i++) {
    active->a[i] = (int64_t) i;
  }

  // Set mask to selected prefix
  memset(mask->a, 0x00, TK_CVEC_BITS_BYTES(n_dims));
  for (uint64_t i = 0; i < best_k; i++) {
    mask->a[TK_CVEC_BITS_BYTE(i)] |= (1 << TK_CVEC_BITS_BIT(i));
  }
  state->mask = (tk_bits_t *) mask->a;
  state->mask_popcount = best_k;

  // Report initial state
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) best_k);
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, best_score);
    lua_pushstring(L, "init");
    lua_call(L, 4, 0);
  }

  // Phase 2: Standard SFBS - backward selection with floating
  // Stack order after creates: mask, active, removed
  tk_ivec_t *removed = tk_ivec_create(L, n_dims, 0, 0); // Track removed bits for floating
  bool converged = false;

  while (!converged && active->n > 0) {
    double current_score = score_function(L, state, pool);

    int64_t best_bit_to_remove = -1;
    uint64_t best_idx_to_remove = 0;
    double best_score_after_remove = current_score - epsilon;

    // Try removing each active bit
    for (uint64_t i = 0; i < active->n; i++) {
      int64_t bit = active->a[i];

      // Temporarily remove bit
      state->mask_popcount--;
      mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));

      double score = score_function(L, state, pool);

      // Restore bit
      state->mask_popcount++;
      mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));

      // Check if this removal is beneficial or neutral
      if (score >= best_score_after_remove) {
        best_score_after_remove = score;
        best_bit_to_remove = bit;
        best_idx_to_remove = i;
      }
    }

    // If we found a bit to remove
    if (best_bit_to_remove >= 0 && best_score_after_remove >= current_score - epsilon) {
      // Remove the bit permanently
      state->mask_popcount--;
      mask->a[TK_CVEC_BITS_BYTE(best_bit_to_remove)] &= ~(1 << TK_CVEC_BITS_BIT(best_bit_to_remove));
      tk_ivec_push(removed, best_bit_to_remove);
      active->a[best_idx_to_remove] = active->a[--active->n];

      if (i_each != -1) {
        lua_pushvalue(L, i_each);
        lua_pushinteger(L, best_bit_to_remove);
        lua_pushnumber(L, 0.0);
        lua_pushnumber(L, best_score_after_remove);
        lua_pushstring(L, "remove");
        lua_call(L, 4, 0);
      }

      // Floating addition: Try adding back previously removed bits
      bool float_improved = true;
      while (float_improved && removed->n > 0) {
        float_improved = false;
        double float_current_score = score_function(L, state, pool);

        int64_t best_float_bit_to_add = -1;
        uint64_t best_float_idx_to_add = 0;
        double best_float_score_after_add = float_current_score;

        for (uint64_t i = 0; i < removed->n; i++) {
          int64_t bit = removed->a[i];

          // Temporarily add bit
          state->mask_popcount++;
          mask->a[TK_CVEC_BITS_BYTE(bit)] |= (1 << TK_CVEC_BITS_BIT(bit));

          double score = score_function(L, state, pool);

          // Remove bit again
          state->mask_popcount--;
          mask->a[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));

          if (score > best_float_score_after_add + epsilon) {
            best_float_score_after_add = score;
            best_float_bit_to_add = bit;
            best_float_idx_to_add = i;
          }
        }

        if (best_float_bit_to_add >= 0 && best_float_score_after_add > float_current_score + epsilon) {
          // Add the bit back permanently
          state->mask_popcount++;
          mask->a[TK_CVEC_BITS_BYTE(best_float_bit_to_add)] |= (1 << TK_CVEC_BITS_BIT(best_float_bit_to_add));
          tk_ivec_push(active, best_float_bit_to_add);
          removed->a[best_float_idx_to_add] = removed->a[--removed->n];
          float_improved = true;

          if (i_each != -1) {
            lua_pushvalue(L, i_each);
            lua_pushinteger(L, best_float_bit_to_add);
            lua_pushnumber(L, 0.0);
            lua_pushnumber(L, best_float_score_after_add);
            lua_pushstring(L, "float-add");
            lua_call(L, 4, 0);
          }
        }
      }
    } else {
      converged = true;
    }
  }

  // Clean up - stack has: mask, active, removed
  tk_ivec_destroy(removed);
  tk_cvec_destroy(mask);
  lua_pop(L, 1); // pop removed from stack
  // Now stack just has active, which we return

  return active;
}

static inline int tm_optimize_bits (lua_State *L)
{
  lua_settop(L, 1);

  // === Parse all inputs first ===
  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peek(L, -1, "cvec");

  const char *type = tk_lua_foptstring(L, 1, "optimize_bits", "type", NULL);
  const char *method = tk_lua_foptstring(L, 1, "optimize_bits", "method", "sffs");
  const char *dir = tk_lua_foptstring(L, 1, "optimize_bits", "direction", "backward");
  bool use_float = tk_lua_foptboolean(L, 1, "optimize_bits", "float", true);
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_bits", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_bits", "threads");

  bool forward;
  if (!strcmp(dir, "forward"))
    forward = true;
  else if (!strcmp(dir, "backward"))
    forward = false;
  else
    return tk_lua_verror(L, 3, "optimize_bits", "direction",
                        "must be either 'backward' or 'forward'");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.codes = (tk_bits_t *) cvec->a;
  state.dcodes = NULL;

  lua_getfield(L, 1, "offsets");
  state.offsets = tk_ivec_peekopt(L, -1);
  lua_getfield(L, 1, "neighbors");
  state.neighbors = tk_ivec_peekopt(L, -1);
  lua_getfield(L, 1, "weights");
  state.weights = tk_dvec_peekopt(L, -1);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  lua_getfield(L, 1, "pos");
  state.pos = tk_pvec_peekopt(L, -1);
  lua_getfield(L, 1, "neg");
  state.neg = tk_pvec_peekopt(L, -1);

  bool is_graph = false;
  if (type) {
    if (!strcmp(type, "graph"))
      is_graph = true;
    else if (strcmp(type, "auc") != 0)
      return tk_lua_verror(L, 3, "optimize_bits", "type", "must be either 'auc' or 'graph'");
  } else {
    if (state.offsets && state.neighbors && state.weights)
      is_graph = true;
    else if (!state.pos || !state.neg)
      return tk_lua_verror(L, 3, "optimize_bits", "data", "must provide either graph data (offsets, neighbors, weights) or pair data (pos, neg)");
  }

  if (is_graph) {
    if (!state.offsets || !state.neighbors || !state.weights)
      return tk_lua_verror(L, 3, "optimize_bits", "graph data", "graph mode requires offsets, neighbors, and weights");
    state.pl = NULL;
    state.id_code = NULL;
    state.pos = NULL;
    state.neg = NULL;
    state.n_pos = 0;
    state.n_neg = 0;
  } else {
    if (!state.pos || !state.neg)
      return tk_lua_verror(L, 3, "optimize_bits", "pair data", "auc mode requires pos and neg pairs");
    state.n_pos = state.pos->n;
    state.n_neg = state.neg->n;
    state.id_code = ids ? tk_iumap_from_ivec(ids) : NULL;
    state.pl = malloc((state.n_pos + state.n_neg) * sizeof(tm_dl_t));
    state.offsets = NULL;
    state.neighbors = NULL;
    state.weights = NULL;
  }

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    if (!is_graph && (state.n_pos + state.n_neg) > 0)
      tk_thread_range(i, n_threads, state.n_pos + state.n_neg, &data[i].pfirst, &data[i].plast);
  }

  tk_ivec_t *result;
  if (!strcmp(method, "prefix")) {
    // Use prefix selection method with SFBS
    if (is_graph)
      result = tm_optimize_bits_prefix(L, &state, pool, i_each, _tm_graph_reconstruction);
    else
      result = tm_optimize_bits_prefix(L, &state, pool, i_each, _tm_auc);
  } else if (!strcmp(method, "prefix-only")) {
    // Use prefix selection method without SFBS
    if (is_graph)
      result = tm_optimize_bits_prefix_only(L, &state, pool, i_each, _tm_graph_reconstruction);
    else
      result = tm_optimize_bits_prefix_only(L, &state, pool, i_each, _tm_auc);
  } else if (!strcmp(method, "sffs")) {
    // Use original SFFS method
    if (is_graph)
      result = tm_optimize_bits_graph(L, &state, pool, forward, use_float, i_each);
    else
      result = tm_optimize_bits_auc(L, &state, pool, forward, use_float, i_each);
  } else {
    return tk_lua_verror(L, 3, "optimize_bits", "method",
                        "must be either 'sffs', 'prefix', or 'prefix-only'");
  }

  if (state.pl)
    free(state.pl);
  if (state.id_code)
    tk_iumap_destroy(state.id_code);
  tk_threads_destroy(pool);

  return 1;
}

static inline int tm_entropy_stats (lua_State *L)
{
  lua_settop(L, 4);
  unsigned int n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 3, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 4, "n_threads", NULL);
  tk_cvec_t *cvec = tk_cvec_peekopt(L, 1);
  tk_ivec_t *ivec = NULL;
  tk_dvec_t *entropies = NULL;
  if (cvec == NULL) {
    ivec = tk_ivec_peekopt(L, 1);
    entropies = tk_ivec_bits_score_entropy(L, ivec, n_samples, n_dims, n_threads);
  } else {
    entropies = tk_cvec_bits_score_entropy(L, cvec, n_samples, n_dims, n_threads);
  }

  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_dims; j ++) {
    double entropy = entropies->a[j];
    lua_pushinteger(L, (int64_t) j + 1);
    lua_pushnumber(L, entropy);
    lua_settable(L, -3);
    if (entropy < min_entropy)
      min_entropy = entropy;
    if (entropy > max_entropy)
      max_entropy = entropy;
    sum_entropy += entropy;
  }
  lua_setfield(L, -2, "bits");

  // Aggregate stats
  double mean = sum_entropy / n_dims;
  double variance = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double entropy = entropies->a[j];
    variance += (entropy - mean) * (entropy - mean);
  }

  variance /= n_dims;
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, min_entropy);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_entropy);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, sqrt(variance));
  lua_setfield(L, -2, "std");

  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "clustering_accuracy", tm_clustering_accuracy_lua },
  { "optimize_bits", tm_optimize_bits },
  { "optimize_retrieval", tm_optimize_retrieval },
  { "optimize_clustering", tm_optimize_clustering },
  { "entropy_stats", tm_entropy_stats },
  { "auc", tm_auc },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}

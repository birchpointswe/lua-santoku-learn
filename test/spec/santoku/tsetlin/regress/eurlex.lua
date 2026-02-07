local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local cvec = require("santoku.cvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local hlth = require("santoku.tsetlin.hlth")
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local pvec = require("santoku.pvec")
local rvec = require("santoku.rvec")
local optimize = require("santoku.tsetlin.optimize")
local spectral = require("santoku.tsetlin.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = false,
    skips = 1,
  },
  feature_selection = {
    min_df = 2,
    max_df = 0.5,
    bns_tokens = 65536,
    per_class = 8192,
  },
  regressor = {
    clauses = 256,
    clause_tolerance = { def = 869, min = 8, max = 1024, int = true },
    clause_maximum = { def = 765, min = 8, max = 1024, int = true },
    target = { def = 119, min = 8, max = 1024, int = true },
    specificity = { def = 1182, min = 2, max = 2000 },
    search_rounds = 0,
    search_trials = 20,
    search_iterations = 40,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
  nystrom = {
    normalized = false,
    cholesky = false,
    n_landmarks = 8192,
    n_dims = 8192,
    select = true,
    n_select = 256,
    n_select_min = 8,
    n_select_neg = 32,
    n_project = nil,
    decay = 1.0,
    bandwidth = -1,
    rounds = 0,
    samples = 20,
  },
  rp = {
    all_dims = true,
    max_bits = -4096,
    tolerance = 0.005,
  },
  eval = {
    knn = 16,
    random_pairs = 16,
    ranking = "ndcg",
    retrieval_k = 32,
    retrieval_radius = 3,
  },
}

test("eurlex-docs", function()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local train_lc = train.label_counts:to_dvec()
  train_lc:asc()
  str.printf("  Train: %d  Dev: %d  Test: %d  Labels: %d  Labels/doc: %.1f median\n",
    train.n, dev.n, test_set.n, n_labels, train_lc:get(train_lc:size() / 2))

  print("\nCreating IDs")
  train.ids = ivec.create(train.n):fill_indices()
  dev.ids = ivec.create(dev.n):fill_indices():add(train.n)
  test_set.ids = ivec.create(test_set.n):fill_indices():add(train.n + dev.n)

  print("\nBuilding label CSR for lookups")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }
  local dev_label_offsets, dev_label_neighbors = dev.solutions:bits_to_csr(dev.n, n_labels)
  dev.label_csr = { offsets = dev_label_offsets, neighbors = dev_label_neighbors }
  local test_label_offsets, test_label_neighbors = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  test_set.label_csr = { offsets = test_label_offsets, neighbors = test_label_neighbors }

  print("\nBuilding bipartite graph")
  local n_total_nodes = train.n + n_labels
  local all_ids = ivec.create(n_total_nodes):fill_indices()
  local label_node_ids = ivec.create(n_labels):fill_indices():add(train.n)

  local doc_node_ids = ivec.create(train.n):fill_indices()
  local features, n_feat = train.solutions:bits_bipartite(train.n, n_labels, "adjacency")
  local graph_ranks = ivec.create(n_feat):fill(1, 0, train.n):fill(0, train.n, n_feat)
  local idf_ids, idf_scores = features:bits_top_df(n_total_nodes, n_feat)
  features:bits_select(idf_ids, nil, n_feat)
  local n_features = idf_ids:size()
  local ranks = ivec.create():copy(graph_ranks, idf_ids)
  train.graph_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.graph_index:add(features, all_ids)
  local label_features = ivec.create()
  features:bits_select(nil, label_node_ids, n_features, label_features)
  train.labels_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.labels_index:add(label_features, label_node_ids)
  local doc_features = ivec.create()
  features:bits_select(nil, doc_node_ids, n_features, doc_features)
  train.docs_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.docs_index:add(doc_features, doc_node_ids)
  train.all_features = features
  train.all_ids = all_ids
  train.label_node_ids = label_node_ids
  str.printf("  Adjacency index: %d idf features, %d nodes, %d label nodes\n",
    n_features, n_total_nodes, n_labels)

  print("\nSpot-checking graph index")
  local hood = rvec.create()
  local doc_labels = {}
  for i = 0, train.solutions:size() - 1 do
    local bit = train.solutions:get(i)
    local doc = math.floor(bit / n_labels)
    local label = bit % n_labels
    if not doc_labels[doc] then doc_labels[doc] = {} end
    doc_labels[doc][#doc_labels[doc] + 1] = label
  end

  for _, d in ipairs({ 0, 100, 1000 }) do
    local labels = doc_labels[d] or {}
    str.printf("  Doc %d: %d GT labels\n", d, #labels)
    train.graph_index:neighbors(d, 10, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Top 10 neighbors:\n")
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local kind = nid >= train.n and "label" or "doc"
      local display_id = nid >= train.n and (nid - train.n) or nid
      local is_gt = ""
      if kind == "label" then
        for _, gl in ipairs(labels) do
          if gl == display_id then is_gt = " *GT*"; break end
        end
      end
      str.printf("      [%d] %s %d  sim=%.4f%s\n", i, kind, display_id, 1 - dist, is_gt)
    end

    if #labels > 0 then
      local l0 = labels[1]
      str.printf("    doc-%d <-> GT label %d: %.6f\n", d, l0, train.graph_index:similarity(d, train.n + l0, cfg.nystrom.decay, cfg.nystrom.bandwidth))
      if #labels > 1 then
        local l1 = labels[2]
        str.printf("    doc-%d <-> GT label %d: %.6f\n", d, l1, train.graph_index:similarity(d, train.n + l1, cfg.nystrom.decay, cfg.nystrom.bandwidth))
      end
      local gt_set = {}
      for _, gl in ipairs(labels) do gt_set[gl] = true end
      local neg_l = nil
      for off = 1, n_labels - 1 do
        local c = (l0 + off) % n_labels
        if not gt_set[c] then neg_l = c; break end
      end
      if neg_l then
        str.printf("    doc-%d <-> NEG label %d: %.6f\n", d, neg_l, train.graph_index:similarity(d, train.n + neg_l, cfg.nystrom.decay, cfg.nystrom.bandwidth))
      end

      local found_sharing = false
      for d2 = 0, train.n - 1 do
        if found_sharing then break end
        if d2 ~= d and doc_labels[d2] then
          for _, gl in ipairs(doc_labels[d2]) do
            if gl == l0 then
              str.printf("    doc-%d <-> doc-%d (share label %d): %.6f\n", d, d2, l0, train.graph_index:similarity(d, d2, cfg.nystrom.decay, cfg.nystrom.bandwidth))
              found_sharing = true
              break
            end
          end
        end
      end

      for d3 = 0, train.n - 1 do
        if doc_labels[d3] then
          local has = false
          for _, gl in ipairs(doc_labels[d3]) do
            for _, ml in ipairs(labels) do
              if gl == ml then has = true; break end
            end
            if has then break end
          end
          if not has then
            str.printf("    doc-%d <-> doc-%d (no shared labels): %.6f\n", d, d3, train.graph_index:similarity(d, d3, cfg.nystrom.decay, cfg.nystrom.bandwidth))
            break
          end
        end
      end

      str.printf("    label-%d <-> label-%d: %.6f\n", l0, (l0 + 1) % n_labels,
        train.graph_index:similarity(train.n + l0, train.n + ((l0 + 1) % n_labels), cfg.nystrom.decay, cfg.nystrom.bandwidth))
    end
  end

  print("\n  Doc->label lookup (labels-only index):")
  local doc_feats = ivec.create()
  for _, d in ipairs({ 0, 100, 1000 }) do
    local labels = doc_labels[d] or {}
    train.graph_index:get(d, doc_feats)
    train.labels_index:neighbors(doc_feats, 10, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Doc %d (%d GT labels): top 10 label neighbors:\n", d, #labels)
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local lid = nid - train.n
      local is_gt = ""
      for _, gl in ipairs(labels) do
        if gl == lid then is_gt = " *GT*"; break end
      end
      str.printf("      [%d] label %d  sim=%.4f%s\n", i, lid, 1 - dist, is_gt)
    end
  end

  print("\n  Label neighbors (sample):")
  for _, l in ipairs({ 0, 100 }) do
    train.graph_index:neighbors(train.n + l, 5, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Label %d: top 5:\n", l)
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local kind = nid >= train.n and "label" or "doc"
      local display_id = nid >= train.n and (nid - train.n) or nid
      str.printf("      [%d] %s %d  sim=%.4f\n", i, kind, display_id, 1 - dist)
    end
  end

  print("\nBuilding full evaluation adjacency (bipartite graph)")
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      knn_index = train.graph_index,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      knn_decay = cfg.nystrom.decay,
      knn_bandwidth = cfg.nystrom.bandwidth,
      weight_index = train.graph_index,
      weight_decay = cfg.nystrom.decay,
      weight_bandwidth = cfg.nystrom.bandwidth,
      random_pairs = cfg.eval.random_pairs,
      bridge = "none",
    })

  print("\n--- Kernel quality diagnostics ---")

  do
    local n_eval = train_eval_ids:size()
    local dd, dl, ld, ll = 0, 0, 0, 0
    local dd_w, dl_w, ld_w, ll_w = 0, 0, 0, 0
    local dd_min, dl_min, ld_min, ll_min = 1/0, 1/0, 1/0, 1/0
    local dd_max, dl_max, ld_max, ll_max = -1/0, -1/0, -1/0, -1/0
    local dl_gt_w, dl_gt_n, dl_neg_w, dl_neg_n = 0, 0, 0, 0
    for i = 0, n_eval - 1 do
      local uid = train_eval_ids:get(i)
      local o0 = train_eval_offsets:get(i)
      local o1 = train_eval_offsets:get(i + 1)
      for j = o0, o1 - 1 do
        local nid = train_eval_ids:get(train_eval_neighbors:get(j))
        local w = train_eval_weights:get(j)
        if uid < train.n then
          if nid < train.n then
            dd = dd + 1; dd_w = dd_w + w
            if w < dd_min then dd_min = w end
            if w > dd_max then dd_max = w end
          else
            dl = dl + 1; dl_w = dl_w + w
            if w < dl_min then dl_min = w end
            if w > dl_max then dl_max = w end
            local li = nid - train.n
            local is_gt = false
            local gs = train.label_csr.offsets:get(uid)
            local ge = train.label_csr.offsets:get(uid + 1)
            for g = gs, ge - 1 do
              if train.label_csr.neighbors:get(g) == li then is_gt = true; break end
            end
            if is_gt then dl_gt_w = dl_gt_w + w; dl_gt_n = dl_gt_n + 1
            else dl_neg_w = dl_neg_w + w; dl_neg_n = dl_neg_n + 1 end
          end
        else
          if nid < train.n then
            ld = ld + 1; ld_w = ld_w + w
            if w < ld_min then ld_min = w end
            if w > ld_max then ld_max = w end
          else
            ll = ll + 1; ll_w = ll_w + w
            if w < ll_min then ll_min = w end
            if w > ll_max then ll_max = w end
          end
        end
      end
    end
    str.printf("  Eval adj: %d nodes, %d edges\n", n_eval, train_eval_neighbors:size())
    str.printf("  %-12s %8s %10s %10s %10s\n", "type", "count", "mean_w", "min_w", "max_w")
    str.printf("  %-12s %8d %10.4f %10.4f %10.4f\n", "doc-doc", dd, dd > 0 and dd_w/dd or 0, dd_min, dd_max)
    str.printf("  %-12s %8d %10.4f %10.4f %10.4f\n", "doc-label", dl, dl > 0 and dl_w/dl or 0, dl_min, dl_max)
    str.printf("  %-12s %8d %10.4f %10.4f %10.4f\n", "label-doc", ld, ld > 0 and ld_w/ld or 0, ld_min, ld_max)
    str.printf("  %-12s %8d %10.4f %10.4f %10.4f\n", "label-label", ll, ll > 0 and ll_w/ll or 0, ll_min, ll_max)
    str.printf("  doc->label GT edges: %d mean_w=%.4f\n", dl_gt_n, dl_gt_n > 0 and dl_gt_w/dl_gt_n or 0)
    str.printf("  doc->label non-GT edges: %d mean_w=%.4f\n", dl_neg_n, dl_neg_n > 0 and dl_neg_w/dl_neg_n or 0)
  end

  do
    local kern_all = eval.ranking_accuracy({
      kernel_index = train.graph_index,
      kernel_decay = cfg.nystrom.decay,
      kernel_bandwidth = cfg.nystrom.bandwidth,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
      ranking = cfg.eval.ranking,
    })
    str.printf("  Kernel NDCG (full eval adj): %.4f\n", kern_all.score)
  end

  str.printf("\n  Doc->label kernel ranking (GT + %d random neg):\n", cfg.eval.random_pairs)
  do
    local n_random = cfg.eval.random_pairs
    local dl_ids = ivec.create(train.n):fill_indices()
    dl_ids:copy(label_node_ids)
    local dl_offsets = ivec.create()
    local dl_neighbors = ivec.create()
    local dl_weights = dvec.create()
    for d = 0, train.n - 1 do
      dl_offsets:push(dl_neighbors:size())
      local gt_start = train.label_csr.offsets:get(d)
      local gt_end = train.label_csr.offsets:get(d + 1)
      local gt_set = {}
      for j = gt_start, gt_end - 1 do
        local li = train.label_csr.neighbors:get(j)
        gt_set[li] = true
        dl_neighbors:push(train.n + li)
        dl_weights:push(1.0)
      end
      local added = 0
      while added < n_random do
        local li = math.random(0, n_labels - 1)
        if not gt_set[li] then
          gt_set[li] = true
          dl_neighbors:push(train.n + li)
          dl_weights:push(0.0)
          added = added + 1
        end
      end
    end
    dl_offsets:push(dl_neighbors:size())
    local dl_kern = eval.ranking_accuracy({
      kernel_index = train.graph_index,
      kernel_decay = cfg.nystrom.decay,
      kernel_bandwidth = cfg.nystrom.bandwidth,
      eval_ids = dl_ids, eval_offsets = dl_offsets,
      eval_neighbors = dl_neighbors, eval_weights = dl_weights,
      ranking = cfg.eval.ranking,
    })
    str.printf("    NDCG: %.4f\n", dl_kern.score)
  end

  str.printf("\n  Doc->label kernel retrieval (k=%d):\n", cfg.eval.retrieval_k)
  do
    local k = cfg.eval.retrieval_k
    local hood = rvec.create()
    local doc_feats = ivec.create()
    local total_p, total_r, total_f1 = 0, 0, 0
    local n_docs_eval = 0
    local n_perfect = 0
    for d = 0, train.n - 1 do
      local gt_start = train.label_csr.offsets:get(d)
      local gt_end = train.label_csr.offsets:get(d + 1)
      local n_gt = gt_end - gt_start
      if n_gt > 0 then
        local gt_set = {}
        for j = gt_start, gt_end - 1 do
          gt_set[train.label_csr.neighbors:get(j)] = true
        end
        train.graph_index:get(d, doc_feats)
        train.labels_index:neighbors(doc_feats, k, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
        local hits = 0
        for i = 0, hood:size() - 1 do
          local nid = hood:get(i)
          if gt_set[nid - train.n] then hits = hits + 1 end
        end
        local p = (hood:size() > 0) and hits / hood:size() or 0
        local r = hits / n_gt
        local f1 = (p + r > 0) and 2 * p * r / (p + r) or 0
        total_p = total_p + p
        total_r = total_r + r
        total_f1 = total_f1 + f1
        n_docs_eval = n_docs_eval + 1
        if r == 1.0 then n_perfect = n_perfect + 1 end
      end
    end
    str.printf("    P=%.4f R=%.4f F1=%.4f (%d docs, %d perfect recall)\n",
      total_p / n_docs_eval, total_r / n_docs_eval, total_f1 / n_docs_eval,
      n_docs_eval, n_perfect)
  end

  str.printf("\n  Label->doc kernel retrieval (k=%d):\n", cfg.eval.retrieval_k)
  do
    local k = cfg.eval.retrieval_k
    local hood = rvec.create()
    local label_feats = ivec.create()
    local label_doc_sets = {}
    for d = 0, train.n - 1 do
      local s = train.label_csr.offsets:get(d)
      local e = train.label_csr.offsets:get(d + 1)
      for j = s, e - 1 do
        local l = train.label_csr.neighbors:get(j)
        if not label_doc_sets[l] then label_doc_sets[l] = {} end
        label_doc_sets[l][d] = true
      end
    end
    local total_p, total_r = 0, 0
    local n_labels_eval = 0
    for l = 0, n_labels - 1 do
      local gt_docs = label_doc_sets[l]
      if gt_docs then
        local n_gt = 0
        for _ in pairs(gt_docs) do n_gt = n_gt + 1 end
        train.graph_index:get(train.n + l, label_feats)
        train.docs_index:neighbors(label_feats, k, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
        local hits = 0
        for i = 0, hood:size() - 1 do
          local nid = hood:get(i)
          if gt_docs[nid] then hits = hits + 1 end
        end
        total_p = total_p + ((hood:size() > 0) and hits / hood:size() or 0)
        total_r = total_r + hits / n_gt
        n_labels_eval = n_labels_eval + 1
      end
    end
    str.printf("    P=%.4f R=%.4f (%d labels)\n",
      total_p / n_labels_eval, total_r / n_labels_eval, n_labels_eval)
  end

  print("\nRunning spectral embedding (Nystrom)")
  local landmarks_index = train.labels_index
  local spectral_metrics
  local model = optimize.spectral({
    index = train.graph_index,
    landmarks_index = landmarks_index,
    normalized = cfg.nystrom.normalized,
    cholesky = cfg.nystrom.cholesky,
    train_tokens = train.all_features,
    train_ids = train.all_ids,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
    rounds = cfg.nystrom.rounds,
    samples = cfg.nystrom.samples,
    expected = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    eval = { ranking = cfg.eval.ranking },
    each = function(ev)
      util.spectral_log(ev)
      if ev.event == "eval" or ev.event == "done" then
        spectral_metrics = ev.metrics or ev.best_metrics
      end
    end,
  })
  local spectral_dims = model.dims
  local all_raw_codes = model.raw_codes
  local embedded_ids = model.ids
  local n_embedded = embedded_ids:size()
  str.printf("  Spectral dims: %d, embedded: %d, expected: %d (docs=%d, labels=%d)\n",
    spectral_dims, n_embedded, n_total_nodes, train.n, n_labels)
  str.printf("  embedded_ids: size=%d, min=%d, max=%d, train.n=%d\n",
    embedded_ids:size(), embedded_ids:min(), embedded_ids:max(), train.n)

  print("\nAnalyzing landmark selection:")
  local n_doc_landmarks = 0
  local n_label_landmarks = 0
  for i = 0, model.landmark_ids:size() - 1 do
    local lid = model.landmark_ids:get(i)
    if lid < train.n then n_doc_landmarks = n_doc_landmarks + 1
    else n_label_landmarks = n_label_landmarks + 1 end
  end
  str.printf("  Landmarks: %d docs, %d labels (%.1f%% labels)\n",
    n_doc_landmarks, n_label_landmarks, 100 * n_label_landmarks / model.landmark_ids:size())
  str.printf("  Expected if uniform: %.1f%% labels\n", 100 * n_labels / n_total_nodes)

  if cfg.nystrom.select then
    print("\nDimension selection (SFBS)")
    local n_neg = cfg.nystrom.n_select_neg or 16
    local sel_ids = ivec.create(train.n):fill_indices()
    sel_ids:copy(ivec.create(n_labels):fill_indices():add(train.n))
    local sel_offsets = ivec.create()
    local sel_neighbors = ivec.create()
    local sel_weights = dvec.create()
    for d = 0, train.n - 1 do
      sel_offsets:push(sel_neighbors:size())
      local gt_start = train.label_csr.offsets:get(d)
      local gt_end = train.label_csr.offsets:get(d + 1)
      local gt_set = {}
      for j = gt_start, gt_end - 1 do
        local li = train.label_csr.neighbors:get(j)
        gt_set[li] = true
        sel_neighbors:push(train.n + li)
        sel_weights:push(1.0)
      end
      local added = 0
      while added < n_neg do
        local li = math.random(0, n_labels - 1)
        if not gt_set[li] then
          gt_set[li] = true
          sel_neighbors:push(train.n + li)
          sel_weights:push(0.0)
          added = added + 1
        end
      end
    end
    for _ = 0, n_labels - 1 do
      sel_offsets:push(sel_neighbors:size())
    end
    sel_offsets:push(sel_neighbors:size())
    str.printf("  Doc->label eval: %d nodes, %d edges (%d neg/doc)\n",
      sel_ids:size(), sel_neighbors:size(), n_neg)
    local selected_dims = eval.optimize_bits({
      raw_codes = all_raw_codes,
      ids = embedded_ids,
      screen_k = cfg.nystrom.n_select_screen,
      proxy_only = cfg.nystrom.n_select_proxy,
      n_dims = spectral_dims,
      target_dims = cfg.nystrom.n_select,
      min_dims = cfg.nystrom.n_select_min or 0,
      expected_ids = sel_ids,
      expected_offsets = sel_offsets,
      expected_neighbors = sel_neighbors,
      expected_weights = sel_weights,
      each = function(bit, gain, score, action)
        str.printf("  %s dim %d: gain=%.6f score=%.4f\n", action, bit, gain, score)
      end,
    })
    str.printf("  Selected %d / %d dims\n", selected_dims:size(), spectral_dims)
    local filtered = dvec.create()
    all_raw_codes:mtx_select(selected_dims, nil, spectral_dims, filtered)
    all_raw_codes = filtered
    spectral_dims = selected_dims:size()
  end

  if cfg.nystrom.n_project then
    print("\nSupervised projection")
    local proj_n_out = cfg.nystrom.n_project or spectral_dims
    local gt_doc_ids = ivec.create(train.n):fill_indices()
    local gt_label_uids = ivec.create():copy(train.label_csr.neighbors):add(train.n)
    local projected, proj_eigs = spectral.supervised_project({
      raw_codes = all_raw_codes,
      ids = embedded_ids,
      gt_ids = gt_doc_ids,
      gt_offsets = train.label_csr.offsets,
      gt_neighbors = gt_label_uids,
      n_dims = spectral_dims,
      n_out = proj_n_out,
    })
    all_raw_codes = projected
    spectral_dims = proj_n_out
    str.printf("  Projected: %d x %d, eigs: %.4f .. %.4f\n",
      n_embedded, spectral_dims, proj_eigs:get(0), proj_eigs:get(proj_n_out - 1))
  end

  print("\nOptimizing SignRP configuration")
  local rp_ranks, rp_scores, rp_dims, rp_bits = optimize.rp({
    raw_codes = all_raw_codes,
    ids = embedded_ids,
    n_samples = n_embedded,
    n_dims = cfg.rp.all_dims and -spectral_dims or spectral_dims,
    max_bits = cfg.rp.max_bits,
    tolerance = cfg.rp.tolerance,
    eval = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    ranking = cfg.eval.ranking,
  })
  local best_idx, _ = rp_ranks:get(0)
  local best_score = rp_scores:get(best_idx)
  local best_dims = rp_dims:get(best_idx)
  local best_bits = rp_bits:get(best_idx)
  str.printf("  Best: dims=%d bits=%d score=%.4f\n", best_dims, best_bits, best_score)
  str.printf("  Scores (by preference):\n")
  for i = 0, math.min(10, rp_ranks:size() - 1) do
    local idx, rank = rp_ranks:get(i)
    local marker = (rank == 0) and " *" or ""
    str.printf("    dims=%d bits=%d: %.4f%s\n", rp_dims:get(idx), rp_bits:get(idx), rp_scores:get(idx), marker)
  end

  print("\nTruncating spectral codes to best dims")
  str.printf("  spectral_dims=%d, best_dims=%d, n_embedded=%d\n", spectral_dims, best_dims, n_embedded)
  str.printf("  all_raw_codes:size() before truncation = %d (expected %d)\n",
    all_raw_codes:size(), n_embedded * spectral_dims)
  if best_dims < spectral_dims then
    local selected_cols = ivec.create(best_dims):fill_indices()
    local truncated_all = dvec.create()
    all_raw_codes:mtx_select(selected_cols, nil, spectral_dims, truncated_all)
    all_raw_codes = truncated_all
    str.printf("  all_raw_codes:size() after truncation = %d (expected %d)\n",
      all_raw_codes:size(), n_embedded * best_dims)
  end
  train.dims = best_dims
  train.rp_bits = best_bits

  print("\nCreating SignRP encoder from spectral codes")
  local rp_encode, rp_n_bits = hlth.rp_encoder({
    n_dims = train.dims,
    rp_dims = train.rp_bits,
  })

  local train_wanted = ivec.create(train.n):fill_indices()
  str.printf("  train_wanted: size=%d, min=%d, max=%d\n",
    train_wanted:size(), train_wanted:min(), train_wanted:max())
  local train_embedded_ids = embedded_ids:set_intersect(train_wanted)
  str.printf("  train_embedded_ids: size=%d\n", train_embedded_ids:size())
  local train_raw_codes = dvec.create():mtx_extend(all_raw_codes, train_embedded_ids, embedded_ids, 0, best_dims, true)

  local label_wanted = ivec.create(n_labels):fill_indices():add(train.n)
  str.printf("  label_wanted: size=%d, min=%d, max=%d\n",
    label_wanted:size(), label_wanted:min(), label_wanted:max())
  local embedded_label_ids = embedded_ids:set_intersect(label_wanted)
  str.printf("  embedded_label_ids: size=%d\n", embedded_label_ids:size())
  local n_embedded_labels = embedded_label_ids:size()
  local label_raw_codes = dvec.create():mtx_extend(all_raw_codes, embedded_label_ids, embedded_ids, 0, best_dims, true)

  print("\nPre-computing label RP codes")
  local label_codes_rp = rp_encode(label_raw_codes)
  str.printf("  Label codes: %d/%d labels embedded, %d dims, %d bits RP\n",
    n_embedded_labels, n_labels, train.dims, rp_n_bits)
  train.embedded_label_ids = embedded_label_ids

  print("\n" .. string.rep("=", 60))
  print("PRE-REGRESSOR DIAGNOSTICS (all similarity [0,1], higher=closer)")
  print(string.rep("=", 60))

  local train_codes_rp = rp_encode(train_raw_codes)

  local sp_label_ann = ann.create({ features = rp_n_bits })
  sp_label_ann:add(label_codes_rp, train.embedded_label_ids)

  local spectral_index = ann.create({ features = rp_n_bits })
  local all_rp = cvec.create()
  all_rp:copy(train_codes_rp)
  all_rp:copy(label_codes_rp)
  local all_rp_ids = ivec.create(train.n):fill_indices()
  all_rp_ids:copy(train.embedded_label_ids)
  spectral_index:add(all_rp, all_rp_ids)

  str.printf("\n1. Ranking accuracy (%s, all<->all eval adjacency)\n", cfg.eval.ranking)
  local sp_combined_norm = dvec.create()
  sp_combined_norm:copy(train_raw_codes)
  sp_combined_norm:copy(label_raw_codes)
  local sp_combined_rp = cvec.create()
  sp_combined_rp:copy(train_codes_rp)
  sp_combined_rp:copy(label_codes_rp)
  local sp_combined_ids = ivec.create(train.n):fill_indices()
  sp_combined_ids:copy(train.embedded_label_ids)
  local sp_raw_stats = eval.ranking_accuracy({
    raw_codes = sp_combined_norm, ids = sp_combined_ids, n_dims = train.dims,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local sp_rp_stats = eval.ranking_accuracy({
    codes = sp_combined_rp, ids = sp_combined_ids, n_dims = rp_n_bits,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })
  str.printf("  Graph kernel:  %.4f\n", spectral_metrics.kernel_score)
  str.printf("  Raw cosine:    %.4f\n", sp_raw_stats.score)
  str.printf("  Hamming:       %.4f\n", sp_rp_stats.score)

  str.printf("\n1b. Doc->label ranking accuracy (%s, GT + %d random negatives)\n",
    cfg.eval.ranking, cfg.eval.random_pairs)
  do
    local n_random = cfg.eval.random_pairs
    local dl_ids = ivec.create(train.n):fill_indices()
    dl_ids:copy(train.embedded_label_ids)
    local dl_offsets = ivec.create()
    local dl_neighbors = ivec.create()
    local dl_weights = dvec.create()
    for d = 0, train.n - 1 do
      dl_offsets:push(dl_neighbors:size())
      local gt_start = train.label_csr.offsets:get(d)
      local gt_end = train.label_csr.offsets:get(d + 1)
      local gt_set = {}
      for j = gt_start, gt_end - 1 do
        local li = train.label_csr.neighbors:get(j)
        gt_set[li] = true
        dl_neighbors:push(train.n + li)
        dl_weights:push(1.0)
      end
      local added = 0
      while added < n_random do
        local li = math.random(0, n_labels - 1)
        if not gt_set[li] then
          gt_set[li] = true
          dl_neighbors:push(train.n + li)
          dl_weights:push(0.0)
          added = added + 1
        end
      end
    end
    dl_offsets:push(dl_neighbors:size())
    local dl_kern = eval.ranking_accuracy({
      kernel_index = train.graph_index,
      kernel_decay = cfg.nystrom.decay,
      kernel_bandwidth = cfg.nystrom.bandwidth,
      eval_ids = dl_ids, eval_offsets = dl_offsets,
      eval_neighbors = dl_neighbors, eval_weights = dl_weights,
      ranking = cfg.eval.ranking,
    })
    local dl_raw = eval.ranking_accuracy({
      raw_codes = sp_combined_norm, ids = sp_combined_ids, n_dims = train.dims,
      eval_ids = dl_ids, eval_offsets = dl_offsets,
      eval_neighbors = dl_neighbors, eval_weights = dl_weights,
      ranking = cfg.eval.ranking,
    })
    local dl_rp = eval.ranking_accuracy({
      codes = sp_combined_rp, ids = sp_combined_ids, n_dims = rp_n_bits,
      eval_ids = dl_ids, eval_offsets = dl_offsets,
      eval_neighbors = dl_neighbors, eval_weights = dl_weights,
      ranking = cfg.eval.ranking,
    })
    str.printf("  Graph kernel:  %.4f\n", dl_kern.score)
    str.printf("  Raw cosine:    %.4f\n", dl_raw.score)
    str.printf("  Hamming:       %.4f\n", dl_rp.score)
  end

  str.printf("\n2. RP bit entropy\n")
  local sp_entropy = eval.entropy_stats(train_codes_rp, train.n, rp_n_bits)
  str.printf("  mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)

  str.printf("\n3. Label retrieval (doc->label, Hamming ANN, k=%d, radius=%d)\n",
    cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
  local sp_expected_neighbors = ivec.create():copy(train.label_csr.neighbors):add(train.n)
  local sp_hood_ids, sp_hoods = sp_label_ann:neighborhoods_by_vecs(
    train_codes_rp, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
  local sp_ks, sp_ret = eval.retrieval_ks({
    hoods = sp_hoods,
    hood_ids = sp_hood_ids,
    expected_offsets = train.label_csr.offsets,
    expected_neighbors = sp_expected_neighbors,
  })
  str.printf("  micro: P=%.4f R=%.4f F1=%.4f\n",
    sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1)
  str.printf("  macro: P=%.4f R=%.4f F1=%.4f\n",
    sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)

  str.printf("\n4. Separation analysis (doc->label, sampled)\n")
  do
    local n_samples = math.min(200, train.n)
    local step = math.max(1, math.floor(train.n / n_samples))
    local g_gt_sum, g_neg_sum, h_gt_sum, h_neg_sum = 0, 0, 0, 0
    local n_gt_pairs, n_neg_pairs = 0, 0
    local neg_step = math.max(1, math.floor(n_labels / 10))
    for si = 0, n_samples - 1 do
      local d = si * step
      if d >= train.n then break end
      local gt_start = train.label_csr.offsets:get(d)
      local gt_end = train.label_csr.offsets:get(d + 1)
      local gt_set = {}
      for j = gt_start, gt_end - 1 do
        local li = train.label_csr.neighbors:get(j)
        gt_set[li] = true
        local nid = li + train.n
        g_gt_sum = g_gt_sum + train.graph_index:similarity(d, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
        h_gt_sum = h_gt_sum + spectral_index:similarity(d, nid)
        n_gt_pairs = n_gt_pairs + 1
      end
      local nc = 0
      for l = 0, n_labels - 1, neg_step do
        if not gt_set[l] and nc < 10 then
          local nid = l + train.n
          g_neg_sum = g_neg_sum + train.graph_index:similarity(d, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
          h_neg_sum = h_neg_sum + spectral_index:similarity(d, nid)
          n_neg_pairs = n_neg_pairs + 1
          nc = nc + 1
        end
      end
    end
    local g_gt = n_gt_pairs > 0 and g_gt_sum / n_gt_pairs or 0
    local g_neg = n_neg_pairs > 0 and g_neg_sum / n_neg_pairs or 0
    local h_gt = n_gt_pairs > 0 and h_gt_sum / n_gt_pairs or 0
    local h_neg = n_neg_pairs > 0 and h_neg_sum / n_neg_pairs or 0
    str.printf("               graph    hamming\n")
    str.printf("  mean GT:     %.4f   %.4f\n", g_gt, h_gt)
    str.printf("  mean NEG:    %.4f   %.4f\n", g_neg, h_neg)
    str.printf("  gap:         %.4f   %.4f\n", g_gt - g_neg, h_gt - h_neg)
    str.printf("  (%d docs, %d GT pairs, %d NEG pairs)\n", n_samples, n_gt_pairs, n_neg_pairs)
  end

  str.printf("\n5. Label code quality (%d labels, %d RP bits)\n", n_embedded_labels, rp_n_bits)
  do
    local sample_step = math.max(1, math.floor(n_embedded_labels / 30))
    local sim_sum, n_pairs, n_collisions = 0, 0, 0
    local collision_threshold = 1 - 5 / rp_n_bits
    for i = 0, n_embedded_labels - 1, sample_step do
      for j = i + 1, n_embedded_labels - 1, sample_step do
        local sim = spectral_index:similarity(
          train.embedded_label_ids:get(i),
          train.embedded_label_ids:get(j))
        sim_sum = sim_sum + sim
        n_pairs = n_pairs + 1
        if sim >= collision_threshold then
          n_collisions = n_collisions + 1
        end
      end
    end
    str.printf("  Mean pairwise label sim: %.4f (random expect: 0.5000)\n",
      n_pairs > 0 and sim_sum / n_pairs or 0)
    str.printf("  Near-collisions (<=5 bits): %d/%d pairs (%.1f%%)\n",
      n_collisions, n_pairs, n_pairs > 0 and 100 * n_collisions / n_pairs or 0)
  end

  str.printf("\n6. Spot-checks (GT labels + top non-GT retrieval neighbors)\n")
  for _, doc_id in ipairs({ 0, 100, 1000 }) do
    local gt_start = train.label_csr.offsets:get(doc_id)
    local gt_end = train.label_csr.offsets:get(doc_id + 1)
    local n_gt = gt_end - gt_start
    local gt_set = {}
    for j = gt_start, gt_end - 1 do
      gt_set[train.label_csr.neighbors:get(j)] = true
    end
    str.printf("\n  Doc %d (%d GT labels):\n", doc_id, n_gt)
    str.printf("    %-8s %-4s %-10s %-10s\n", "label", "GT?", "graph", "hamming")
    str.printf("    %-8s %-4s %-10s %-10s\n", "-----", "---", "-----", "-------")
    for j = gt_start, gt_end - 1 do
      local li = train.label_csr.neighbors:get(j)
      local nid = li + train.n
      str.printf("    %-8d  *   %.4f    %.4f\n", li,
        train.graph_index:similarity(doc_id, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth),
        spectral_index:similarity(doc_id, nid))
    end
    local sp_hood = sp_hoods:get(doc_id)
    local shown = 0
    for j = 0, sp_hood:size() - 1 do
      if shown >= 5 then break end
      local idx, _ = sp_hood:get(j)
      local uid = sp_hood_ids:get(idx)
      local li = uid - train.n
      if not gt_set[li] then
        str.printf("    %-8d       %.4f    %.4f  <- non-GT\n", li,
          train.graph_index:similarity(doc_id, uid, cfg.nystrom.decay, cfg.nystrom.bandwidth),
          spectral_index:similarity(doc_id, uid))
        shown = shown + 1
      end
    end
    local hits = 0
    for j = 0, sp_hood:size() - 1 do
      local idx, _ = sp_hood:get(j)
      if gt_set[sp_hood_ids:get(idx) - train.n] then hits = hits + 1 end
    end
    str.printf("    Retrieval: %d/%d GT in top-%d\n", hits, n_gt, sp_hood:size())
  end

  if true then
    return
  end

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  local token_index = tok:index()
  tok = nil
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil

  print("\nFeature selection (DF filter)")
  local min_df = cfg.feature_selection.min_df
  local max_df = cfg.feature_selection.max_df
  local df_ids, idf_weights = train.tokens:bits_top_df(train.n, n_tokens, nil, -min_df, max_df)
  str.printf("  DF filter (min=%d, max=%.0f%%): %d -> %d\n", min_df, max_df * 100, n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  dev.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()

  print("\nBNS feature selection")
  local bns_ids, bns_weights = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels, nil, cfg.feature_selection.bns_tokens, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  BNS selection: %d -> %d tokens\n", n_tokens, n_bns_tokens)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = n_bns_tokens

  print("\nFeature selection for regressor (F-score)")
  local union_feat_ids, _, class_offsets, class_feat_ids, class_scores = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, train.dims, cfg.feature_selection.per_class)
  str.printf("  Features: %d union, %d grouped (%d per dim x %d dims)\n",
    union_feat_ids:size(), class_feat_ids:size(), cfg.feature_selection.per_class, train.dims)
  str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
    class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())

  local make_regressor_sentences = function(tokens, n_samples)
    return tokens:bits_to_cvec(n_samples, n_tokens, class_offsets, class_feat_ids, true)
  end

  local train_regressor_sentences, max_k = make_regressor_sentences(train.tokens, train.n)
  local n_regressor_features = max_k
  str.printf("  Regressor features: %d\n", n_regressor_features)

  print("\nTraining TM regressor")
  local predicted_buf = dvec.create()
  train.tm = optimize.regressor({
    outputs = train.dims,
    samples = train.n,
    problems = train_regressor_sentences,
    features = n_regressor_features,
    grouped = true,
    clauses = cfg.regressor.clauses,
    clause_tolerance = cfg.regressor.clause_tolerance,
    clause_maximum = cfg.regressor.clause_maximum,
    target = cfg.regressor.target,
    specificity = cfg.regressor.specificity,
    search_rounds = cfg.regressor.search_rounds,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    targets = train_raw_codes,
    search_metric = function (t)
      local predicted = t:regress(train_regressor_sentences, train.n, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, train_raw_codes)
      return -stats.mean, stats
    end,
    each = util.make_regressor_log(stopwatch),
  })

  print("\nBuilding labels-only ANN for retrieval evaluation")
  local label_ann = ann.create({ features = rp_n_bits })
  label_ann:add(label_codes_rp, train.embedded_label_ids)
  str.printf("  Label ANN: %d labels indexed, %d bits\n", train.embedded_label_ids:size(), rp_n_bits)

  local train_eval_adj = {
    ids = train_eval_ids, offsets = train_eval_offsets,
    neighbors = train_eval_neighbors, weights = train_eval_weights,
  }

  local function evaluate_split(split, name, eval_adj)
    print("\nEvaluating " .. name)
    local sentences = make_regressor_sentences(split.tokens, split.n)
    local predicted_raw = train.tm:regress(sentences, split.n, true)
    local predicted_rp = rp_encode(predicted_raw)
    local combined_raw = dvec.create()
    combined_raw:copy(predicted_raw)
    combined_raw:copy(label_raw_codes)
    local combined_rp = cvec.create()
    combined_rp:copy(predicted_rp)
    combined_rp:copy(label_codes_rp)
    local combined_ids = ivec.create(split.n):fill_indices()
    combined_ids:copy(train.embedded_label_ids)
    local raw_stats = eval.ranking_accuracy({
      raw_codes = combined_raw, ids = combined_ids, n_dims = train.dims,
      eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
      eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      ranking = cfg.eval.ranking,
    })
    local rp_stats = eval.ranking_accuracy({
      codes = combined_rp, ids = combined_ids, n_dims = rp_n_bits,
      eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
      eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      ranking = cfg.eval.ranking,
    })
    local rp_entropy = eval.entropy_stats(predicted_rp, split.n, rp_n_bits)
    str.printf("  Raw ranking:  %.4f\n", raw_stats.score)
    str.printf("  RP ranking:   %.4f\n", rp_stats.score)
    str.printf("  RP entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
      rp_entropy.mean, rp_entropy.min, rp_entropy.max, rp_entropy.std)
    local expected_neighbors = ivec.create():copy(split.label_csr.neighbors):add(train.n)
    local hood_ids, hoods = label_ann:neighborhoods_by_vecs(predicted_rp, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
    local ks, ret = eval.retrieval_ks({
      hoods = hoods,
      hood_ids = hood_ids,
      expected_offsets = split.label_csr.offsets,
      expected_neighbors = expected_neighbors,
    })
    str.printf("  Retrieval micro: P=%.4f R=%.4f F1=%.4f\n",
      ret.micro_precision, ret.micro_recall, ret.micro_f1)
    str.printf("  Retrieval macro: P=%.4f R=%.4f F1=%.4f\n",
      ret.macro_precision, ret.macro_recall, ret.macro_f1)
    return raw_stats.score, rp_stats.score, ret
  end

  local train_raw, train_rp, train_ret = evaluate_split(train, "train", train_eval_adj)
  local dev_raw, dev_rp, dev_ret = evaluate_split(dev, "dev", train_eval_adj)
  local test_raw, test_rp, test_ret = evaluate_split(test_set, "test", train_eval_adj)

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d  RP bits: %d\n", train.dims, rp_n_bits)
  str.printf("\n  Ranking Accuracy (%s):\n", cfg.eval.ranking)
  str.printf("                          Train      Dev      Test\n")
  str.printf("    Spectral kernel:     %.4f\n", spectral_metrics.kernel_score)
  str.printf("    Spectral raw:        %.4f\n", spectral_metrics.raw_score)
  str.printf("    Spectral SignRP:     %.4f\n", best_score)
  str.printf("    Spectral train raw:  %.4f\n", sp_raw_stats.score)
  str.printf("    Spectral train RP:   %.4f\n", sp_rp_stats.score)
  str.printf("    Predicted raw:       %.4f   %.4f   %.4f\n", train_raw, dev_raw, test_raw)
  str.printf("    Predicted SignRP:    %.4f   %.4f   %.4f\n", train_rp, dev_rp, test_rp)
  str.printf("\n  Entropy:\n")
  str.printf("    Spectral train RP:   mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)
  str.printf("\n  Retrieval (labels-only ANN, k=%d, radius=%d):\n", cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
  str.printf("                          micro P    micro R  micro F1   macro P    macro R  macro F1\n")
  str.printf("    Spectral train:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1,
    sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)
  str.printf("    Predicted train:     %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    train_ret.micro_precision, train_ret.micro_recall, train_ret.micro_f1,
    train_ret.macro_precision, train_ret.macro_recall, train_ret.macro_f1)
  str.printf("    Predicted dev:       %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    dev_ret.micro_precision, dev_ret.micro_recall, dev_ret.micro_f1,
    dev_ret.macro_precision, dev_ret.macro_recall, dev_ret.macro_f1)
  str.printf("    Predicted test:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    test_ret.micro_precision, test_ret.micro_recall, test_ret.micro_f1,
    test_ret.macro_precision, test_ret.macro_recall, test_ret.macro_f1)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)

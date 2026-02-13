local str = require("santoku.string")
local ann = require("santoku.learn.ann")
local eval = require("santoku.learn.evaluator")

local M = {}

function M.spot_check_codes (codes, n_samples, n_dims, label)
  local samples_to_show = math.min(10, n_samples)
  local bits_to_show = math.min(48, n_dims)
  str.printf("  [%s] %d samples × %d dims\n", label, n_samples, n_dims)
  for i = 0, samples_to_show - 1 do
    local start_bit = i * n_dims
    local bits_str = codes:bits_to_ascii(start_bit, start_bit + bits_to_show)
    local full_str = codes:bits_to_ascii(start_bit, start_bit + n_dims)
    local pop = 0
    for j = 1, #full_str do
      if full_str:sub(j, j) == "1" then pop = pop + 1 end
    end
    str.printf("    [%d] %s%s pop=%d/%d (%.0f%%)\n",
      i, bits_str, n_dims > bits_to_show and "..." or "", pop, n_dims, 100 * pop / n_dims)
  end
end

function M.node_type (node_id, n_docs)
  if n_docs and node_id >= n_docs then
    return "label", node_id - n_docs
  else
    return "doc", node_id
  end
end

function M.format_node (node_id, n_docs)
  local ntype, idx = M.node_type(node_id, n_docs)
  if ntype == "label" then
    return str.format("L%d", idx)
  else
    return str.format("D%d", idx)
  end
end

function M.spot_check_adjacency (ids, offsets, neighbors, weights, label, n_docs)
  local n_nodes = ids:size()
  local n_edges = neighbors:size()
  local avg_deg = n_edges / n_nodes
  str.printf("  [%s] %d nodes, %d edges, avg_deg=%.1f\n", label, n_nodes, n_edges, avg_deg)
  local id_min, id_max = ids:min(), ids:max()
  str.printf("    ID range: [%d, %d]", id_min, id_max)
  if n_docs then
    local n_doc_nodes, n_label_nodes = 0, 0
    for i = 0, n_nodes - 1 do
      if ids:get(i) < n_docs then n_doc_nodes = n_doc_nodes + 1
      else n_label_nodes = n_label_nodes + 1 end
    end
    str.printf(" (docs=%d, labels=%d)", n_doc_nodes, n_label_nodes)
  end
  str.printf("\n")
  if weights then
    str.printf("    Weight range: [%.4f, %.4f]\n", weights:min(), weights:max())
  end
  local samples_to_show = math.min(5, n_nodes)
  for i = 0, samples_to_show - 1 do
    local node_id = ids:get(i)
    local s = offsets:get(i)
    local e = offsets:get(i + 1)
    local deg = e - s
    if n_docs then
      local doc_nb, label_nb = {}, {}
      for j = s, e - 1 do
        local nidx = neighbors:get(j)
        local nid = ids:get(nidx)
        if nid < n_docs then
          if #doc_nb < 3 then doc_nb[#doc_nb + 1] = M.format_node(nid, n_docs) end
        else
          if #label_nb < 3 then label_nb[#label_nb + 1] = M.format_node(nid, n_docs) end
        end
        if #doc_nb >= 3 and #label_nb >= 3 then break end
      end
      local neigh_sample = {}
      for _, v in ipairs(doc_nb) do neigh_sample[#neigh_sample + 1] = v end
      for _, v in ipairs(label_nb) do neigh_sample[#neigh_sample + 1] = v end
      str.printf("    %s: deg=%d neighbors=[%s%s]\n",
        M.format_node(node_id, n_docs), deg, table.concat(neigh_sample, ","), deg > 6 and ",..." or "")
    else
      local neigh_sample = {}
      for j = s, math.min(s + 4, e - 1) do
        local nidx = neighbors:get(j)
        local nid = ids:get(nidx)
        neigh_sample[#neigh_sample + 1] = M.format_node(nid, n_docs)
      end
      str.printf("    %s: deg=%d neighbors=[%s%s]\n",
        M.format_node(node_id, n_docs), deg, table.concat(neigh_sample, ","), deg > 5 and ",..." or "")
    end
  end
end

function M.get_doc_labels (doc_id, label_csr, id_offset)
  local idx = doc_id - (id_offset or 0)
  if idx < 0 or idx >= label_csr.offsets:size() - 1 then
    return {}
  end
  local s = label_csr.offsets:get(idx)
  local e = label_csr.offsets:get(idx + 1)
  local labels = {}
  for j = s, e - 1 do
    labels[#labels + 1] = label_csr.neighbors:get(j)
  end
  return labels
end

function M.format_labels (labels, max_show)
  max_show = max_show or 5
  if #labels == 0 then return "[]" end
  local shown = {}
  for i = 1, math.min(max_show, #labels) do
    shown[#shown + 1] = labels[i]
  end
  local suffix = #labels > max_show and str.format("...+%d", #labels - max_show) or ""
  return "[" .. table.concat(shown, ",") .. suffix .. "]"
end

function M.label_overlap (labels1, labels2)
  local set1 = {}
  for _, l in ipairs(labels1) do set1[l] = true end
  local overlap = 0
  for _, l in ipairs(labels2) do
    if set1[l] then overlap = overlap + 1 end
  end
  return overlap
end

function M.spot_check_neighbors_with_labels (ids, offsets, neighbors, weights, label_csr, id_offset, label, check_ids, n_neighbors, n_docs)
  n_neighbors = n_neighbors or 10
  local n_nodes = ids:size()
  local id_to_idx = {}
  for i = 0, n_nodes - 1 do
    id_to_idx[ids:get(i)] = i
  end
  local per_type = math.max(1, math.floor(n_neighbors / 2))
  str.printf("  [%s] Spot-checking %d nodes (up to %d docs + %d labels each):\n", label, #check_ids, per_type, per_type)
  for _, node_id in ipairs(check_ids) do
    local idx = id_to_idx[node_id]
    if idx then
      local ntype, nidx_local = M.node_type(node_id, n_docs)
      local s = offsets:get(idx)
      local e = offsets:get(idx + 1)
      local deg = e - s
      local n_doc_nb, n_label_nb = 0, 0
      if n_docs then
        for j = s, e - 1 do
          local nid = ids:get(neighbors:get(j))
          if nid < n_docs then n_doc_nb = n_doc_nb + 1
          else n_label_nb = n_label_nb + 1 end
        end
      end
      if ntype == "label" then
        str.printf("    Label %d (deg=%d: %d docs, %d labels):\n", nidx_local, deg, n_doc_nb, n_label_nb)
      else
        local doc_labels = M.get_doc_labels(node_id, label_csr, id_offset)
        str.printf("    Doc %d %s (deg=%d: %d docs, %d labels):\n", node_id, M.format_labels(doc_labels), deg, n_doc_nb, n_label_nb)
      end
      local doc_shown, label_shown = 0, 0
      for j = s, e - 1 do
        local neighbor_idx = neighbors:get(j)
        local neighbor_id = ids:get(neighbor_idx)
        local neighbor_type, neighbor_local = M.node_type(neighbor_id, n_docs)
        local w_str = weights and str.format(" w=%.3f", weights:get(j)) or ""
        if neighbor_type == "label" then
          if label_shown < per_type then
            str.printf("      -> L%d%s\n", neighbor_local, w_str)
            label_shown = label_shown + 1
          end
        else
          if doc_shown < per_type then
            local n_labels = M.get_doc_labels(neighbor_id, label_csr, id_offset)
            if ntype == "doc" then
              local doc_labels = M.get_doc_labels(node_id, label_csr, id_offset)
              local overlap = M.label_overlap(doc_labels, n_labels)
              str.printf("      -> D%d %s overlap=%d/%d%s\n",
                neighbor_id, M.format_labels(n_labels), overlap, #doc_labels, w_str)
            else
              str.printf("      -> D%d %s%s\n", neighbor_id, M.format_labels(n_labels), w_str)
            end
            doc_shown = doc_shown + 1
          end
        end
        if doc_shown >= per_type and label_shown >= per_type then break end
      end
    else
      str.printf("    Node %d: not found in index\n", node_id)
    end
  end
end

local function format_phase (ev)
  if ev.is_final then return "F" end
  local tag = ev.phase or "lhs"
  return str.format("%s %d/%d", tag, ev.trial or 1, ev.trials or 1)
end

local function format_best (best, current)
  if not best or best == -math.huge then
    return ""
  end
  if current and current > best + 1e-6 then
    return str.format(" (best=%.4f ++)", current)
  else
    return str.format(" (best=%.4f)", best)
  end
end

function M.spectral_log (info)
  if info.event == "sample" then
    str.printf("[SPECTRAL] L=%d D=%d decay=%.2f\n", info.n_landmarks, info.n_dims, info.decay)
  elseif info.event == "spectral_result" then
    local trace_info = info.trace_ratio and str.format(" trace=%.2e", info.trace_ratio) or ""
    local eig_info = (info.eig_min and info.eig_max) and str.format(" eig=[%.4f, %.4f]", info.eig_min, info.eig_max) or ""
    local dims_info = info.n_dims and str.format(" D=%d", info.n_dims) or ""
    str.printf("[SPECTRAL]   -> L=%d%s%s%s\n",
      info.n_landmarks or 0, dims_info, eig_info, trace_info)
  elseif info.event == "eval" then
    local m = info.metrics or {}
    local scores = {}
    if m.kernel_score then scores[#scores + 1] = str.format("kernel=%.4f", m.kernel_score) end
    scores[#scores + 1] = str.format("raw=%.4f", info.score or 0)
    if m.binary_score then scores[#scores + 1] = str.format("bin=%.4f", m.binary_score) end
    str.printf("[SPECTRAL]   -> eval: %s\n", table.concat(scores, " "))
  elseif info.event == "done" then
    print(string.rep("-", 50))
    local p = info.best_params or {}
    local m = info.best_metrics or {}
    local scores = {}
    if m.kernel_score then scores[#scores + 1] = str.format("kernel=%.4f", m.kernel_score) end
    if info.best_score then scores[#scores + 1] = str.format("raw=%.4f", info.best_score) end
    if m.binary_score then scores[#scores + 1] = str.format("bin=%.4f", m.binary_score) end
    str.printf("[SPECTRAL] DONE: L=%d D=%d decay=%.2f %s\n",
      p.n_landmarks or 0, p.n_dims or 0, p.decay or 0, table.concat(scores, " "))
  end
end

function M.make_bits_log (log_interval)
  log_interval = log_interval or 8
  local last_logged = 0
  local n_bits = 0
  return function (_, gain, score, action)
    if action == "add" then
      n_bits = n_bits + 1
      if n_bits - last_logged >= log_interval or n_bits <= 1 then
        str.printf("[BITS] %d bits: score=%.4f (+%.4f)\n", n_bits, score, gain)
        last_logged = n_bits
      end
    elseif action == "remove" then
      n_bits = n_bits - 1
    end
  end
end

function M.make_classifier_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local params = ev.params
    local metrics = ev.metrics
    local running_best = math.max(ev.global_best_score or -math.huge, ev.best_epoch_score or -math.huge)
    local best = ev.is_final and "" or format_best(running_best, metrics.f1)
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    local absorb = ""
    if params.absorb_interval then
      absorb = str.format(" AI=%d A=%d/%d/%d", params.absorb_interval, params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum or 0)
    end
    str.printf("[CLASSIFY %s E%d] C=%d L=%d/%d T=%d S=%.0f%s F1=%.4f%s%s\n",
      phase, ev.epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, absorb, metrics.f1, best, timing)
  end
end

function M.make_regressor_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local params = ev.params
    local metrics = ev.metrics
    local mae = metrics.mean
    local running_best = math.max(ev.global_best_score or -math.huge, ev.best_epoch_score or -math.huge)
    local running_best_mae = (running_best > -math.huge) and -running_best or nil
    local best = ""
    if not ev.is_final and running_best_mae then
      local marker = (mae < running_best_mae - 1e-6) and " ++" or ""
      best = str.format(" (best=%.4f%s)", running_best_mae, marker)
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    local absorb = ""
    if params.absorb_interval then
      absorb = str.format(" AI=%d A=%d/%d/%d", params.absorb_interval, params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum or 0)
    end
    str.printf("[REGRESS %s E%d] C=%d L=%d/%d T=%d S=%.0f%s MAE=%.4f%s%s\n",
      phase, ev.epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, absorb, mae, best, timing)
  end
end

function M.make_regressor_acc_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local params = ev.params
    local metrics = ev.metrics
    local acc = (1 - metrics.nmae) * 100
    local running_best = math.max(ev.global_best_score or -math.huge, ev.best_epoch_score or -math.huge)
    local running_best_acc = (running_best > -math.huge) and (1 + running_best) * 100 or nil
    local best = ""
    if not ev.is_final and running_best_acc then
      local marker = (acc > running_best_acc + 1e-4) and " ++" or ""
      best = str.format(" (best=%.1f%%%s)", running_best_acc, marker)
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    local absorb = ""
    if params.absorb_interval then
      absorb = str.format(" AI=%d A=%d/%d/%d", params.absorb_interval, params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum or 0)
    end
    str.printf("[REGRESS %s E%d] C=%d L=%d/%d T=%d S=%.0f%s ACC=%.1f%%%s%s\n",
      phase, ev.epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, absorb, acc, best, timing)
  end
end

function M.make_ranking_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local params = ev.params
    local metrics = ev.metrics
    local score = metrics.score or 0
    local running_best = math.max(ev.global_best_score or -math.huge, ev.best_epoch_score or -math.huge)
    local best = ev.is_final and "" or format_best(running_best, score)
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    local absorb = ""
    if params.absorb_interval then
      absorb = str.format(" AI=%d A=%d/%d/%d", params.absorb_interval, params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum or 0)
    end
    str.printf("[RANKING %s E%d] C=%d L=%d/%d T=%d S=%.0f%s score=%.4f%s%s\n",
      phase, ev.epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, absorb, score, best, timing)
  end
end

function M.cluster_stats (args)
  local codes = args.codes
  local ids = args.ids
  local n_dims = args.n_dims
  local knn = args.knn or 16
  local eval_offsets = args.eval_offsets
  local eval_neighbors = args.eval_neighbors
  local eval_weights = args.eval_weights
  local label = args.label or "codes"
  local codes_ann = ann.create({ features = n_dims })
  codes_ann:add(codes, ids)
  local adj_ids, adj_hoods = codes_ann:neighborhoods(knn)
  local adj_offsets, adj_neighbors, _ = adj_hoods:to_csr(adj_ids, n_dims)
  local result = eval.cluster({
    codes = codes,
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    n_dims = n_dims,
    expected_offsets = eval_offsets,
    expected_neighbors = eval_neighbors,
    expected_weights = eval_weights,
  })
  local nc = result.n_clusters_curve
  local rc = result.radius_curve
  local ac = result.auc_curve
  local nc_min, nc_max = nc:get(0), nc:get(nc:size() - 1)
  local rc_min, rc_max = rc:min(), rc:max()
  local rc_elbow_val, rc_elbow_idx = rc:scores_elbow("lmethod")
  local rc_elbow_nc = nc:get(rc_elbow_idx - 1)
  str.printf("  [%s] clusters: %d→%d, radius: %.4f→%.4f (elbow: %.4f @%d clusters)",
    label, nc_min, nc_max, rc_min, rc_max, rc_elbow_val, rc_elbow_nc)
  if ac then
    local ac_min, ac_max = ac:min(), ac:max()
    local ac_elbow_val, ac_elbow_idx = ac:scores_elbow("lmethod")
    local ac_elbow_nc = nc:get(ac_elbow_idx - 1)
    str.printf(", auc: %.4f→%.4f (elbow: %.4f @%d)", ac_min, ac_max, ac_elbow_val, ac_elbow_nc)
  end
  str.printf("\n")
  return result
end

return M

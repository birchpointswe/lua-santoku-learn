local str = require("santoku.string")

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

function M.spot_check_adjacency (ids, offsets, neighbors, weights, label)
  local n_nodes = ids:size()
  local n_edges = neighbors:size()
  local avg_deg = n_edges / n_nodes
  str.printf("  [%s] %d nodes, %d edges, avg_deg=%.1f\n", label, n_nodes, n_edges, avg_deg)
  local id_min, id_max = ids:min(), ids:max()
  str.printf("    ID range: [%d, %d]\n", id_min, id_max)
  if weights then
    str.printf("    Weight range: [%.4f, %.4f]\n", weights:min(), weights:max())
  end
  local samples_to_show = math.min(5, n_nodes)
  for i = 0, samples_to_show - 1 do
    local node_id = ids:get(i)
    local s = offsets:get(i)
    local e = offsets:get(i + 1)
    local deg = e - s
    local neigh_sample = {}
    for j = s, math.min(s + 4, e - 1) do
      local nidx = neighbors:get(j)
      local nid = ids:get(nidx)
      neigh_sample[#neigh_sample + 1] = nid
    end
    str.printf("    doc %d: deg=%d neighbors=[%s%s]\n",
      node_id, deg, table.concat(neigh_sample, ","), deg > 5 and ",..." or "")
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

function M.spot_check_neighbors_with_labels (ids, offsets, neighbors, weights, label_csr, id_offset, label, check_ids, n_neighbors)
  n_neighbors = n_neighbors or 10
  local n_nodes = ids:size()
  local id_to_idx = {}
  for i = 0, n_nodes - 1 do
    id_to_idx[ids:get(i)] = i
  end
  str.printf("  [%s] Spot-checking %d docs with up to %d neighbors:\n", label, #check_ids, n_neighbors)
  for _, doc_id in ipairs(check_ids) do
    local idx = id_to_idx[doc_id]
    if idx then
      local doc_labels = M.get_doc_labels(doc_id, label_csr, id_offset)
      local s = offsets:get(idx)
      local e = offsets:get(idx + 1)
      local deg = e - s
      str.printf("    Doc %d %s (deg=%d):\n", doc_id, M.format_labels(doc_labels), deg)
      for j = s, math.min(s + n_neighbors - 1, e - 1) do
        local nidx = neighbors:get(j)
        local nid = ids:get(nidx)
        local n_labels = M.get_doc_labels(nid, label_csr, id_offset)
        local overlap = M.label_overlap(doc_labels, n_labels)
        local w_str = weights and str.format(" w=%.3f", weights:get(j)) or ""
        str.printf("      -> %d %s overlap=%d/%d%s\n",
          nid, M.format_labels(n_labels), overlap, #doc_labels, w_str)
      end
    else
      str.printf("    Doc %d: not found in index\n", doc_id)
    end
  end
end

local function format_best_cmp (score, best)
  if not best or best == -math.huge then
    return ""
  end
  if score > best + 1e-6 then
    return str.format(" (+%.4f new best)", score - best)
  else
    return str.format(" (best=%.4f)", best)
  end
end

function M.spectral_log (info)
  if info.event == "round_end" then
    local p = info.best_params or {}
    local m = info.best_metrics or {}
    local skipped = (info.trials or 0) - (info.round_samples or 0)
    local stats = str.format(" | %d/%d improved", info.round_improvements or 0, info.round_samples or 0)
    if skipped > 0 then
      stats = stats .. str.format(", %d skipped", skipped)
    end
    if info.adapt_factor and info.adapt_factor ~= 1.0 then
      stats = stats .. str.format(", jitter×%.2f", info.adapt_factor)
    end
    if m.kernel_score then
      str.printf("[SPECTRAL R%d/%d] best: kernel=%.4f raw=%.4f L=%d D=%d decay=%.2f%s\n",
        info.round, info.rounds or 0, m.kernel_score, info.global_best_score,
        p.n_landmarks or 0, p.n_dims or 0, p.decay or 0, stats)
    else
      str.printf("[SPECTRAL R%d/%d] best: raw=%.4f L=%d D=%d decay=%.2f%s\n",
        info.round, info.rounds or 0, info.global_best_score,
        p.n_landmarks or 0, p.n_dims or 0, p.decay or 0, stats)
    end
  elseif info.event == "sample" then
    if info.round and info.trial then
      str.printf("[SPECTRAL R%d/%d T%d/%d] L=%d D=%d decay=%.2f\n",
        info.round, info.rounds or 0, info.trial, info.trials or 0,
        info.n_landmarks, info.n_dims, info.decay)
    else
      str.printf("[SPECTRAL] L=%d D=%d decay=%.2f\n", info.n_landmarks, info.n_dims, info.decay)
    end
  elseif info.event == "spectral_result" then
    local trace_info = info.trace_ratio and str.format(" trace=%.2e", info.trace_ratio) or ""
    str.printf("[SPECTRAL]   -> L=%d eig=[%.4f, %.4f]%s\n",
      info.n_landmarks or 0, info.eig_min, info.eig_max, trace_info)
  elseif info.event == "eval" then
    local m = info.metrics or {}
    local cmp = format_best_cmp(info.score, info.global_best_score)
    if m.kernel_score then
      str.printf("[SPECTRAL]   -> eval: kernel=%.4f raw=%.4f%s\n",
        m.kernel_score, info.score, cmp)
    else
      str.printf("[SPECTRAL]   -> eval: raw=%.4f%s\n", info.score, cmp)
    end
  elseif info.event == "done" then
    print(string.rep("-", 50))
    local p = info.best_params or {}
    local m = info.best_metrics or {}
    str.printf("[SPECTRAL] DONE: L=%d D=%d decay=%.2f",
      p.n_landmarks or 0, p.n_dims or 0, p.decay or 0)
    if m.kernel_score and info.best_score then
      str.printf(" kernel=%.4f raw=%.4f", m.kernel_score, info.best_score)
    elseif info.best_score then
      str.printf(" raw=%.4f", info.best_score)
    end
    print()
  end
end

function M.encoder_log (_, is_final, metrics, params, epoch, round, trial, rounds, trials, global_best_score)
  local phase
  if is_final then
    phase = "F"
  elseif rounds and trials then
    phase = str.format("R%d/%d T%d/%d", round, rounds, trial, trials)
  else
    phase = str.format("R%d T%d", round, trial)
  end
  local score = metrics.score or metrics.mean_hamming
  local cmp = ""
  if not is_final and global_best_score and global_best_score > -math.huge then
    if score > global_best_score + 1e-6 then
      cmp = str.format(" (+%.4f new best)", score - global_best_score)
    else
      cmp = str.format(" (best=%.4f)", global_best_score)
    end
  end
  local bit_stats = ""
  if metrics.ber_min and metrics.ber_max and metrics.ber_std then
    bit_stats = str.format(" bits=[%.2f,%.2f] std=%.3f", metrics.ber_max, metrics.ber_min, metrics.ber_std)
  end
  str.printf("[ENCODER %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d score=%.4f%s%s\n",
    phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
    params.target, params.specificity, params.include_bits, score, bit_stats, cmp)
end

function M.make_classifier_log (stopwatch)
  return function (_, is_final, metrics, params, epoch, round, trial, rounds, trials, global_best_score)
    local phase
    if is_final then
      phase = "F"
    elseif rounds and trials then
      phase = str.format("R%d/%d T%d/%d", round, rounds, trial, trials)
    else
      phase = str.format("R%d T%d", round, trial)
    end
    local cmp = ""
    if not is_final and global_best_score and global_best_score > -math.huge then
      if metrics.f1 > global_best_score + 1e-6 then
        cmp = str.format(" (+%.2f new best)", metrics.f1 - global_best_score)
      else
        cmp = str.format(" (best=%.2f)", global_best_score)
      end
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    str.printf("[CLASSIFY %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d F1=%.2f%s%s\n",
      phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, params.include_bits, metrics.f1, cmp, timing)
  end
end

function M.make_regressor_log (stopwatch)
  return function (_, is_final, metrics, params, epoch, round, trial, rounds, trials, global_best_score)
    local phase
    if is_final then
      phase = "F"
    elseif rounds and trials then
      phase = str.format("R%d/%d T%d/%d", round, rounds, trial, trials)
    else
      phase = str.format("R%d T%d", round, trial)
    end
    local mae = metrics.mean
    local cmp = ""
    if not is_final and global_best_score and global_best_score > -math.huge then
      local best_mae = -global_best_score
      if mae < best_mae - 1e-6 then
        cmp = str.format(" (%.4f new best)", mae)
      else
        cmp = str.format(" (best=%.4f)", best_mae)
      end
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    str.printf("[REGRESS %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d MAE=%.4f%s%s\n",
      phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, params.include_bits, mae, cmp, timing)
  end
end

function M.make_regressor_acc_log (stopwatch)
  return function (_, is_final, metrics, params, epoch, round, trial, rounds, trials, global_best_score)
    local phase
    if is_final then
      phase = "F"
    elseif rounds and trials then
      phase = str.format("R%d/%d T%d/%d", round, rounds, trial, trials)
    else
      phase = str.format("R%d T%d", round, trial)
    end
    local acc = (1 - metrics.nmae) * 100
    local cmp = ""
    if not is_final and global_best_score and global_best_score > -math.huge then
      local best_acc = (1 + global_best_score) * 100
      if acc > best_acc + 1e-4 then
        cmp = str.format(" (+%.1f new best)", acc - best_acc)
      else
        cmp = str.format(" (best=%.1f%%)", best_acc)
      end
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    str.printf("[REGRESS %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d ACC=%.1f%%%s%s\n",
      phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
      params.target, params.specificity, params.include_bits, acc, cmp, timing)
  end
end

return M

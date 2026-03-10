local str = require("santoku.string")
local ann = require("santoku.learn.ann")
local eval = require("santoku.learn.evaluator")

local M = {}

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
    if params.absorb_threshold then
      absorb = str.format(" A=%d/%d/%.3f R=%.3f", params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum_fraction or 0, params.absorb_ranking_fraction or 0)
    end
    local lt, lm, tt, ss
    if params.alpha_specificity then
      lt = str.format("(%+.1f)", params.alpha_tolerance)
      lm = str.format("(%+.1f)", params.alpha_maximum)
      tt = str.format("(%+.1f)", params.alpha_target)
      ss = str.format("(%+.1f)", params.alpha_specificity)
    else
      lt, lm, tt, ss = "", "", "", ""
    end
    local skip = params.flat_skip and str.format(" SK=%.2f", params.flat_skip) or ""
    str.printf("[REGRESS %s E%d] F=%d C=%d L=%.2f%s/%.3f%s T=%.2f%s S=%.4f%s%s%s MAE=%.4f%s%s\n",
      phase, ev.epoch, params.features, params.clauses, params.clause_tolerance_fraction, lt, params.clause_maximum_fraction, lm,
      params.target_fraction, tt, params.specificity_fraction, ss, absorb, skip, mae, best, timing)
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
    if params.absorb_threshold then
      absorb = str.format(" A=%d/%d/%.3f R=%.3f", params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum_fraction or 0, params.absorb_ranking_fraction or 0)
    end
    local lt, lm, tt, ss
    if params.alpha_specificity then
      lt = str.format("(%+.1f)", params.alpha_tolerance)
      lm = str.format("(%+.1f)", params.alpha_maximum)
      tt = str.format("(%+.1f)", params.alpha_target)
      ss = str.format("(%+.1f)", params.alpha_specificity)
    else
      lt, lm, tt, ss = "", "", "", ""
    end
    local skip = params.flat_skip and str.format(" SK=%.2f", params.flat_skip) or ""
    str.printf("[REGRESS %s E%d] F=%d C=%d L=%.2f%s/%.3f%s T=%.2f%s S=%.4f%s%s%s ACC=%.1f%%%s%s\n",
      phase, ev.epoch, params.features, params.clauses, params.clause_tolerance_fraction, lt, params.clause_maximum_fraction, lm,
      params.target_fraction, tt, params.specificity_fraction, ss, absorb, skip, acc, best, timing)
  end
end

function M.make_labeler_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local params = ev.params
    local metrics = ev.metrics
    local micro = metrics.micro_f1 or 0
    local macro = metrics.sample_f1 or 0
    local running_best = math.max(ev.global_best_score or -math.huge, ev.best_epoch_score or -math.huge)
    local best = ev.is_final and "" or format_best(running_best, micro)
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.2fs +%.2fs)", d, dd)
    end
    local absorb = ""
    if params.absorb_threshold then
      absorb = str.format(" A=%d/%d/%.3f R=%.3f", params.absorb_threshold or 0, params.absorb_insert or 0, params.absorb_maximum_fraction or 0, params.absorb_ranking_fraction or 0)
    end
    local lt, lm, tt, ss
    if params.alpha_specificity then
      lt = str.format("(%+.1f)", params.alpha_tolerance)
      lm = str.format("(%+.1f)", params.alpha_maximum)
      tt = str.format("(%+.1f)", params.alpha_target)
      ss = str.format("(%+.1f)", params.alpha_specificity)
    else
      lt, lm, tt, ss = "", "", "", ""
    end
    local skip = params.flat_skip and str.format(" SK=%.2f", params.flat_skip) or ""
    str.printf("[LABEL %s E%d] F=%d C=%d L=%.2f%s/%.3f%s T=%.2f%s S=%.4f%s%s%s miF1=%.4f saF1=%.4f%s%s\n",
      phase, ev.epoch, params.features, params.clauses, params.clause_tolerance_fraction, lt, params.clause_maximum_fraction, lm,
      params.target_fraction, tt, params.specificity_fraction, ss, absorb, skip, micro, macro, best, timing)
  end
end

function M.make_ridge_log (stopwatch)
  return function (ev)
    local phase = format_phase(ev)
    local p = ev.params or {}
    local m = ev.metrics or {}
    local score = ev.score or 0
    local best = format_best(ev.global_best_score, score)
    local detail = ""
    if m.mae then
      detail = str.format(" mae=%.6f", m.mae)
    elseif m.oracle then
      detail = str.format(" saF1=%.4f miF1=%.4f", m.oracle.sample_f1, m.oracle.micro_f1)
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.1fs +%.1fs)", d, dd)
    end
    local prop = ""
    if p.propensity_a then
      prop = str.format(" pa=%.2f pb=%.2f", p.propensity_a, p.propensity_b)
    end
    local mode = (p.elm or p.mode) and str.format(" mode=%s", p.elm or p.mode) or ""
    str.printf("[Ridge %s]%s lambda=%.4e%s score=%.4f%s%s%s\n",
      phase, mode, p.lambda or 0, prop, score, detail, best, timing)
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
  local mih = ann.create({ data = codes, features = n_dims })
  local adj_offsets, adj_neighbors = mih:neighborhoods(knn)
  local result = eval.cluster({
    codes = codes,
    ids = ids,
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
  str.printf("[%s] clusters: %d→%d, radius: %.4f→%.4f (elbow: %.4f @%d clusters)",
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

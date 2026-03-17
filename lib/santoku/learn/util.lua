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

function M.make_ridge_log (stopwatch, metric_fmt)
  return function (ev)
    local phase = format_phase(ev)
    local p = ev.params or {}
    local m = ev.metrics or {}
    local score = ev.score or 0
    local best = format_best(ev.global_best_score, score)
    local detail = ""
    if metric_fmt then
      detail = " " .. metric_fmt(m)
    elseif m.mae then
      detail = str.format(" mae=%.6f", m.mae)
    elseif m.gfm_f1 then
      detail = str.format(" oracle=%.4f", m.oracle.micro_f1)
    elseif m.oracle then
      detail = str.format(" saF1=%.4f miF1=%.4f", m.oracle.sample_f1, m.oracle.micro_f1)
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.1fs +%.1fs)", d, dd)
    end
    local kern = ""
    if p.kernel then
      kern = str.format(" kernel=%s", p.kernel)
    end
    local prop = ""
    if p.propensity_a then
      prop = str.format(" pa=%.2f pb=%.2f", p.propensity_a, p.propensity_b)
    end
    str.printf("[Ridge %s]%s lambda=%.4e%s score=%.4f%s%s%s\n",
      phase, kern, p.lambda or 0, prop, score, detail, best, timing)
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

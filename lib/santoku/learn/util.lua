local str = require("santoku.string")

local M = {}

-- Uniform one-line formatting of a decode metrics table, across decode types:
--   multilabel  (micro_f1/precision/recall, or the bundle's f1/precision/recall)  -> miF1/miP/miR
--   single      (macro_f1/accuracy)                                               -> maF1/acc
function M.fmt_metrics (m)
  if m.micro_f1 ~= nil then
    return str.format("miF1=%.4f miP=%.4f miR=%.4f", m.micro_f1, m.micro_precision, m.micro_recall)
  elseif m.precision ~= nil and m.f1 ~= nil then
    return str.format("miF1=%.4f miP=%.4f miR=%.4f", m.f1, m.precision, m.recall)
  elseif m.macro_f1 ~= nil then
    return str.format("maF1=%.4f acc=%.4f", m.macro_f1, m.accuracy or 0)
  elseif m.span_f1 ~= nil then
    return str.format("spF1=%.4f P=%.4f R=%.4f", m.span_f1, m.precision, m.recall)
  end
  return "?"
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

function M.make_ridge_log (stopwatch, metric_fmt)
  return function (ev)
    if ev.event == "done" then
      local p = ev.params or {}
      local emb = ev.emb_d and str.format(" emb_d=%d", ev.emb_d) or ""
      local md = p.mode and str.format(" mode=%s", p.mode) or ""
      local kern = p.kernel and str.format(" kernel=%s", p.kernel) or ""
      local act = p.activation and str.format(" act=%s", p.activation) or ""
      local gam = p.gamma and str.format(" gamma=%.4g", p.gamma) or ""
      local solve = ev.solve and str.format(" solve=%s", ev.solve) or ""
      local prop = ""
      if p.propensity_a then
        prop = str.format(" pa=%.4f pb=%.4f", p.propensity_a, p.propensity_b)
      end
      local sc = ev.score and str.format(" score=%.4f", ev.score) or ""
      local timing = ""
      if stopwatch then
        local d, dd = stopwatch()
        timing = str.format(" (%.1fs +%.1fs)", d, dd)
      end
      str.printf("[Ridge Done]%s%s%s%s%s%s lambda=%.4e%s%s%s\n",
        emb, md, kern, act, gam, solve, p.lambda or 0, prop, sc, timing)
      return
    end
    local phase = format_phase(ev)
    local p = ev.params or {}
    local m = ev.metrics or {}
    local score = ev.score or 0
    local best = format_best(ev.global_best_score, score)
    local detail = ""
    if metric_fmt then
      detail = " " .. metric_fmt(m)
    elseif m.mae then
      detail = str.format(" mae=%.6f nmae=%.4f", m.mae, m.nmae)
    elseif m.ndcg then
      detail = str.format(" NDCG=%.4f", m.ndcg)
    elseif m.recall and not m.f1 then
      detail = str.format(" R=%.4f", m.recall)
    elseif m.f1 then
      detail = str.format(" F1=%.4f P=%.4f R=%.4f", m.f1, m.precision, m.recall)
    end
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.1fs +%.1fs)", d, dd)
    end
    local kern = ""
    if p.kernel then
      kern = str.format(" kernel=%s", p.kernel)
    elseif p.activation then
      kern = str.format(" act=%s", p.activation)
    end
    local gam = p.gamma and str.format(" gamma=%.4g", p.gamma) or ""
    -- In a nested BO the outer (kernel) trial carries lambda/propensity in its metrics (the inner
    -- winner), not its params; fall back to metrics so the outer line shows the inner winner.
    local lambda = p.lambda or m.lambda or 0
    local pa = p.propensity_a or m.propensity_a
    local pb = p.propensity_b or m.propensity_b
    local prop = ""
    if pa then
      prop = str.format(" pa=%.2f pb=%.2f", pa, pb)
    end
    str.printf("[Ridge %s]%s%s lambda=%.4e%s score=%.4f%s%s%s\n",
      phase, kern, gam, lambda, prop, score, detail, best, timing)
  end
end


return M

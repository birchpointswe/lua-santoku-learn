local tm = require("santoku.tsetlin.capi")
local num = require("santoku.num")
local arr = require("santoku.array")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")

local function wrap (t)
  return {
    train = function (...)
      return tm.train(t, ...)
    end,
    predict = function (...)
      return tm.predict(t, ...)
    end,
    destroy = function (...)
      return tm.destroy(t, ...)
    end,
    persist = function (...)
      return tm.persist(t, ...)
    end,
    type = function (...)
      return tm.type(t, ...)
    end,
  }
end

local M = {}

M.load = function (...)
  return wrap(tm.load(...))
end

M.classifier = function (...)
  return wrap(tm.create("classifier", ...))
end

M.encoder = function (...)
  return wrap(tm.create("encoder", ...))
end

M.align = tm.align

M.optimize_classifier = function ()
  -- TODO
  return err.error("unimplemented", "optimize_classifier")
end

local function build_sampler (spec, global_dev, global_shrink)
  if type(spec) == "number" then
    return {
      type = "fixed",
      center = spec,
      sample = function ()
        return spec
      end
    }
  end
  if type(spec) == "table" and spec.def ~= nil then
    local def, minv, maxv = spec.def, spec.min, spec.max
    err.assert(def and minv and maxv, "range spec missing def|min|max")
    local is_log = not not spec.log
    local is_int = not not spec.int
    local span, base_center
    if is_log then
      span = math.log(maxv) - math.log(minv)
      base_center = math.log(def)
    else
      span = math.min(def - minv, maxv - def)
      base_center = def
    end
    local shrinker
    if type(spec.shrink) == "function" then
      shrinker = spec.shrink
    else
      local factor = (spec.shrink ~= nil) and spec.shrink or (global_shrink or 0.5)
      shrinker = function (v)
        return v * factor
      end
    end
    local jitter = (spec.dev or global_dev or 0.5) * span
    local round = 0
    return {
      type = "range",
      center = def,
      sample = function (center)
        local c = center and (is_log and math.log(center) or center) or base_center
        local x = rand.fast_normal(c, jitter)
        if is_log then x = math.exp(x) end
        if is_int then x = num.floor(x + 0.5) end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        return x
      end,
      shrink = function ()
        round = round + 1
        jitter = shrinker(jitter, round)
      end,
    }
  end
  if type(spec) == "table" and #spec > 0 then
    return {
      type = "list",
      center = spec[1],
      sample = function ()
        return spec[num.random(#spec)]
      end,
    }
  end
  err.error("Bad hyper‑parameter specification: " .. tostring(spec))
end

M.optimize_encoder = function (args)

  local patience = args.search_patience or 0
  local use_early_stop = patience > 0
  local rounds = args.search_rounds or 3
  local trials = args.search_trials or 10
  local iters_search = args.search_iterations or 10
  local global_dev = args.search_dev or 0.5
  local global_shrink = args.search_shrink or 0.5
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local param_names = { "clauses", "target", "specificity" }

  local samplers = {}
  for _, pname in ipairs(param_names) do
    samplers[pname] = build_sampler(args[pname], global_dev, global_shrink)
  end

  local all_fixed = true
  for _, s in pairs(samplers) do
    if s.type ~= "fixed" then
      all_fixed = false
      break
    end
  end

  local function sample_params ()
    local p = {}
    for name, s in pairs(samplers) do
      if s.type == "range" then
        p[name] = s.sample(s.center)
      else
        p[name] = s.sample()
      end
    end
    return p
  end

  local best_score = -num.huge
  local best_params  = nil
  local best_metrics = nil

  if all_fixed then
    best_params = sample_params()
  else

    for r = 1, rounds do

      local seen = {}
      local round_best_score  = -num.huge
      local round_best_params = nil

      for t = 1, trials do
        local params = sample_params()
        local key = str.format("%d|%f|%f", params.clauses, params.target, params.specificity)
        if not seen[key] then
          seen[key] = true
          local enc = M.encoder({
            visible = args.visible,
            hidden = args.hidden,
            clauses = params.clauses,
            target = params.target,
            specificity = params.specificity,
          })
          local best_epoch_score = -num.huge
          local best_epoch_metrics = nil
          local epochs_since_improve = 0
          enc.train({
            sentences  = args.sentences,
            codes = args.codes,
            samples = args.samples,
            iterations = iters_search,
            each = function (epoch)
              local score, metrics = metric_fn(enc)
              if score > best_epoch_score + 1e-8 then
                best_epoch_score, best_epoch_metrics = score, metrics
                epochs_since_improve = 0
              else
                epochs_since_improve = epochs_since_improve + 1
              end
              if use_early_stop and epochs_since_improve >= patience then
                return false
              end
              if each_cb then
                return each_cb(enc, false, metrics, params, epoch, r, t)
              end
            end,
          })
          local trial_score = best_epoch_score
          local trial_metrics = best_epoch_metrics
          if trial_score > round_best_score then
            round_best_score = trial_score
            round_best_params = params
          end
          if trial_score > best_score then
            best_score, best_params, best_metrics =
              trial_score, params, trial_metrics
          end
        end
      end

      for _, pname in ipairs(param_names) do
        local s = samplers[pname]
        if s.type == "range" then
          s.center = round_best_params[pname]
          if s.shrink then
            s.shrink()
          end
        end
      end

      collectgarbage("collect")
    end
  end

  local final_iters = args.final_iterations or (iters_search * 10)
  local final_best_score = -num.huge
  local final_enc = M.encoder({
    visible = args.visible,
    hidden = args.hidden,
    clauses = best_params.clauses,
    target = best_params.target,
    specificity = best_params.specificity,
  })
  final_enc.train({
    sentences  = args.sentences,
    codes = args.codes,
    samples = args.samples,
    iterations = final_iters,
    each = function (epoch)
      if each_cb then
        local score, metrics = metric_fn(final_enc)
        return each_cb(final_enc, true, metrics, best_params, epoch)
      end
    end,
  })

  return final_enc

end

return M

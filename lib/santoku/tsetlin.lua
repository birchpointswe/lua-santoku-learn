local tm = require("santoku.tsetlin.capi")
local num = require("santoku.num")
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

local function build_sampler (spec, global_dev)
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
      span = math.max(def - minv, maxv - def)
      base_center = def
    end
    local jitter = (spec.dev or global_dev or 1.0) * span
    local round = 0
    return {
      type = "range",
      center = def,
      sample = function (center)
        local c = center and (is_log and math.log(center) or center) or base_center
        local x = rand.fast_normal(c, jitter)
        if is_log then x = math.exp(x) end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        if is_int then x = num.floor(x + 0.5) end
        return x
      end,
      shrink = function ()
        round  = round + 1
        jitter = jitter / math.sqrt(round)
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

M.optimize = function (args, typ)

  local patience = args.search_patience or 0
  local use_early_stop = patience > 0
  local rounds = args.search_rounds or 3
  local trials = args.search_trials or 10
  local iters_search = args.search_iterations or 10
  local global_dev = args.search_dev or 1.0
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local param_names = { "clauses", "target", "specificity" }

  local samplers = {}
  for _, pname in ipairs(param_names) do
    samplers[pname] = build_sampler(args[pname], global_dev)
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
  local best_params = nil

  if all_fixed then
    best_params = sample_params()
  else

    for r = 1, rounds do

      local seen = {}
      local round_best_score  = -num.huge
      local round_best_params = nil

      for t = 1, trials do
        local params = sample_params()
        local key = str.format("%d|%d|%d",
          params.clauses,
          num.floor(params.target * 1e6 + 0.5),
          num.floor(params.specificity * 1e6 + 0.5))
        if not seen[key] then
          seen[key] = true

          local best_epoch_score = -num.huge
          local epochs_since_improve = 0

          local function each (epoch, tm)
            local score, metrics = metric_fn(tm)
            local cb_result = nil
            if each_cb then
              cb_result = each_cb(tm, false, metrics, params, epoch, r, t)
            end
            if score > best_epoch_score + 1e-8 then
              best_epoch_score = score
              epochs_since_improve = 0
            else
              epochs_since_improve = epochs_since_improve + 1
            end
            if use_early_stop and epochs_since_improve >= patience then
              return false
            end
            return cb_result
          end

          if typ == "encoder" then

            local encoder = M.encoder({
              visible = args.visible,
              hidden = args.hidden,
              clauses = params.clauses,
              target = params.target,
              specificity = params.specificity,
            })

            encoder.train({
              sentences  = args.sentences,
              codes = args.codes,
              samples = args.samples,
              iterations = iters_search,
              each = function (epoch)
                return each(epoch, encoder)
              end,
            })

          elseif typ == "classifier" then

            local classifier = M.classifier({
              features = args.features,
              classes = args.classes,
              negative = args.negative,
              clauses = params.clauses,
              target = params.target,
              specificity = params.specificity,
            })

            classifier.train({
              samples = args.samples,
              problems = args.problems,
              solutions = args.solutions,
              iterations = iters_search,
              each = function (epoch)
                return each(epoch, classifier)
              end,
            })

          else
            err.error("unexpected type to optimize", typ)
          end

          local trial_score = best_epoch_score
          if trial_score > round_best_score then
            round_best_score = trial_score
            round_best_params = params
          end
          if trial_score > best_score then
            best_score, best_params = trial_score, params
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

  if typ == "encoder" then

    local encoder = M.encoder({
      visible = args.visible,
      hidden = args.hidden,
      clauses = best_params.clauses,
      target = best_params.target,
      specificity = best_params.specificity,
    })

    encoder.train({
      sentences  = args.sentences,
      codes = args.codes,
      samples = args.samples,
      iterations = final_iters,
      each = function (epoch)
        if each_cb then
          local _, metrics = metric_fn(encoder)
          return each_cb(encoder, true, metrics, best_params, epoch)
        end
      end,
    })

    collectgarbage("collect")
    return encoder

  elseif typ == "classifier" then

    local classifier = M.classifier({
      features = args.features,
      classes = args.classes,
      negative = args.negative,
      clauses = best_params.clauses,
      target = best_params.target,
      specificity = best_params.specificity,
    })

    classifier.train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      iterations = iters_search,
      each = function (epoch)
        if each_cb then
          local _, metrics = metric_fn(classifier)
          return each_cb(classifier, true, metrics, best_params, epoch)
        end
      end,
    })

    collectgarbage("collect")
    return classifier

  else
    err.error("unexpected type to optimize", typ)
  end

end

M.optimize_classifier = function (args)
  return M.optimize(args, "classifier")
end

M.optimize_encoder = function (args)
  return M.optimize(args, "encoder")
end

return M

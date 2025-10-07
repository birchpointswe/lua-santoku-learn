local tm = require("santoku.tsetlin.capi")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")

local M = {}

-- Re-export capi functions (userdata now has methods directly via metatable)
M.load = tm.load
M.classifier = function (...)
  return tm.create("classifier", ...)
end
M.encoder = function (...)
  return tm.create("encoder", ...)
end
M.align = tm.align

local function round_to_pow2 (x)
  local log2x = num.log(x) / num.log(2)
  return num.pow(2, num.floor(log2x + 0.5))
end

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
    local is_pow2 = not not spec.pow2
    local span, base_center
    if is_log then
      span = num.log(maxv) - num.log(minv)
      base_center = num.log(def)
    else
      span = maxv - minv
      base_center = def
    end
    local jitter = (spec.dev or global_dev or 1.0) * span
    local round = 0
    return {
      type = "range",
      center = def,
      sample = function (center)
        local c = center and (is_log and num.log(center) or center) or base_center
        local x = rand.fast_normal(c, jitter)
        if is_log then
          x = num.exp(x)
        end
        if x < minv then
          x = minv
        elseif x > maxv then
          x = maxv
        end
        if is_pow2 then
          x = round_to_pow2(x)
          if x < minv then
            x = minv
          elseif x > maxv then x = maxv
          end
        elseif is_int then
          x = num.floor(x + 0.5)
        end
        return x
      end,
      shrink = function ()
        round  = round + 1
        jitter = jitter / num.sqrt(round)
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
  err.error("Bad hyper‑parameter specification: " .. spec)
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

  local param_names = { "clauses", "clause_tolerance", "clause_maximum", "target", "specificity" }

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
    -- Ensure FPTM constraint: clause_tolerance (LF) <= clause_maximum (L)
    if p.clause_tolerance and p.clause_maximum and p.clause_tolerance > p.clause_maximum then
      -- Swap them to maintain the constraint
      p.clause_tolerance, p.clause_maximum = p.clause_maximum, p.clause_tolerance
    end
    return p
  end

  local best_score = -num.huge
  local best_params = nil

  if all_fixed or not (trials and trials > 0) or not (rounds and rounds > 0) then
    best_params = sample_params()
  else

    for r = 1, rounds do

      local seen = {}
      local round_best_score  = -num.huge
      local round_best_params = nil

      for t = 1, trials do
        local params = sample_params()
        local key = str.format("%d|%d|%d|%d|%d",
          params.clauses,
          params.clause_tolerance,
          params.clause_maximum,
          num.floor(params.target * 1e6 + 0.5),
          num.floor(params.specificity * 1e6 + 0.5))
        if not seen[key] then
          seen[key] = true

          local best_epoch_score = -num.huge
          local last_epoch_score = -num.huge
          local epochs_since_improve = 0

          local function each (epoch, tm)
            local score, metrics = metric_fn(tm)
            local cb_result = nil
            if each_cb then
              cb_result = each_cb(tm, false, metrics, params, epoch, r, t)
            end
            last_epoch_score = score
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
              clause_tolerance = params.clause_tolerance,
              clause_maximum = params.clause_maximum,
              target = params.target,
              specificity = params.specificity,
            })

            encoder:train({
              sentences  = args.sentences,
              codes = args.codes,
              samples = args.samples,
              iterations = iters_search,
              threads = args.threads,
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
              clause_tolerance = params.clause_tolerance,
              clause_maximum = params.clause_maximum,
              target = params.target,
              specificity = params.specificity,
            })

            classifier:train({
              samples = args.samples,
              problems = args.problems,
              solutions = args.solutions,
              iterations = iters_search,
              threads = args.threads,
              each = function (epoch)
                return each(epoch, classifier)
              end,
            })

          else
            err.error("unexpected type to optimize", typ)
          end

          -- Use the last score (where training ended) rather than best score
          -- This prefers params that maintain good performance rather than spike and degrade
          local trial_score = last_epoch_score

          -- For round best, use final score
          if trial_score > round_best_score then
            round_best_score = trial_score
            round_best_params = params
          end

          -- For global best, also use final score
          if trial_score > best_score then
            best_score = trial_score
            best_params = params
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
      clause_tolerance = best_params.clause_tolerance,
      clause_maximum = best_params.clause_maximum,
      target = best_params.target,
      specificity = best_params.specificity,
    })

    encoder:train({
      sentences  = args.sentences,
      codes = args.codes,
      samples = args.samples,
      iterations = final_iters,
      threads = args.threads,
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
      clause_tolerance = best_params.clause_tolerance,
      clause_maximum = best_params.clause_maximum,
      target = best_params.target,
      specificity = best_params.specificity,
    })

    classifier:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      iterations = final_iters,
      threads = args.threads,
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

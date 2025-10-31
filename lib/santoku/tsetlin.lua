local tm = require("santoku.tsetlin.capi")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")
local cvec = require("santoku.cvec")
local utc = require("santoku.utc")

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
  local final_patience = args.final_patience or 0
  local use_final_early_stop = final_patience > 0
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
  local best_time = num.huge
  local best_params = nil

  if all_fixed or not (trials and trials > 0) or not (rounds and rounds > 0) then
    best_params = sample_params()
  else

    -- Create reusable TM for search trials
    local search_tm
    if typ == "encoder" then
      search_tm = M.encoder({
        visible = args.visible,
        hidden = args.hidden,
        clauses = 8,
        clause_tolerance = 8,
        clause_maximum = 8,
        target = 4,
        specificity = 1000,
        reusable = true,
      })
    elseif typ == "classifier" then
      search_tm = M.classifier({
        features = args.features,
        classes = args.classes,
        negative = args.negative,
        clauses = 8,
        clause_tolerance = 8,
        clause_maximum = 8,
        target = 4,
        specificity = 1000,
        reusable = true,
      })
    else
      err.error("unexpected type to optimize", typ)
    end

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

          search_tm:reconfigure(params)
          local trial_start_time = utc.time(true)

          if typ == "encoder" then
            search_tm:train({
              sentences  = args.sentences,
              codes = args.codes,
              samples = args.samples,
              iterations = iters_search,
              each = function (epoch)
                return each(epoch, search_tm)
              end,
            })
          elseif typ == "classifier" then
            search_tm:train({
              samples = args.samples,
              problems = args.problems,
              solutions = args.solutions,
              iterations = iters_search,
              each = function (epoch)
                return each(epoch, search_tm)
              end,
            })
          end

          local trial_elapsed = utc.time(true) - trial_start_time

          -- Use the last score (where training ended) rather than best score
          -- This prefers params that maintain good performance rather than spike and degrade
          local trial_score = last_epoch_score

          -- For round best, use final score
          if trial_score > round_best_score then
            round_best_score = trial_score
            round_best_params = params
          end

          -- For global best, use final score and break ties with elapsed time
          local is_better = trial_score > best_score + 1e-8 or
                           (num.abs(trial_score - best_score) < 1e-8 and trial_elapsed < best_time)
          if is_better then
            best_score = trial_score
            best_time = trial_elapsed
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

    search_tm:destroy()
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

    local best_final_score = -num.huge
    local best_final_metrics = nil
    local checkpoint = use_final_early_stop and cvec.create(0) or nil
    local has_checkpoint = false
    local epochs_since_improve = 0

    encoder:train({
      sentences  = args.sentences,
      codes = args.codes,
      samples = args.samples,
      iterations = final_iters,
      each = function (epoch)
        local cb_result = nil
        if each_cb then
          local score, metrics = metric_fn(encoder)
          cb_result = each_cb(encoder, true, metrics, best_params, epoch)
          if score > best_final_score + 1e-8 then
            best_final_score = score
            best_final_metrics = metrics
            epochs_since_improve = 0
            if use_final_early_stop then
              encoder:checkpoint(checkpoint)
              has_checkpoint = true
            end
          else
            epochs_since_improve = epochs_since_improve + 1
          end
          if use_final_early_stop and epochs_since_improve >= final_patience then
            return false
          end
        end
        return cb_result
      end,
    })

    if use_final_early_stop and has_checkpoint then
      encoder:restore(checkpoint)
    end

    collectgarbage("collect")
    return encoder, best_final_metrics

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

    local best_final_score = -num.huge
    local best_final_metrics = nil
    local checkpoint = use_final_early_stop and cvec.create(0) or nil
    local has_checkpoint = false
    local epochs_since_improve = 0

    classifier:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      iterations = final_iters,
      each = function (epoch)
        local cb_result = nil
        if each_cb then
          local score, metrics = metric_fn(classifier)
          cb_result = each_cb(classifier, true, metrics, best_params, epoch)
          if score > best_final_score + 1e-8 then
            best_final_score = score
            best_final_metrics = metrics
            epochs_since_improve = 0
            if use_final_early_stop then
              classifier:checkpoint(checkpoint)
              has_checkpoint = true
            end
          else
            epochs_since_improve = epochs_since_improve + 1
          end
          if use_final_early_stop and epochs_since_improve >= final_patience then
            return false
          end
        end
        return cb_result
      end,
    })

    if use_final_early_stop and has_checkpoint then
      classifier:restore(checkpoint)
    end

    collectgarbage("collect")
    return classifier, best_final_metrics

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

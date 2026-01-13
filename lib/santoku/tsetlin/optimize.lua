local tm = require("santoku.tsetlin.capi")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local itq = require("santoku.tsetlin.itq")
local ann = require("santoku.tsetlin.ann")
local hlth = require("santoku.tsetlin.hlth")
local evaluator = require("santoku.tsetlin.evaluator")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")
local cvec = require("santoku.cvec")
local dvec = require("santoku.dvec")
local ivec = require("santoku.ivec")

local M = {}

M.elbow_methods = {
  max_gap = {},
  otsu = {},
  lmethod = {},
  max_curvature = {},
  first_gap = { alpha = { def = 5, min = 1, max = 50 } },
  plateau = { alpha = { def = 1, min = 0, max = 10 } },
  first_gap_ratio = { alpha = { def = 3, min = 1, max = 10 } },
  kneedle = { alpha = { def = 1, min = 0.1, max = 10, log = true } },
}

M.select_metrics = {
  variance = {},
  skewness = {},
  entropy = { n_bins = { def = 0, min = 0, max = 100, int = true } },
  bimodality = {},
  dip = {},
  eigs = {},
}

local function sample_alpha_spec (spec)
  if spec == nil then
    return nil
  end
  if type(spec) == "number" then
    return spec
  end
  if type(spec) ~= "table" then
    return nil
  end
  if spec.min == nil or spec.max == nil then
    return spec.def
  end
  local val
  if spec.log then
    -- For log scale with non-positive min, shift range so shifted_min = 1
    local shift = spec.min <= 0 and (1 - spec.min) or 0
    local smin = spec.min + shift
    local smax = spec.max + shift
    local log_val = num.random() * (num.log(smax) - num.log(smin)) + num.log(smin)
    val = num.exp(log_val) - shift
  else
    val = num.random() * (spec.max - spec.min) + spec.min
  end
  if spec.int then
    val = num.floor(val + 0.5)
  end
  return val
end

M.threshold_methods = {
  itq = {
    iterations = { def = 100, min = 50, max = 500, int = true },
    tolerance = 1e-8,
  },
  otsu = {
    metric = { "variance", "entropy" },
    minimize = { true, false },
    n_bins = 32,
  },
  median = {},
  sign = {},
}

M.sample_threshold = function (threshold_cfg)
  if type(threshold_cfg) == "function" then
    return nil, threshold_cfg
  end
  local method
  if type(threshold_cfg) == "string" then
    method = threshold_cfg
  elseif type(threshold_cfg) == "table" and threshold_cfg.method then
    if type(threshold_cfg.method) == "table" and #threshold_cfg.method > 0 then
      method = threshold_cfg.method[num.random(#threshold_cfg.method)]
    else
      method = threshold_cfg.method
    end
  else
    method = "median"
  end
  local method_info = M.threshold_methods[method]
  if not method_info then
    err.error("Unknown threshold method: " .. tostring(method))
  end
  local params = { method = method }
  for param_name, param_spec in pairs(method_info) do
    local override = threshold_cfg and type(threshold_cfg) == "table" and threshold_cfg[method .. "_" .. param_name]
    local spec = override ~= nil and override or param_spec
    if type(spec) == "table" and #spec > 0 then
      params[param_name] = spec[num.random(#spec)]
    elseif type(spec) == "table" and (spec.min ~= nil or spec.def ~= nil) then
      params[param_name] = sample_alpha_spec(spec)
    else
      params[param_name] = spec
    end
  end
  return params, nil
end

M.build_threshold_fn = function (threshold_params)
  if not threshold_params then
    return function (codes, dims)
      return itq.median({ codes = codes, n_dims = dims }), dims
    end
  end
  local method = threshold_params.method
  if method == "itq" then
    return function (codes, dims)
      return itq.itq({
        codes = codes,
        n_dims = dims,
        iterations = threshold_params.iterations or 100,
        tolerance = threshold_params.tolerance or 1e-8,
      }), dims
    end
  elseif method == "otsu" then
    return function (codes, dims)
      return itq.otsu({
        codes = codes,
        n_dims = dims,
        metric = threshold_params.metric or "variance",
        minimize = threshold_params.minimize or false,
        n_bins = threshold_params.n_bins or 32,
      }), dims
    end
  elseif method == "median" then
    return function (codes, dims)
      return itq.median({ codes = codes, n_dims = dims }), dims
    end
  elseif method == "sign" then
    return function (codes, dims)
      return itq.sign({ codes = codes, n_dims = dims }), dims
    end
  else
    err.error("Unknown threshold method: " .. tostring(method))
  end
end

M.sample_elbow = function (elbow_list, elbow_alpha_override)
  local method
  if type(elbow_list) ~= "table" or #elbow_list == 0 then
    method = elbow_list
  else
    method = elbow_list[num.random(#elbow_list)]
  end

  if method == "none" then
    return "none", nil
  end

  local method_info = M.elbow_methods[method]
  if not method_info or not method_info.alpha then
    return method, nil
  end

  local spec
  if elbow_alpha_override ~= nil then
    if type(elbow_alpha_override) == "number" then
      return method, elbow_alpha_override
    elseif type(elbow_alpha_override) == "table" then
      if elbow_alpha_override[method] then
        spec = elbow_alpha_override[method]
      elseif elbow_alpha_override.min ~= nil or elbow_alpha_override.def ~= nil then
        spec = elbow_alpha_override
      end
    end
  end
  spec = spec or method_info.alpha

  return method, sample_alpha_spec(spec)
end

M.sample_metric = function (metric_list, metric_alpha_override)
  local metric
  if type(metric_list) ~= "table" or #metric_list == 0 then
    metric = metric_list
  else
    metric = metric_list[num.random(#metric_list)]
  end

  local metric_info = M.select_metrics[metric]
  if not metric_info then
    return metric, nil
  end

  local alpha = {}
  for param_name, param_spec in pairs(metric_info) do
    local spec
    if metric_alpha_override then
      if metric_alpha_override[metric] and metric_alpha_override[metric][param_name] then
        spec = metric_alpha_override[metric][param_name]
      elseif metric_alpha_override[param_name] then
        spec = metric_alpha_override[param_name]
      end
    end
    spec = spec or param_spec
    alpha[param_name] = sample_alpha_spec(spec)
  end
  return metric, alpha
end

local function round_to_pow2 (x)
  local log2x = num.log(x) / num.log(2)
  return num.pow(2, num.floor(log2x + 0.5))
end

M.build_sampler = function (spec, global_dev)
  if spec == nil then
    return nil
  end
  if type(spec) == "number" or type(spec) == "boolean" or type(spec) == "string" then
    return {
      type = "fixed",
      center = spec,
      sample = function ()
        return spec
      end
    }
  end
  if type(spec) == "table" and (spec.def ~= nil or (spec.min ~= nil and spec.max ~= nil)) then
    local def, minv, maxv = spec.def, spec.min, spec.max
    err.assert(minv and maxv, "range spec missing min|max")
    local is_log = not not spec.log
    local is_int = not not spec.int
    local is_pow2 = not not spec.pow2
    -- For log scale with non-positive min, shift range so shifted_min = 1
    local shift = (is_log and minv <= 0) and (1 - minv) or 0
    local smin = minv + shift
    local smax = maxv + shift
    local span = is_log and (num.log(smax) - num.log(smin)) or (maxv - minv)
    local init_jitter = (spec.dev or global_dev or 1.0) * span
    local jitter = init_jitter
    return {
      type = "range",
      center = def,
      sample = function (center)
        local x
        if center then
          local c = is_log and num.log(center + shift) or center
          x = rand.fast_normal(c, jitter)
          if is_log then x = num.exp(x) - shift end
        else
          if is_log then
            x = num.exp(num.random() * span + num.log(smin)) - shift
          else
            x = num.random() * span + minv
          end
        end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        if is_pow2 then
          x = round_to_pow2(x)
          if x < minv then x = minv elseif x > maxv then x = maxv end
        elseif is_int then
          x = num.floor(x + 0.5)
        end
        return x
      end,
      adapt = function (factor)
        jitter = jitter * factor
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
  err.error("Bad hyper-parameter specification", spec)
end

M.build_samplers = function (args, param_names, global_dev)
  local samplers = {}
  for _, pname in ipairs(param_names) do
    samplers[pname] = M.build_sampler(args[pname], global_dev)
  end
  return samplers
end

M.sample_params = function (samplers, param_names, base_cfg)
  local p = {}
  -- Start with a shallow copy of base config if provided
  if base_cfg then
    for k, v in pairs(base_cfg) do
      p[k] = v
    end
  end
  -- Override with sampled values
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s then
      if s.type == "range" then
        p[name] = s.sample(s.center)
      else
        p[name] = s.sample()
      end
    end
  end
  return p
end

M.sample_tier = function (tier_cfg)
  local result = {}
  for k, v in pairs(tier_cfg) do
    if type(v) == "table" and (v.def ~= nil or (v.min ~= nil and v.max ~= nil)) then
      local minv, maxv = v.min, v.max
      local is_log = v.log
      local is_int = v.int
      local is_pow2 = v.pow2
      local x
      if is_log then
        -- For log scale with non-positive min, shift range so shifted_min = 1
        local shift = minv <= 0 and (1 - minv) or 0
        local smin = minv + shift
        local smax = maxv + shift
        local log_val = num.random() * (num.log(smax) - num.log(smin)) + num.log(smin)
        x = num.exp(log_val) - shift
      else
        x = num.random() * (maxv - minv) + minv
      end
      if x < minv then x = minv elseif x > maxv then x = maxv end
      if is_pow2 then
        x = round_to_pow2(x)
        if x < minv then x = minv elseif x > maxv then x = maxv end
      elseif is_int then
        x = num.floor(x + 0.5)
      end
      result[k] = x
    elseif type(v) == "table" and #v > 0 then
      result[k] = v[num.random(#v)]
    else
      result[k] = v
    end
  end
  return result
end

M.tier_all_fixed = function (tier_cfg)
  for _, v in pairs(tier_cfg) do
    if type(v) == "table" and v.def ~= nil then
      return false
    end
    if type(v) == "table" and v.min ~= nil and v.max ~= nil then
      return false
    end
    if type(v) == "table" and #v > 1 then
      return false
    end
  end
  return true
end

M.all_fixed = function (samplers)
  for _, s in pairs(samplers) do
    if s and s.type ~= "fixed" then
      return false
    end
  end
  return true
end

local function has_alpha_range (alpha_cfg)
  if alpha_cfg == nil then return false end
  if type(alpha_cfg) == "number" then return false end
  if type(alpha_cfg) ~= "table" then return false end
  if alpha_cfg.min ~= nil and alpha_cfg.max ~= nil then return true end
  for _, spec in pairs(alpha_cfg) do
    if type(spec) == "table" and spec.min ~= nil and spec.max ~= nil then
      return true
    end
  end
  return false
end

-- Recenter samplers on best-known parameters
M.recenter = function (samplers, param_names, best_params)
  for _, pname in ipairs(param_names) do
    local s = samplers[pname]
    if s and s.type == "range" and best_params[pname] then
      s.center = best_params[pname]
    end
  end
end

-- Adapt jitter for all samplers by a factor
M.adapt_jitter = function (samplers, factor)
  for _, s in pairs(samplers) do
    if s and s.adapt then
      s.adapt(factor)
    end
  end
end

-- 1/5th success rule: compute adaptive factor based on improvement rate
-- >20% improving -> don't shrink (exploration working)
-- 5-20% improving -> moderate shrink
-- <5% improving -> aggressive shrink (need to focus)
M.success_factor = function (success_rate)
  if success_rate > 0.2 then
    return 1.0
  elseif success_rate > 0.05 then
    return 0.85
  else
    return 0.5
  end
end

M.search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local rounds = args.rounds or 3
  local trials = args.trials or 10
  local tolerance = args.tolerance or 1e-8
  local make_key = args.make_key
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local best_score = -num.huge
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if M.all_fixed(samplers) or rounds <= 0 or trials <= 0 then
    best_params = M.sample_params(samplers, param_names)
    if skip_final then
      return nil, best_params, nil
    else
      local _, metrics, result = trial_fn(best_params, { is_final = true })
      return result, best_params, metrics
    end
  end

  for r = 1, rounds do
    local seen = {}
    local round_best_score = -num.huge
    local round_best_params = nil
    local round_improvements = 0
    local round_samples = 0
    for t = 1, trials do
      local params = M.sample_params(samplers, param_names)
      local dominated = false
      if make_key then
        local key = make_key(params)
        if seen[key] then
          dominated = true
        else
          seen[key] = true
        end
      end
      if not dominated then
        local score, metrics, result = trial_fn(params, {
          round = r,
          trial = t,
          is_final = false,
        })
        round_samples = round_samples + 1
        if each_cb then
          each_cb({
            event = "trial",
            round = r,
            trial = t,
            params = params,
            score = score,
            metrics = metrics,
          })
        end
        if score > round_best_score then
          round_best_score = score
          round_best_params = params
        end
        if score > best_score + tolerance then
          round_improvements = round_improvements + 1
          if best_result and cleanup_fn then
            cleanup_fn(best_result)
          end
          best_score = score
          best_params = params
          best_result = result
          best_metrics = metrics
        else
          if result and cleanup_fn then
            cleanup_fn(result)
          end
        end
      end
    end
    -- Recenter on global best and adapt jitter using 1/5th success rule
    if best_params then
      M.recenter(samplers, param_names, best_params)
    end
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor = M.success_factor(success_rate)
    M.adapt_jitter(samplers, adapt_factor)
    if each_cb then
      each_cb({
        event = "round",
        round = r,
        round_best_score = round_best_score,
        round_best_params = round_best_params,
        global_best_score = best_score,
        global_best_params = best_params,
        success_rate = success_rate,
        adapt_factor = adapt_factor,
      })
    end
    collectgarbage("collect")
  end

  if not skip_final and best_params then
    if each_cb then
      each_cb({ event = "final_start", params = best_params })
    end
    if best_result and cleanup_fn then
      cleanup_fn(best_result)
    end
    local final_score, final_metrics, final_result = trial_fn(best_params, { is_final = true })
    best_result = final_result
    best_metrics = final_metrics
    if each_cb then
      each_cb({
        event = "final_end",
        params = best_params,
        score = final_score,
        metrics = final_metrics,
      })
    end
  end

  return best_result, best_params, best_metrics

end

local function create_tm (typ, args)
  if typ == "encoder" then
    return tm.create("encoder", {
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
    return tm.create("classifier", {
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
    err.error("unexpected type", typ)
  end
end

local function create_final_tm (typ, args, params)
  if typ == "encoder" then
    return tm.create("encoder", {
      visible = args.visible,
      hidden = args.hidden,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
    })
  elseif typ == "classifier" then
    return tm.create("classifier", {
      features = args.features,
      classes = args.classes,
      negative = args.negative,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
    })
  else
    err.error("unexpected type", typ)
  end
end

local function train_tm (typ, tmobj, args, params, iterations, early_patience, tolerance, metric_fn, each_cb, info, encoding_info)
  local best_epoch_score = -num.huge
  local last_epoch_score = -num.huge
  local last_metrics = nil
  local epochs_since_improve = 0
  local checkpoint = (early_patience and early_patience > 0) and cvec.create(0) or nil
  local has_checkpoint = false

  local enc_info = encoding_info or {
    sentences = args.sentences,
    samples = args.samples,
  }

  local function on_epoch (epoch)
    local score, metrics = metric_fn(tmobj, enc_info)
    last_epoch_score = score
    last_metrics = metrics
    if each_cb then
      local cb_result = each_cb(tmobj, info.is_final, metrics, params, epoch,
        not info.is_final and info.round or nil,
        not info.is_final and info.trial or nil)
      if cb_result == false then
        return false
      end
    end
    if score > best_epoch_score + tolerance then
      best_epoch_score = score
      epochs_since_improve = 0
      if checkpoint then
        tmobj:checkpoint(checkpoint)
        has_checkpoint = true
      end
    else
      epochs_since_improve = epochs_since_improve + 1
    end
    if early_patience and early_patience > 0 and epochs_since_improve >= early_patience then
      return false
    end
  end

  if typ == "encoder" then
    tmobj:train({
      sentences = args.sentences,
      codes = args.codes,
      samples = args.samples,
      iterations = iterations,
      each = on_epoch,
    })
  elseif typ == "classifier" then
    tmobj:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      iterations = iterations,
      each = on_epoch,
    })
  end

  if checkpoint and has_checkpoint then
    tmobj:restore(checkpoint)
    last_epoch_score, last_metrics = metric_fn(tmobj, enc_info)
  end

  return last_epoch_score, last_metrics
end

local function make_landmark_key (p)
  return str.format("%d|%d|%s", p.n_landmarks or 0, p.n_thresholds or 0, p.landmark_mode or "frequency")
end

local function optimize_tm (args, typ)

  local patience = args.search_patience or 10
  local use_early_stop = patience > 0
  local final_patience = args.final_patience or 40
  local use_final_early_stop = final_patience > 0
  local tolerance = args.search_tolerance or 1e-8
  local iters_search = args.search_iterations or 10
  local final_iters = args.final_iterations or (iters_search * 10)
  local global_dev = args.search_dev or 1.0
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local use_landmarks = typ == "encoder" and args.landmarks_index and args.codes_index and args.features
  local landmark_cache = {}
  local current_sentences = args.sentences
  local current_visible = args.visible

  local param_names = { "clauses", "clause_tolerance", "clause_maximum", "target", "specificity" }
  if use_landmarks then
    param_names[#param_names + 1] = "n_landmarks"
    param_names[#param_names + 1] = "n_thresholds"
    param_names[#param_names + 1] = "landmark_mode"
  end

  local samplers = M.build_samplers(args, param_names, global_dev)

  local function get_encoded_data (params)
    if not use_landmarks then
      return current_sentences, current_visible
    end
    local key = make_landmark_key(params)
    if landmark_cache[key] then
      return landmark_cache[key].sentences, landmark_cache[key].visible
    end
    local mode = params.landmark_mode or "frequency"
    local encode_fn, n_visible = hlth.landmark_encoder({
      landmarks_index = args.landmarks_index,
      codes_index = args.codes_index,
      n_landmarks = params.n_landmarks,
      mode = mode,
      n_thresholds = mode == "frequency" and params.n_thresholds or nil,
    })
    local sentences = encode_fn(args.features, args.samples)
    sentences:bits_flip_interleave(n_visible)
    landmark_cache[key] = { sentences = sentences, visible = n_visible }
    return sentences, n_visible
  end

  local search_tm
  local search_tm_visible
  if not M.all_fixed(samplers) then
    local _, init_visible = get_encoded_data({
      n_landmarks = samplers.n_landmarks and samplers.n_landmarks.center or args.n_landmarks,
      n_thresholds = samplers.n_thresholds and samplers.n_thresholds.center or args.n_thresholds,
    })
    search_tm = create_tm(typ, {
      visible = init_visible,
      hidden = args.hidden,
      features = args.features,
      classes = args.classes,
      negative = args.negative,
    })
    search_tm_visible = init_visible
  end

  local function search_trial_fn (params, info)
    if params.clause_tolerance and params.clause_maximum and params.clause_tolerance > params.clause_maximum then
      params.clause_tolerance, params.clause_maximum = params.clause_maximum, params.clause_tolerance
    end
    local sentences, visible = get_encoded_data(params)
    if use_landmarks and visible ~= search_tm_visible then
      search_tm:destroy()
      search_tm = create_tm(typ, {
        visible = visible,
        hidden = args.hidden,
        features = args.features,
        classes = args.classes,
        negative = args.negative,
      })
      search_tm_visible = visible
    end
    search_tm:reconfigure(params)
    local train_args = {
      sentences = sentences,
      codes = args.codes,
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
    }
    local encoding_info = {
      sentences = sentences,
      visible = visible,
      samples = args.samples,
      n_landmarks = params.n_landmarks,
      n_thresholds = params.n_thresholds,
    }
    local score, metrics = train_tm(typ, search_tm, train_args, params, iters_search,
      use_early_stop and patience or nil, tolerance, metric_fn, each_cb, info, encoding_info)
    return score, metrics, nil
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    rounds = args.search_rounds or 3,
    trials = args.search_trials or 10,
    tolerance = tolerance,
    trial_fn = search_trial_fn,
    skip_final = true,
    make_key = function (p)
      local base = str.format("%d|%d|%d|%d|%d",
        p.clauses,
        p.clause_tolerance,
        p.clause_maximum,
        num.floor(p.target * 1e6 + 0.5),
        num.floor(p.specificity * 1e6 + 0.5))
      if use_landmarks then
        return base .. "|" .. make_landmark_key(p)
      end
      return base
    end,
  })

  if search_tm then
    search_tm:destroy()
  end

  local final_sentences, final_visible = get_encoded_data(best_params)
  local final_tm = create_final_tm(typ, {
    visible = final_visible,
    hidden = args.hidden,
    features = args.features,
    classes = args.classes,
    negative = args.negative,
  }, best_params)
  local final_train_args = {
    sentences = final_sentences,
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
  }
  local final_encoding_info = {
    sentences = final_sentences,
    visible = final_visible,
    samples = args.samples,
    n_landmarks = best_params.n_landmarks,
    n_thresholds = best_params.n_thresholds,
    landmark_mode = best_params.landmark_mode,
  }
  local _, final_metrics = train_tm(typ, final_tm, final_train_args, best_params, final_iters,
    use_final_early_stop and final_patience or nil, tolerance, metric_fn, each_cb, { is_final = true }, final_encoding_info)

  for _, cached in pairs(landmark_cache) do
    cached.sentences:destroy()
  end

  collectgarbage("collect")
  return final_tm, final_metrics, best_params
end

M.classifier = function (args)
  return optimize_tm(args, "classifier")
end

M.encoder = function (args)
  return optimize_tm(args, "encoder")
end

M.destroy_spectral = function (model)
  if not model then return end
  if model.ids then model.ids:destroy() end
  if model.index then model.index:destroy() end
  if model.codes then model.codes:destroy() end
  if model.raw_codes then model.raw_codes:destroy() end
  if model.eigs then model.eigs:destroy() end
  if model.adj_ids then model.adj_ids:destroy() end
  if model.adj_offsets then model.adj_offsets:destroy() end
  if model.adj_neighbors then model.adj_neighbors:destroy() end
  if model.adj_weights then model.adj_weights:destroy() end
end

M.build_adjacency = function (args)
  local index = args.index
  local knn_index = args.knn_index or index
  local p = args.params
  local each = args.each

  return graph.adjacency({
    weight_index = index,
    weight_cmp = p.cmp,
    weight_decay = p.weight_decay,
    knn_index = knn_index,
    knn = p.knn,
    knn_mode = p.knn_mode,
    knn_alpha = p.knn_alpha,
    knn_mutual = p.knn_mutual,
    knn_cache = p.knn_cache or 32,
    bridge = p.bridge,
    each = each,
  })
end

M.build_spectral_raw = function (args)
  local adj_ids, adj_offsets, adj_neighbors, adj_weights = args.adj_ids, args.adj_offsets, args.adj_neighbors, args.adj_weights
  local p = args.params
  local each = args.each

  local ids, raw_codes, _, eigs = spectral.encode({
    type = p.laplacian or "unnormalized",
    n_hidden = p.n_dims,
    eps = p.eps or 1e-10,
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    weights = adj_weights,
    each = each,
  })

  return {
    ids = ids,
    raw_codes = raw_codes,
    eigs = eigs,
    dims = p.n_dims,
  }
end

M.apply_threshold = function (args)
  local raw_model = args.raw_model
  local threshold_params = args.threshold_params
  local bucket_size = args.bucket_size

  local threshold_fn = M.build_threshold_fn(threshold_params)
  local codes = threshold_fn(raw_model.raw_codes, raw_model.dims)

  local ann_index = ann.create({
    expected_size = raw_model.ids:size(),
    bucket_size = bucket_size,
    features = raw_model.dims,
  })
  ann_index:add(codes, raw_model.ids)

  return {
    ids = raw_model.ids,
    codes = codes,
    raw_codes = raw_model.raw_codes,
    threshold_fn = threshold_fn,
    threshold_params = threshold_params,
    eigs = raw_model.eigs,
    dims = raw_model.dims,
    index = ann_index,
  }
end

M.destroy_threshold_model = function (model)
  if not model then return end
  if model.index then model.index:destroy() end
  if model.codes then model.codes:destroy() end
end

M.destroy_spectral_raw = function (raw_model)
  if not raw_model then return end
  if raw_model.raw_codes then raw_model.raw_codes:destroy() end
  if raw_model.eigs then raw_model.eigs:destroy() end
end

M.build_spectral_from_adjacency = function (args)
  local p = args.params
  local bucket_size = args.bucket_size

  local raw_model = M.build_spectral_raw({
    adj_ids = args.adj_ids,
    adj_offsets = args.adj_offsets,
    adj_neighbors = args.adj_neighbors,
    adj_weights = args.adj_weights,
    params = p,
    each = args.each,
  })

  local threshold_params
  if type(p.threshold) == "function" then
    threshold_params = p.threshold_params or { method = "custom" }
  else
    threshold_params = M.sample_threshold(p.threshold)
  end

  return M.apply_threshold({
    raw_model = raw_model,
    threshold_params = threshold_params,
    bucket_size = bucket_size,
  })
end

M.score_spectral_eval = function (args)
  local model = args.model
  local eval_params = args.eval_params
  local expected = args.expected
  local bucket_size = args.bucket_size

  local eval_codes
  local eval_dims = model.dims
  local eval_index = model.index
  local temp_raw = nil
  local temp_codes = nil
  local temp_index = nil
  local temp_kept = nil
  local selected_elbow = nil

  local select_metric = eval_params.select_metric or "entropy"
  if eval_params.select_elbow and eval_params.select_elbow ~= "none" and select_metric ~= "none" then
    local metric = select_metric
    local malpha = eval_params.select_metric_alpha or {}
    local n_samples = model.ids:size()
    local indices, scores
    if metric == "entropy" then
      indices, scores = model.raw_codes:mtx_top_entropy(n_samples, eval_dims, eval_dims, malpha.n_bins)
    elseif metric == "variance" then
      indices, scores = model.raw_codes:mtx_top_variance(n_samples, eval_dims, eval_dims)
    elseif metric == "skewness" then
      indices, scores = model.raw_codes:mtx_top_skewness(n_samples, eval_dims, eval_dims)
    elseif metric == "bimodality" then
      indices, scores = model.raw_codes:mtx_top_bimodality(n_samples, eval_dims, eval_dims)
    elseif metric == "dip" then
      indices, scores = model.raw_codes:mtx_top_dip(n_samples, eval_dims, eval_dims)
    elseif metric == "eigs" then
      indices = ivec.create()
      for i = 0, eval_dims - 1 do
        indices:push(i)
      end
      scores = dvec.create()
      for i = 0, eval_dims - 1 do
        scores:push(model.eigs:get(i))
      end
    else
      err.error("Unknown select_metric: " .. tostring(metric))
    end
    local _, elbow_idx = scores:scores_elbow(eval_params.select_elbow, eval_params.select_elbow_alpha)
    scores:destroy()
    local selected_dims = elbow_idx
    selected_elbow = elbow_idx

    if selected_dims > 0 and selected_dims < eval_dims then
      indices:setn(selected_dims)
      temp_kept = indices
      temp_raw = dvec.create()
      model.raw_codes:mtx_select(temp_kept, nil, eval_dims, temp_raw)
      temp_codes = model.threshold_fn(temp_raw, selected_dims)
      eval_codes = temp_codes
      eval_dims = selected_dims

      temp_index = ann.create({
        expected_size = model.ids:size(),
        bucket_size = bucket_size,
        features = eval_dims,
      })
      temp_index:add(eval_codes, model.ids)
      eval_index = temp_index
    else
      indices:destroy()
    end
  end

  local adj_retrieved_ids, adj_retrieved_offsets, adj_retrieved_neighbors, adj_retrieved_weights =
    graph.adjacency({
      weight_index = eval_index,
      seed_ids = expected.ids,
      seed_offsets = expected.offsets,
      seed_neighbors = expected.neighbors,
    })

  local stats = evaluator.score_retrieval({
    retrieved_ids = adj_retrieved_ids,
    retrieved_offsets = adj_retrieved_offsets,
    retrieved_neighbors = adj_retrieved_neighbors,
    retrieved_weights = adj_retrieved_weights,
    expected_ids = expected.ids,
    expected_offsets = expected.offsets,
    expected_neighbors = expected.neighbors,
    expected_weights = expected.weights,
    query_ids = eval_params.query_ids,
    ranking = eval_params.ranking,
    metric = eval_params.metric,
    elbow = eval_params.elbow,
    elbow_alpha = eval_params.elbow_alpha,
    n_dims = eval_dims,
  })

  if adj_retrieved_ids then adj_retrieved_ids:destroy() end
  if adj_retrieved_offsets then adj_retrieved_offsets:destroy() end
  if adj_retrieved_neighbors then adj_retrieved_neighbors:destroy() end
  if adj_retrieved_weights then adj_retrieved_weights:destroy() end
  if temp_index then temp_index:destroy() end
  if temp_codes then temp_codes:destroy() end
  if temp_raw then temp_raw:destroy() end
  if temp_kept then temp_kept:destroy() end

  local target = eval_params.target or "combined"
  local target_score = stats[target]

  return target_score, {
    score = stats.score,
    quality = stats.quality,
    combined = stats.combined,
    n_dims = eval_dims,
    selected_elbow = selected_elbow,
    total_queries = stats.total_queries,
  }
end

M.spectral = function (args)
  local index = args.index
  local knn_index = args.knn_index
  local bucket_size = args.bucket_size
  local each_cb = args.each
  local adjacency_each = args.adjacency_each
  local spectral_each = args.spectral_each
  local knn_target_ids = args.knn_target_ids
  local query_ids = args.query_ids

  local adjacency_cfg = err.assert(args.adjacency, "adjacency config required")
  local spectral_cfg = err.assert(args.spectral, "spectral config required")
  local eval_cfg = err.assert(args.eval, "eval config required")

  local adjacency_samples = args.adjacency_samples or 1
  local spectral_samples = args.spectral_samples or 1
  local eval_samples = args.eval_samples or 1
  local rounds = args.rounds or 1
  local patience = args.patience or 0
  local global_dev = args.dev

  local expected = {
    ids = args.expected_ids,
    offsets = args.expected_offsets,
    neighbors = args.expected_neighbors,
    weights = args.expected_weights,
  }

  local best_score = -num.huge
  local best_params = nil
  local best_model = nil
  local best_metrics = nil
  local best_adj_ids = nil
  local best_adj_offsets = nil
  local best_adj_neighbors = nil
  local best_adj_weights = nil
  local sample_count = 0
  local rounds_without_improvement = 0

  local adj_fixed = M.tier_all_fixed(adjacency_cfg)
  local spec_fixed = M.tier_all_fixed(spectral_cfg)
  local eval_fixed = M.tier_all_fixed(eval_cfg)
    and (type(eval_cfg.elbow) ~= "table" or #eval_cfg.elbow <= 1)
    and not has_alpha_range(eval_cfg.elbow_alpha)
    and (type(eval_cfg.select_elbow) ~= "table" or #eval_cfg.select_elbow <= 1)
    and not has_alpha_range(eval_cfg.select_elbow_alpha)
    and (type(eval_cfg.select_metric) ~= "table" or #eval_cfg.select_metric <= 1)
    and not has_alpha_range(eval_cfg.select_metric_alpha)

  -- Build samplers for hill climbing (adjacency and spectral tiers)
  local adj_param_names = { "knn", "knn_alpha", "weight_decay", "knn_mutual", "knn_mode", "cmp", "bridge" }
  local spec_param_names = { "n_dims", "laplacian", "eps" }
  local adj_samplers = M.build_samplers(adjacency_cfg, adj_param_names, global_dev)
  local spec_samplers = M.build_samplers(spectral_cfg, spec_param_names, global_dev)

  local function make_adj_key (p)
    return str.format("%s|%s|%s|%s",
      tostring(p.knn), tostring(p.knn_alpha), tostring(p.weight_decay), tostring(p.knn_mutual))
  end

  local function make_spec_key (adj_key, p)
    return str.format("%s|%s|%s", adj_key, tostring(p.n_dims), tostring(p.laplacian))
  end

  local function make_eval_key (spec_key, p)
    local alpha_str = p.elbow_alpha and str.format("%.2f", p.elbow_alpha) or "nil"
    local select_str
    if p.select_elbow then
      local metric_alpha_str = ""
      if p.select_metric_alpha then
        for k, v in pairs(p.select_metric_alpha) do
          metric_alpha_str = metric_alpha_str .. str.format("|%s=%.2f", k, v)
        end
      end
      select_str = str.format("%s%s|%s|%.2f", p.select_metric or "entropy", metric_alpha_str, p.select_elbow, p.select_elbow_alpha or 0)
    else
      select_str = "none"
    end
    return str.format("%s|%s|%s|%s|%s", spec_key, p.elbow, alpha_str, p.ranking, select_str)
  end

  local function mem_kb ()
    return collectgarbage("count")
  end

  for round = 1, rounds do
    local seen = {}
    local round_best_score = -num.huge
    local round_best_params = nil
    local round_improvements = 0
    local round_samples = 0
    local round_start_mem = mem_kb()

    if each_cb then
      each_cb({
        event = "round_start",
        round = round,
        rounds = rounds,
        mem_kb = round_start_mem,
      })
    end

  for _ = 1, (adj_fixed and 1 or adjacency_samples) do
    local adj_params = M.sample_params(adj_samplers, adj_param_names, adjacency_cfg)
    local adj_key = make_adj_key(adj_params)

    if not seen[adj_key] then
      seen[adj_key] = true
      sample_count = sample_count + 1

      local adj_mem_before = mem_kb()
      if each_cb then
        each_cb({
          event = "stage",
          stage = "adjacency",
          sample = sample_count,
          is_final = false,
          params = { adjacency = adj_params },
          mem_kb = adj_mem_before,
        })
      end

      local adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
        index = index,
        knn_index = knn_index,
        params = adj_params,
        each = adjacency_each,
      })

      for _ = 1, (spec_fixed and 1 or spectral_samples) do
        local spec_params = M.sample_params(spec_samplers, spec_param_names, spectral_cfg)

        local threshold_params, threshold_fn = M.sample_threshold(spectral_cfg.threshold)
        if threshold_params then
          spec_params.threshold_params = threshold_params
          spec_params.threshold = M.build_threshold_fn(threshold_params)
        elseif threshold_fn then
          spec_params.threshold = threshold_fn
        end

        local spec_key = make_spec_key(adj_key, spec_params)
        if threshold_params then
          spec_key = spec_key .. "|" .. threshold_params.method
        end

        if not seen[spec_key] then
          seen[spec_key] = true

          local spec_mem_before = mem_kb()
          if each_cb then
            each_cb({
              event = "stage",
              stage = "spectral",
              sample = sample_count,
              is_final = false,
              params = { adjacency = adj_params, spectral = spec_params },
              mem_kb = spec_mem_before,
            })
          end

          local model = M.build_spectral_from_adjacency({
            adj_ids = adj_ids,
            adj_offsets = adj_offsets,
            adj_neighbors = adj_neighbors,
            adj_weights = adj_weights,
            params = spec_params,
            bucket_size = bucket_size,
            each = spectral_each,
          })

          local model_is_best = false
          local consecutive_zeros = 0
          local max_consecutive_zeros = eval_cfg.max_consecutive_zeros or 3
          local min_selected_dims = eval_cfg.min_selected_dims or 4

          for eval_i = 1, (eval_fixed and 1 or eval_samples) do
            if consecutive_zeros >= max_consecutive_zeros then
              if each_cb then
                each_cb({
                  event = "eval_early_stop",
                  consecutive_zeros = consecutive_zeros,
                  eval_sample = eval_i,
                  eval_samples = eval_samples,
                })
              end
              break
            end

            local eval_params = M.sample_tier(eval_cfg)
            local elbow, elbow_alpha = M.sample_elbow(eval_cfg.elbow, eval_cfg.elbow_alpha)
            eval_params.elbow = elbow
            eval_params.elbow_alpha = elbow_alpha
            if eval_cfg.select_elbow then
              local select_elbow, select_elbow_alpha = M.sample_elbow(eval_cfg.select_elbow, eval_cfg.select_elbow_alpha)
              eval_params.select_elbow = select_elbow
              eval_params.select_elbow_alpha = select_elbow_alpha
              local select_metric, select_metric_alpha = M.sample_metric(eval_cfg.select_metric, eval_cfg.select_metric_alpha)
              eval_params.select_metric = select_metric or "entropy"
              eval_params.select_metric_alpha = select_metric_alpha
            end
            eval_params.query_ids = query_ids

            local eval_key = make_eval_key(spec_key, eval_params)

            if not seen[eval_key] then
              seen[eval_key] = true

              local score, metrics = M.score_spectral_eval({
                model = model,
                eval_params = eval_params,
                expected = expected,
                knn_target_ids = knn_target_ids,
                bucket_size = bucket_size,
              })

              local skipped = false
              if metrics.n_dims and metrics.n_dims < min_selected_dims and metrics.selected_elbow then
                skipped = true
              end

              if each_cb then
                each_cb({
                  event = "eval",
                  sample = sample_count,
                  is_final = false,
                  params = { adjacency = adj_params, spectral = spec_params, eval = eval_params },
                  score = score,
                  metrics = metrics,
                  mem_kb = mem_kb(),
                  skipped = skipped,
                })
              end

              if skipped then
                score = 0
              end

              if score <= 0 then
                consecutive_zeros = consecutive_zeros + 1
              else
                consecutive_zeros = 0
              end

              round_samples = round_samples + 1

              if score > round_best_score then
                round_best_score = score
                round_best_params = { adjacency = adj_params, spectral = spec_params, eval = eval_params }
              end

              if score > best_score then
                round_improvements = round_improvements + 1
                if best_model and best_model ~= model then
                  M.destroy_spectral(best_model)
                end
                if best_adj_ids and best_adj_ids ~= adj_ids then
                  best_adj_ids:destroy()
                  best_adj_offsets:destroy()
                  best_adj_neighbors:destroy()
                  if best_adj_weights then best_adj_weights:destroy() end
                end
                best_score = score
                best_params = { adjacency = adj_params, spectral = spec_params, eval = eval_params }
                best_model = model
                best_metrics = metrics
                best_adj_ids = adj_ids
                best_adj_offsets = adj_offsets
                best_adj_neighbors = adj_neighbors
                best_adj_weights = adj_weights
                model_is_best = true
              end
            end
          end

          if not model_is_best then
            M.destroy_spectral(model)
          end
          collectgarbage("collect")

          if each_cb then
            each_cb({
              event = "spectral_done",
              mem_kb = mem_kb(),
              mem_before = spec_mem_before,
            })
          end
        end
      end

      if adj_ids and adj_ids ~= best_adj_ids then
        adj_ids:destroy()
        adj_offsets:destroy()
        adj_neighbors:destroy()
        if adj_weights then adj_weights:destroy() end
      end
      collectgarbage("collect")

      if each_cb then
        each_cb({
          event = "adjacency_done",
          mem_kb = mem_kb(),
          mem_before = adj_mem_before,
        })
      end
    end
  end

    -- Recenter on global best and adapt jitter using 1/5th success rule
    if best_params then
      M.recenter(adj_samplers, adj_param_names, best_params.adjacency)
      M.recenter(spec_samplers, spec_param_names, best_params.spectral)
    end
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor = M.success_factor(success_rate)
    M.adapt_jitter(adj_samplers, adapt_factor)
    M.adapt_jitter(spec_samplers, adapt_factor)

    if round_improvements > 0 then
      rounds_without_improvement = 0
    else
      rounds_without_improvement = rounds_without_improvement + 1
    end

    collectgarbage("collect")
    local round_end_mem = mem_kb()

    if each_cb then
      each_cb({
        event = "round_end",
        round = round,
        rounds = rounds,
        round_best_score = round_best_score,
        round_best_params = round_best_params,
        global_best_score = best_score,
        global_best_params = best_params,
        success_rate = success_rate,
        adapt_factor = adapt_factor,
        rounds_without_improvement = rounds_without_improvement,
        mem_kb = round_end_mem,
        mem_delta_kb = round_end_mem - round_start_mem,
      })
    end

    if patience > 0 and rounds_without_improvement >= patience then
      break
    end

  end

  local all_fixed = adj_fixed and spec_fixed and eval_fixed
  if best_model and not all_fixed then
    if each_cb then
      each_cb({
        event = "stage",
        stage = "adjacency",
        is_final = true,
        params = best_params,
      })
    end

    if best_adj_ids then
      best_adj_ids:destroy()
      best_adj_offsets:destroy()
      best_adj_neighbors:destroy()
      if best_adj_weights then best_adj_weights:destroy() end
    end

    local adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
      index = index,
      knn_index = knn_index,
      params = best_params.adjacency,
      each = adjacency_each,
    })

    M.destroy_spectral(best_model)

    if each_cb then
      each_cb({
        event = "stage",
        stage = "spectral",
        is_final = true,
        params = best_params,
      })
    end

    local final_spectral_params = {}
    for k, v in pairs(best_params.spectral) do final_spectral_params[k] = v end
    if best_params.spectral.threshold_params then
      final_spectral_params.threshold = M.build_threshold_fn(best_params.spectral.threshold_params)
    end
    best_model = M.build_spectral_from_adjacency({
      adj_ids = adj_ids,
      adj_offsets = adj_offsets,
      adj_neighbors = adj_neighbors,
      adj_weights = adj_weights,
      params = final_spectral_params,
      bucket_size = bucket_size,
      each = spectral_each,
    })

    best_model.adj_ids = adj_ids
    best_model.adj_offsets = adj_offsets
    best_model.adj_neighbors = adj_neighbors
    best_model.adj_weights = adj_weights
  elseif best_model and all_fixed then
    best_model.adj_ids = best_adj_ids
    best_model.adj_offsets = best_adj_offsets
    best_model.adj_neighbors = best_adj_neighbors
    best_model.adj_weights = best_adj_weights
  end

  collectgarbage("collect")
  return best_model, best_params, best_metrics
end

return M

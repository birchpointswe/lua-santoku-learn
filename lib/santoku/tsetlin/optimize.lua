local tm = require("santoku.tsetlin.capi")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local prone = require("santoku.tsetlin.prone")
local itq = require("santoku.tsetlin.itq")
local ann = require("santoku.tsetlin.ann")
local evaluator = require("santoku.tsetlin.evaluator")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")
local cvec = require("santoku.cvec")

local M = {}

local function key_int (v)
  return v and tostring(num.floor(v + 0.5)) or "nil"
end

local function key_int8 (v)
  return v and tostring(num.floor((v + 4) / 8) * 8) or "nil"
end

local function key_float2 (v)
  return v and str.format("%.2f", v) or "nil"
end

local function key_str (v)
  return v ~= nil and tostring(v) or "nil"
end

local function cmp_cost(a, b)
  if b == nil then return -1 end
  if type(a) == "number" then
    if a < b then return -1 end
    if a > b then return 1 end
    return 0
  end
  for i = 1, #a do
    if a[i] < b[i] then return -1 end
    if a[i] > b[i] then return 1 end
  end
  return 0
end

local function is_preferred(new_score, new_cost, best_score, best_cost, tolerance)
  if best_cost == nil then
    return true
  end
  if new_score > best_score + tolerance then
    return true
  end
  if new_score >= best_score - tolerance and cmp_cost(new_cost, best_cost) < 0 then
    return true
  end
  return false
end

local function sample_alpha_spec (spec, use_defaults)
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
  if use_defaults and spec.def ~= nil then
    return spec.def
  end
  local val
  if spec.log then
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

M.build_threshold_fn = function (threshold_params, captured_state)
  if not threshold_params then
    threshold_params = { method = "median" }
  end
  local method = threshold_params.method
  local state = captured_state or {}

  if method == "itq" then
    return function (codes, dims)
      if state.rotation and state.mean then
        return itq.itq({
          codes = codes,
          n_dims = dims,
          rotation = state.rotation,
          mean = state.mean,
        }), dims, state
      else
        local binary, rotation, mean = itq.itq({
          codes = codes,
          n_dims = dims,
          iterations = threshold_params.iterations or 100,
          tolerance = threshold_params.tolerance or 1e-8,
          return_rotation = true,
        })
        state.rotation = rotation
        state.mean = mean
        state.method = "itq"
        return binary, dims, state
      end
    end
  elseif method == "otsu" then
    return function (codes, dims)
      if state.indices and state.thresholds then
        return itq.otsu({
          codes = codes,
          n_dims = dims,
          indices = state.indices,
          thresholds = state.thresholds,
        }), dims, state
      else
        local binary, indices, scores, thresholds = itq.otsu({
          codes = codes,
          n_dims = dims,
          metric = threshold_params.metric or "variance",
          minimize = threshold_params.minimize or false,
          n_bins = threshold_params.n_bins or 32,
          return_thresholds = true,
        })
        state.indices = indices
        state.scores = scores
        state.thresholds = thresholds
        state.method = "otsu"
        return binary, dims, state
      end
    end
  elseif method == "median" then
    return function (codes, dims)
      if state.thresholds then
        return itq.median({
          codes = codes,
          n_dims = dims,
          thresholds = state.thresholds,
        }), dims, state
      else
        local binary, thresholds = itq.median({
          codes = codes,
          n_dims = dims,
          return_thresholds = true,
        })
        state.thresholds = thresholds
        state.method = "median"
        return binary, dims, state
      end
    end
  elseif method == "sign" then
    return function (codes, dims)
      state.method = "sign"
      return itq.sign({ codes = codes, n_dims = dims }), dims, state
    end
  else
    err.error("Unknown threshold method: " .. tostring(method))
  end
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
  if type(spec) == "table" and spec.min ~= nil and spec.max ~= nil then
    local minv, maxv = spec.min, spec.max
    err.assert(minv and maxv, "range spec missing min|max")
    local is_log = not not spec.log
    local is_int = not not spec.int
    local is_pow2 = not not spec.pow2
    local round_to = spec.round
    -- For log scale with non-positive min, shift range so shifted_min = 1
    local shift = (is_log and minv <= 0) and (1 - minv) or 0
    local smin = minv + shift
    local smax = maxv + shift
    local span = is_log and (num.log(smax) - num.log(smin)) or (maxv - minv)
    local init_jitter = (spec.dev or global_dev or 1.0) * span
    local jitter = init_jitter
    return {
      type = "range",
      center = spec.def,
      sample = function (center)
        local x
        if center then
          local c = is_log and num.log(center + shift) or center
          x = rand.fast_normal(c, jitter * jitter)
          if is_log then x = num.exp(x) - shift end
        else
          if is_log then
            x = num.exp(num.random() * span + num.log(smin)) - shift
          else
            x = num.random() * span + minv
          end
        end
        if x > maxv then x = 2 * maxv - x end
        if x < minv then x = 2 * minv - x end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        if is_pow2 then
          x = round_to_pow2(x)
        elseif round_to then
          x = num.floor(x / round_to + 0.5) * round_to
        elseif is_int then
          x = num.floor(x + 0.5)
        end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        return x
      end,
      adapt = function (factor)
        jitter = jitter * factor
        local min_jitter = init_jitter * 0.1
        local max_jitter = init_jitter * 2.0
        if jitter < min_jitter then jitter = min_jitter end
        if jitter > max_jitter then jitter = max_jitter end
      end,
    }
  end
  if type(spec) == "table" and #spec > 0 then
    return {
      type = "list",
      center = spec.def or spec[1],
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

M.sample_params = function (samplers, param_names, base_cfg, use_exact_defaults)
  local p = {}
  if base_cfg then
    for k, v in pairs(base_cfg) do
      p[k] = v
    end
  end
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s then
      if use_exact_defaults and s.center ~= nil then
        p[name] = s.center
      elseif s.type == "range" then
        p[name] = s.sample(s.center)
      else
        p[name] = s.sample()
      end
    end
  end
  return p
end

M.sample_tier = function (tier_cfg, use_defaults)
  local result = {}
  for k, v in pairs(tier_cfg) do
    if type(v) == "table" and #v > 0 then
      result[k] = use_defaults and (v.def or v[1]) or v[num.random(#v)]
    elseif type(v) == "table" and v.min ~= nil and v.max ~= nil then
      if use_defaults and v.def ~= nil then
        result[k] = v.def
      else
        local minv, maxv = v.min, v.max
        local is_log = v.log
        local is_int = v.int
        local is_pow2 = v.pow2
        local round_to = v.round
        local x
        if is_log then
          local shift = minv <= 0 and (1 - minv) or 0
          local smin = minv + shift
          local smax = maxv + shift
          local log_val = num.random() * (num.log(smax) - num.log(smin)) + num.log(smin)
          x = num.exp(log_val) - shift
        else
          x = num.random() * (maxv - minv) + minv
        end
        if x > maxv then x = 2 * maxv - x end
        if x < minv then x = 2 * minv - x end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        if is_pow2 then
          x = round_to_pow2(x)
        elseif round_to then
          x = num.floor(x / round_to + 0.5) * round_to
        elseif is_int then
          x = num.floor(x + 0.5)
        end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        result[k] = x
      end
    else
      result[k] = v
    end
  end
  if result.cgrams_min and result.cgrams_max and result.cgrams_min > result.cgrams_max then
    result.cgrams_min, result.cgrams_max = result.cgrams_max, result.cgrams_min
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

-- 1/5th success rule (bidirectional): adapt step size based on improvement rate
-- >25% improving -> expand (being too conservative)
-- 15-25% improving -> sweet spot, no change
-- 5-15% improving -> moderate shrink
-- <5% improving -> aggressive shrink
M.success_factor = function (success_rate)
  if success_rate > 0.25 then
    return 1.2
  elseif success_rate > 0.15 then
    return 1.0
  elseif success_rate > 0.05 then
    return 0.85
  else
    return 0.7
  end
end

M.search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local rounds = args.rounds or 3
  local trials = args.trials or 10
  local make_key = args.make_key
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local preference_tolerance = args.preference_tolerance or 1e-6
  local size_fn = args.size_fn or function() return 0 end
  local best_score = -num.huge
  local best_size = nil
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if M.all_fixed(samplers) or rounds <= 0 or trials <= 0 then
    best_params = M.sample_params(samplers, param_names, nil, true)
    if skip_final then
      return nil, best_params, nil
    else
      local _, metrics, result = trial_fn(best_params, { is_final = true })
      return result, best_params, metrics
    end
  end

  local seen = {}

  for r = 1, rounds do
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
        local current_size = size_fn(params)
        if is_preferred(score, current_size, best_score, best_size, preference_tolerance) then
          round_improvements = round_improvements + 1
          if best_result and cleanup_fn then
            cleanup_fn(best_result)
          end
          best_score = score
          best_size = current_size
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
    -- Recenter on global best and adapt jitter
    if best_params then
      M.recenter(samplers, param_names, best_params)
    end
    local skip_rate = (trials - round_samples) / trials
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor
    if skip_rate > 0.8 then
      adapt_factor = 1.3
    elseif round_samples < 3 then
      adapt_factor = 1.0
    else
      adapt_factor = M.success_factor(success_rate)
    end
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
        skip_rate = skip_rate,
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
      individualized = args.individualized,
      feat_offsets = args.feat_offsets,
      clauses = 8,
      clause_tolerance = 8,
      clause_maximum = 8,
      target = 4,
      specificity = 1000,
      include_bits = 1,
      reusable = true,
    })
  elseif typ == "classifier" then
    return tm.create("classifier", {
      features = args.features,
      classes = args.classes,
      individualized = args.individualized,
      feat_offsets = args.feat_offsets,
      clauses = 8,
      clause_tolerance = 8,
      clause_maximum = 8,
      target = 4,
      specificity = 1000,
      include_bits = 1,
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
      individualized = args.individualized,
      feat_offsets = args.feat_offsets,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
      include_bits = params.include_bits,
    })
  elseif typ == "classifier" then
    return tm.create("classifier", {
      features = args.features,
      classes = args.classes,
      individualized = args.individualized,
      feat_offsets = args.feat_offsets,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
      include_bits = params.include_bits,
    })
  else
    err.error("unexpected type", typ)
  end
end

local function train_tm (typ, tmobj, args, params, iterations, early_patience, metric_fn, each_cb, info, encoding_info)
  local best_epoch_score = -num.huge
  local last_epoch_score = -num.huge
  local last_metrics = nil
  local epochs_since_improve = 0
  local checkpoint = (early_patience and early_patience > 0) and cvec.create(0) or nil
  local has_checkpoint = false

  local enc_info = encoding_info or {
    sentences = args.sentences,
    samples = args.samples,
    dim_offsets = args.dim_offsets,
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
    if score > best_epoch_score then
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
      dim_offsets = args.dim_offsets,
      iterations = iterations,
      each = on_epoch,
    })
  elseif typ == "classifier" then
    tmobj:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      dim_offsets = args.dim_offsets,
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

local function optimize_tm (args, typ)

  local patience = args.search_patience or 10
  local use_early_stop = patience > 0
  local final_patience = args.final_patience or 40
  local use_final_early_stop = final_patience > 0
  local iters_search = args.search_iterations or 10
  local final_iters = args.final_iterations or (iters_search * 10)
  local global_dev = args.search_dev or 0.2
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local param_names = { "clauses", "clause_tolerance", "clause_maximum", "target", "specificity", "include_bits" }
  local samplers = M.build_samplers(args, param_names, global_dev)

  local search_tm
  if not M.all_fixed(samplers) then
    search_tm = create_tm(typ, {
      visible = args.visible,
      hidden = args.hidden,
      features = args.features,
      classes = args.classes,
      individualized = args.individualized,
      feat_offsets = args.feat_offsets,
    })
  end

  local function search_trial_fn (params, info)
    if params.clause_tolerance and params.clause_maximum and params.clause_tolerance > params.clause_maximum then
      params.clause_tolerance, params.clause_maximum = params.clause_maximum, params.clause_tolerance
    end
    search_tm:reconfigure(params)
    local train_args = {
      sentences = args.sentences,
      codes = args.codes,
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      dim_offsets = args.dim_offsets,
    }
    local encoding_info = {
      sentences = args.sentences,
      visible = args.visible,
      samples = args.samples,
      dim_offsets = args.dim_offsets,
    }
    local score, metrics = train_tm(typ, search_tm, train_args, params, iters_search,
      use_early_stop and patience or nil, metric_fn, each_cb, info, encoding_info)
    return score, metrics, nil
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    rounds = args.search_rounds or 3,
    trials = args.search_trials or 10,
    trial_fn = search_trial_fn,
    skip_final = true,
    preference_tolerance = args.preference_tolerance or 1e-6,
    size_fn = function(p) return { p.clauses or 0, -(p.specificity or 0) } end,
    make_key = function (p)
      return str.format("%s|%s|%s|%s|%s|%s",
        key_int8(p.clauses),
        key_int(p.clause_tolerance),
        key_int(p.clause_maximum),
        key_int(p.target),
        key_int(p.specificity),
        key_int(p.include_bits or 1))
    end,
  })

  if search_tm then
    search_tm:destroy()
  end

  local final_tm = create_final_tm(typ, {
    visible = args.visible,
    hidden = args.hidden,
    features = args.features,
    classes = args.classes,
    individualized = args.individualized,
    feat_offsets = args.feat_offsets,
  }, best_params)
  local final_train_args = {
    sentences = args.sentences,
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
    dim_offsets = args.dim_offsets,
  }
  local final_encoding_info = {
    sentences = args.sentences,
    visible = args.visible,
    samples = args.samples,
    dim_offsets = args.dim_offsets,
  }
  local _, final_metrics = train_tm(typ, final_tm, final_train_args, best_params, final_iters,
    use_final_early_stop and final_patience or nil, metric_fn, each_cb, { is_final = true }, final_encoding_info)

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
  if model.raw_model then
    M.destroy_spectral_raw(model.raw_model)
    model.raw_model = nil
  end
  if model.owns_base ~= false then
    if model.ids then model.ids:destroy() end
    if model.eigs then model.eigs:destroy() end
  end
  if model.index then model.index:destroy() end
  if model.codes then model.codes:destroy() end
  if model.raw_codes and model.owns_raw_codes ~= false then model.raw_codes:destroy() end
  if model.selected_indices then model.selected_indices:destroy() end
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
  local seed_ids = args.seed_ids
  local seed_offsets = args.seed_offsets
  local seed_neighbors = args.seed_neighbors

  return graph.adjacency({
    weight_index = index,
    weight_cmp = p.weight_cmp,
    weight_decay = p.weight_decay,
    weight_pooling = p.weight_pooling,
    knn_index = knn_index,
    knn = p.knn,
    knn_mode = p.knn_mode,
    knn_alpha = p.knn_alpha,
    knn_mutual = p.knn_mutual,
    knn_cache = seed_ids and 0 or (p.knn_cache or 32),
    knn_cmp = p.knn_cmp,
    knn_cmp_alpha = p.knn_cmp_alpha,
    knn_cmp_beta = p.knn_cmp_beta,
    knn_eps = p.knn_eps,
    knn_min = p.knn_min,
    knn_anchor = p.knn_anchor,
    knn_seed = seed_ids and true or p.knn_seed,
    knn_bipartite = p.knn_bipartite,
    seed_ids = seed_ids,
    seed_offsets = seed_offsets,
    seed_neighbors = seed_neighbors,
    probe_radius = p.probe_radius,
    category_cmp = p.category_cmp,
    category_alpha = p.category_alpha,
    category_beta = p.category_beta,
    bridge = p.bridge,
    each = each,
  })
end

M.build_spectral_raw = function (args)
  local adj_ids, adj_offsets, adj_neighbors, adj_weights = args.adj_ids, args.adj_offsets, args.adj_neighbors, args.adj_weights
  local p = args.params
  local each = args.each

  local n_matvecs = 0
  local wrapped_each = each and function (event, matvecs)
    if event == "done" then
      n_matvecs = matvecs
    end
    each(event, matvecs)
  end or function (event, matvecs)
    if event == "done" then
      n_matvecs = matvecs
    end
  end

  local ids, raw_codes, _, eigs = spectral.encode({
    type = p.laplacian or "unnormalized",
    n_hidden = p.n_dims,
    eps = p.eps or 1e-10,
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    weights = adj_weights,
    each = wrapped_each,
  })

  return {
    ids = ids,
    raw_codes = raw_codes,
    eigs = eigs,
    dims = p.n_dims,
    n_matvecs = n_matvecs,
  }
end

M.apply_threshold = function (args)
  local raw_model = args.raw_model
  local threshold_params = args.threshold_params
  local threshold_fn = args.threshold_fn
  local bucket_size = args.bucket_size

  local raw_codes = raw_model.raw_codes
  local dims = raw_model.dims

  threshold_fn = threshold_fn or M.build_threshold_fn(threshold_params)
  local codes, _, threshold_state = threshold_fn(raw_codes, dims)

  local ann_index = ann.create({
    expected_size = raw_model.ids:size(),
    bucket_size = bucket_size,
    features = dims,
  })
  ann_index:add(codes, raw_model.ids)

  return {
    ids = raw_model.ids,
    codes = codes,
    raw_codes = raw_codes,
    threshold_fn = threshold_fn,
    threshold_params = threshold_params,
    threshold_state = threshold_state,
    eigs = raw_model.eigs,
    dims = dims,
    index = ann_index,
    owns_base = false,
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

  local adj_retrieved_ids, adj_retrieved_offsets, adj_retrieved_neighbors, adj_retrieved_weights =
    graph.adjacency({
      weight_index = model.index,
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
    n_dims = model.dims,
  })

  if adj_retrieved_ids then adj_retrieved_ids:destroy() end
  if adj_retrieved_offsets then adj_retrieved_offsets:destroy() end
  if adj_retrieved_neighbors then adj_retrieved_neighbors:destroy() end
  if adj_retrieved_weights then adj_retrieved_weights:destroy() end

  return stats.score, {
    score = stats.score,
    n_dims = model.dims,
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
  local query_ids = args.query_ids

  local prebuilt_adj_ids = args.adj_ids
  local prebuilt_adj_offsets = args.adj_offsets
  local prebuilt_adj_neighbors = args.adj_neighbors
  local prebuilt_adj_weights = args.adj_weights
  local has_prebuilt_adj = prebuilt_adj_ids ~= nil

  local adjacency_cfg = has_prebuilt_adj and {} or err.assert(args.adjacency, "adjacency config required")
  local spectral_cfg = err.assert(args.spectral, "spectral config required")
  local eval_cfg = err.assert(args.eval, "eval config required")

  local adjacency_samples = has_prebuilt_adj and 1 or (args.adjacency_samples or 1)
  local spectral_samples = args.spectral_samples or 1
  local select_samples = args.select_samples or 1
  local eval_samples = args.eval_samples or 1
  local rounds = args.search_rounds or 1
  local global_dev = args.search_dev or 0.2
  local preference_tolerance = args.preference_tolerance or 1e-6

  local expected = {
    ids = args.expected_ids,
    offsets = args.expected_offsets,
    neighbors = args.expected_neighbors,
    weights = args.expected_weights,
  }

  local best_score = -num.huge
  local best_cost = nil
  local best_params = nil
  local best_model = nil
  local best_metrics = nil
  local best_adj_ids = nil
  local best_adj_offsets = nil
  local best_adj_neighbors = nil
  local best_adj_weights = nil
  local sample_count = 0

  local adj_fixed = has_prebuilt_adj or M.tier_all_fixed(adjacency_cfg)
  local spec_fixed = M.tier_all_fixed(spectral_cfg)
  local eval_fixed = M.tier_all_fixed(eval_cfg)

  local all_fixed = adj_fixed and spec_fixed and eval_fixed

  -- When rounds <= 0 or all tiers are fixed, skip search and build single model with defaults
  if rounds <= 0 or all_fixed then
    local adj_params = has_prebuilt_adj and { prebuilt = true } or M.sample_tier(adjacency_cfg, true)
    local spec_params = M.sample_tier(spectral_cfg, true)
    local eval_params = M.sample_tier(eval_cfg, true)

    local threshold_params, threshold_fn = M.sample_threshold(spectral_cfg.threshold)

    if each_cb then
      each_cb({ event = "stage", stage = "adjacency", is_final = true, params = { adjacency = adj_params } })
    end

    local adj_ids, adj_offsets, adj_neighbors, adj_weights
    if has_prebuilt_adj then
      adj_ids = prebuilt_adj_ids
      adj_offsets = prebuilt_adj_offsets
      adj_neighbors = prebuilt_adj_neighbors
      adj_weights = prebuilt_adj_weights
    else
      adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
        index = index,
        knn_index = knn_index,
        params = adj_params,
        each = adjacency_each,
      })
    end

    if each_cb then
      each_cb({ event = "stage", stage = "spectral", is_final = true, params = { adjacency = adj_params, spectral = spec_params } })
    end

    local raw_model = M.build_spectral_raw({
      adj_ids = adj_ids,
      adj_offsets = adj_offsets,
      adj_neighbors = adj_neighbors,
      adj_weights = adj_weights,
      params = spec_params,
      each = spectral_each,
    })

    if each_cb then
      each_cb({
        event = "spectral_result",
        n_nodes = raw_model.ids:size(),
        n_dims = raw_model.dims,
        n_matvecs = raw_model.n_matvecs,
        eig_min = raw_model.eigs and raw_model.eigs:min() or nil,
        eig_max = raw_model.eigs and raw_model.eigs:max() or nil,
        params = { adjacency = adj_params, spectral = spec_params },
      })
    end

    local model = M.apply_threshold({
      raw_model = raw_model,
      threshold_params = threshold_params,
      threshold_fn = threshold_fn,
      bucket_size = bucket_size,
    })
    model.owns_base = true
    model.owns_raw_codes = true

    model.adj_ids = adj_ids
    model.adj_offsets = adj_offsets
    model.adj_neighbors = adj_neighbors
    model.adj_weights = adj_weights

    local best_params = { adjacency = adj_params, spectral = spec_params, eval = eval_params }

    local _, metrics = M.score_spectral_eval({
      model = model,
      eval_params = eval_params,
      expected = expected,
    })

    return model, best_params, metrics
  end

  -- Build samplers for hill climbing (adjacency and spectral tiers)
  local adj_param_names = {
    "knn", "knn_alpha", "knn_mutual", "knn_mode", "knn_cache",
    "knn_cmp", "knn_cmp_alpha", "knn_cmp_beta", "knn_eps", "knn_min",
    "knn_anchor", "knn_seed", "knn_bipartite",
    "weight_cmp", "weight_decay", "weight_pooling",
    "category_cmp", "category_alpha", "category_beta",
    "probe_radius", "bridge",
  }
  local spec_param_names = { "n_dims", "laplacian", "eps" }
  local adj_samplers = M.build_samplers(adjacency_cfg, adj_param_names, global_dev)
  local spec_samplers = M.build_samplers(spectral_cfg, spec_param_names, global_dev)

  local function make_adj_key (p)
    return str.format("%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s",
      key_int(p.knn), key_int(p.knn_alpha), key_str(p.knn_mutual), key_str(p.knn_mode),
      key_int(p.knn_cache), key_str(p.knn_cmp), key_float2(p.knn_cmp_alpha),
      key_float2(p.knn_cmp_beta), key_float2(p.knn_eps), key_int(p.knn_min),
      key_str(p.knn_anchor), key_str(p.knn_seed), key_str(p.knn_bipartite),
      key_str(p.weight_cmp), key_float2(p.weight_decay), key_str(p.weight_pooling),
      key_str(p.category_cmp), key_float2(p.category_alpha), key_float2(p.category_beta),
      key_int(p.probe_radius), key_str(p.bridge))
  end

  local function make_spec_key (adj_key, p)
    return str.format("%s|%s|%s", adj_key, key_int(p.n_dims), key_str(p.laplacian))
  end

  local function make_select_key (spec_key, p)
    local thresh_str = p.threshold_params and p.threshold_params.method or "none"
    return str.format("%s|%s", spec_key, thresh_str)
  end

  local function make_eval_key (select_key, p)
    return str.format("%s|%s", select_key, p.ranking)
  end

  local function mem_kb ()
    return collectgarbage("count")
  end

  local seen = {}

  for round = 1, rounds do
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

  for adj_i = 1, (adj_fixed and 1 or adjacency_samples) do
    local adj_params = has_prebuilt_adj and { prebuilt = true } or M.sample_params(adj_samplers, adj_param_names, adjacency_cfg)
    local adj_key = has_prebuilt_adj and "prebuilt" or make_adj_key(adj_params)

    if seen[adj_key] then
      if each_cb then
        each_cb({
          event = "adjacency_cached",
          round = round,
          adj_sample = adj_i,
          adj_key = adj_key,
          params = { adjacency = adj_params },
        })
      end
    else
      seen[adj_key] = true
      sample_count = sample_count + 1

      local adj_mem_before = mem_kb()
      if each_cb then
        each_cb({
          event = "stage",
          stage = "adjacency",
          round = round,
          sample = sample_count,
          adj_sample = adj_i,
          adj_key = adj_key,
          is_final = false,
          params = { adjacency = adj_params },
          mem_kb = adj_mem_before,
        })
      end

      local adj_ids, adj_offsets, adj_neighbors, adj_weights
      if has_prebuilt_adj then
        adj_ids = prebuilt_adj_ids
        adj_offsets = prebuilt_adj_offsets
        adj_neighbors = prebuilt_adj_neighbors
        adj_weights = prebuilt_adj_weights
      else
        adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
          index = index,
          knn_index = knn_index,
          params = adj_params,
          each = adjacency_each,
        })
      end

      if each_cb then
        each_cb({
          event = "adjacency_result",
          round = round,
          adj_sample = adj_i,
          n_nodes = adj_ids:size(),
          n_edges = adj_neighbors:size(),
          params = { adjacency = adj_params },
        })
      end

      for _ = 1, (spec_fixed and 1 or spectral_samples) do
        local spec_params = M.sample_params(spec_samplers, spec_param_names, spectral_cfg)
        local spec_key = make_spec_key(adj_key, spec_params)

        if not seen[spec_key] then
          seen[spec_key] = true

          local spec_mem_before = mem_kb()
          if each_cb then
            each_cb({
              event = "stage",
              stage = "spectral",
              round = round,
              sample = sample_count,
              is_final = false,
              params = { adjacency = adj_params, spectral = spec_params },
              mem_kb = spec_mem_before,
            })
          end

          local raw_model = M.build_spectral_raw({
            adj_ids = adj_ids,
            adj_offsets = adj_offsets,
            adj_neighbors = adj_neighbors,
            adj_weights = adj_weights,
            params = spec_params,
            each = spectral_each,
          })

          if each_cb then
            each_cb({
              event = "spectral_result",
              n_nodes = raw_model.ids:size(),
              n_dims = raw_model.dims,
              n_matvecs = raw_model.n_matvecs,
              eig_min = raw_model.eigs and raw_model.eigs:min() or nil,
              eig_max = raw_model.eigs and raw_model.eigs:max() or nil,
              params = { adjacency = adj_params, spectral = spec_params },
            })
          end

          local raw_model_has_best = false

          for _ = 1, select_samples do
            local select_params = {}

            local threshold_params, threshold_fn = M.sample_threshold(spectral_cfg.threshold)
            if threshold_params then
              select_params.threshold_params = threshold_params
            elseif threshold_fn then
              select_params.threshold_fn = threshold_fn
            end

            local select_key = make_select_key(spec_key, select_params)

            if not seen[select_key] then
              seen[select_key] = true

              if each_cb then
                each_cb({
                  event = "stage",
                  stage = "select",
                  round = round,
                  sample = sample_count,
                  is_final = false,
                  params = { adjacency = adj_params, spectral = spec_params, select = select_params },
                  mem_kb = mem_kb(),
                })
              end

              local model = M.apply_threshold({
                raw_model = raw_model,
                threshold_params = select_params.threshold_params,
                threshold_fn = select_params.threshold_fn,
                bucket_size = bucket_size,
              })

              if each_cb then
                each_cb({
                  event = "select_result",
                  round = round,
                  params = { adjacency = adj_params, spectral = spec_params, select = select_params },
                  selected_dims = model.dims,
                })
              end

              local model_is_best = false
              local consecutive_zeros = 0
              local max_consecutive_zeros = eval_cfg.max_consecutive_zeros or 3

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
                eval_params.query_ids = query_ids

                local eval_key = make_eval_key(select_key, eval_params)

                if not seen[eval_key] then
                  seen[eval_key] = true

                  local score, metrics = M.score_spectral_eval({
                    model = model,
                    eval_params = eval_params,
                    expected = expected,
                  })

                  if each_cb then
                    each_cb({
                      event = "eval",
                      round = round,
                      sample = sample_count,
                      is_final = false,
                      params = { adjacency = adj_params, spectral = spec_params, select = select_params, eval = eval_params },
                      score = score,
                      metrics = metrics,
                      mem_kb = mem_kb(),
                    })
                  end

                  if score <= 0 then
                    consecutive_zeros = consecutive_zeros + 1
                  else
                    consecutive_zeros = 0
                  end

                  round_samples = round_samples + 1

                  if score > round_best_score then
                    round_best_score = score
                    round_best_params = { adjacency = adj_params, spectral = spec_params, select = select_params, eval = eval_params }
                  end

                  local current_cost = { adj_params.weight_decay or 0, spec_params.n_dims or 0, adj_neighbors:size() }
                  if is_preferred(score, current_cost, best_score, best_cost, preference_tolerance) then
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
                    best_cost = current_cost
                    best_params = { adjacency = adj_params, spectral = spec_params, select = select_params, eval = eval_params }
                    best_model = model
                    best_model.raw_model = raw_model
                    best_metrics = metrics
                    best_adj_ids = adj_ids
                    best_adj_offsets = adj_offsets
                    best_adj_neighbors = adj_neighbors
                    best_adj_weights = adj_weights
                    model_is_best = true
                    raw_model_has_best = true
                  end
                end
              end

              if not model_is_best then
                M.destroy_spectral(model)
              end
            end
          end

          if not raw_model_has_best then
            M.destroy_spectral_raw(raw_model)
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

    -- Recenter on global best and adapt jitter
    if best_params then
      M.recenter(adj_samplers, adj_param_names, best_params.adjacency)
      M.recenter(spec_samplers, spec_param_names, best_params.spectral)
    end
    local expected_samples = adjacency_samples * spectral_samples * select_samples * eval_samples
    local skip_rate = 1 - (round_samples / expected_samples)
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor
    if skip_rate > 0.8 then
      adapt_factor = 1.3
    elseif round_samples < 3 then
      adapt_factor = 1.0
    else
      adapt_factor = M.success_factor(success_rate)
    end
    M.adapt_jitter(adj_samplers, adapt_factor)
    M.adapt_jitter(spec_samplers, adapt_factor)

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
        skip_rate = skip_rate,
        adapt_factor = adapt_factor,
        mem_kb = round_end_mem,
        mem_delta_kb = round_end_mem - round_start_mem,
      })
    end

  end

  if best_model then
    best_model.owns_base = true
    best_model.owns_raw_codes = true
    best_model.adj_ids = best_adj_ids
    best_model.adj_offsets = best_adj_offsets
    best_model.adj_neighbors = best_adj_neighbors
    best_model.adj_weights = best_adj_weights
  end

  collectgarbage("collect")
  return best_model, best_params, best_metrics
end

M.destroy_prone = function (model)
  if not model then return end
  if model.raw_model then
    M.destroy_prone_raw(model.raw_model)
    model.raw_model = nil
  end
  if model.owns_base ~= false then
    if model.ids then model.ids:destroy() end
  end
  if model.index then model.index:destroy() end
  if model.codes then model.codes:destroy() end
  if model.raw_codes and model.owns_raw_codes ~= false then model.raw_codes:destroy() end
  if model.adj_ids then model.adj_ids:destroy() end
  if model.adj_offsets then model.adj_offsets:destroy() end
  if model.adj_neighbors then model.adj_neighbors:destroy() end
  if model.adj_weights then model.adj_weights:destroy() end
end

M.build_prone_raw = function (args)
  local adj_ids, adj_offsets, adj_neighbors, adj_weights = args.adj_ids, args.adj_offsets, args.adj_neighbors, args.adj_weights
  local p = args.params
  local each = args.each

  local ids, raw_codes = prone.encode({
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    weights = adj_weights,
    n_hidden = p.n_dims,
    n_iter = p.n_iter or 5,
    step = p.step or 10,
    mu = p.mu or 0.2,
    theta = p.theta or 0.5,
    neg_samples = p.neg_samples or 5,
    propagate = p.propagate ~= false,
    each = each,
  })

  return {
    ids = ids,
    raw_codes = raw_codes,
    dims = p.n_dims,
  }
end

M.destroy_prone_raw = function (raw_model)
  if not raw_model then return end
  if raw_model.raw_codes then raw_model.raw_codes:destroy() end
end

M.build_prone_from_adjacency = function (args)
  local p = args.params
  local bucket_size = args.bucket_size

  local raw_model = M.build_prone_raw({
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

M.score_prone_eval = function (args)
  local model = args.model
  local eval_params = args.eval_params
  local expected = args.expected

  local adj_retrieved_ids, adj_retrieved_offsets, adj_retrieved_neighbors, adj_retrieved_weights =
    graph.adjacency({
      weight_index = model.index,
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
    n_dims = model.dims,
  })

  if adj_retrieved_ids then adj_retrieved_ids:destroy() end
  if adj_retrieved_offsets then adj_retrieved_offsets:destroy() end
  if adj_retrieved_neighbors then adj_retrieved_neighbors:destroy() end
  if adj_retrieved_weights then adj_retrieved_weights:destroy() end

  return stats.score, {
    score = stats.score,
    n_dims = model.dims,
    total_queries = stats.total_queries,
  }
end

M.prone = function (args)
  local index = args.index
  local knn_index = args.knn_index
  local bucket_size = args.bucket_size
  local each_cb = args.each
  local adjacency_each = args.adjacency_each
  local prone_each = args.prone_each
  local query_ids = args.query_ids

  local prebuilt_adj_ids = args.adj_ids
  local prebuilt_adj_offsets = args.adj_offsets
  local prebuilt_adj_neighbors = args.adj_neighbors
  local prebuilt_adj_weights = args.adj_weights
  local has_prebuilt_adj = prebuilt_adj_ids ~= nil

  local seed_ids = args.seed_ids
  local seed_offsets = args.seed_offsets
  local seed_neighbors = args.seed_neighbors

  local adjacency_cfg = has_prebuilt_adj and {} or err.assert(args.adjacency, "adjacency config required")
  local prone_cfg = err.assert(args.prone, "prone config required")
  local eval_cfg = err.assert(args.eval, "eval config required")

  local adjacency_samples = has_prebuilt_adj and 1 or (args.adjacency_samples or 1)
  local prone_samples = args.prone_samples or 1
  local select_samples = args.select_samples or 1
  local eval_samples = args.eval_samples or 1
  local rounds = args.search_rounds or 1
  local global_dev = args.search_dev or 0.2
  local preference_tolerance = args.preference_tolerance or 1e-6

  local expected = {
    ids = args.expected_ids,
    offsets = args.expected_offsets,
    neighbors = args.expected_neighbors,
    weights = args.expected_weights,
  }

  local best_score = -num.huge
  local best_cost = nil
  local best_params = nil
  local best_model = nil
  local best_metrics = nil
  local best_adj_ids = nil
  local best_adj_offsets = nil
  local best_adj_neighbors = nil
  local best_adj_weights = nil
  local sample_count = 0

  local adj_fixed = has_prebuilt_adj or M.tier_all_fixed(adjacency_cfg)
  local prone_fixed = M.tier_all_fixed(prone_cfg)
  local eval_fixed = M.tier_all_fixed(eval_cfg)

  local all_fixed = adj_fixed and prone_fixed and eval_fixed

  if rounds <= 0 or all_fixed then
    local adj_params = has_prebuilt_adj and { prebuilt = true } or M.sample_tier(adjacency_cfg, true)
    local prone_params = M.sample_tier(prone_cfg, true)
    local eval_params = M.sample_tier(eval_cfg, true)

    local threshold_params, threshold_fn = M.sample_threshold(prone_cfg.threshold)

    if each_cb then
      each_cb({ event = "stage", stage = "adjacency", is_final = true, params = { adjacency = adj_params } })
    end

    local adj_ids, adj_offsets, adj_neighbors, adj_weights
    if has_prebuilt_adj then
      adj_ids = prebuilt_adj_ids
      adj_offsets = prebuilt_adj_offsets
      adj_neighbors = prebuilt_adj_neighbors
      adj_weights = prebuilt_adj_weights
    else
      adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
        index = index,
        knn_index = knn_index,
        params = adj_params,
        seed_ids = seed_ids,
        seed_offsets = seed_offsets,
        seed_neighbors = seed_neighbors,
        each = adjacency_each,
      })
    end

    if each_cb then
      each_cb({ event = "stage", stage = "prone", is_final = true, params = { adjacency = adj_params, prone = prone_params } })
    end

    local raw_model = M.build_prone_raw({
      adj_ids = adj_ids,
      adj_offsets = adj_offsets,
      adj_neighbors = adj_neighbors,
      adj_weights = adj_weights,
      params = prone_params,
      each = prone_each,
    })

    if each_cb then
      each_cb({
        event = "prone_result",
        n_nodes = raw_model.ids:size(),
        n_dims = raw_model.dims,
        params = { adjacency = adj_params, prone = prone_params },
      })
    end

    local model = M.apply_threshold({
      raw_model = raw_model,
      threshold_params = threshold_params,
      threshold_fn = threshold_fn,
      bucket_size = bucket_size,
    })
    model.owns_base = true
    model.owns_raw_codes = true

    model.adj_ids = adj_ids
    model.adj_offsets = adj_offsets
    model.adj_neighbors = adj_neighbors
    model.adj_weights = adj_weights

    local best_params = { adjacency = adj_params, prone = prone_params, eval = eval_params }

    local _, metrics = M.score_prone_eval({
      model = model,
      eval_params = eval_params,
      expected = expected,
    })

    return model, best_params, metrics
  end

  local adj_param_names = {
    "knn", "knn_alpha", "knn_mutual", "knn_mode", "knn_cache",
    "knn_cmp", "knn_cmp_alpha", "knn_cmp_beta", "knn_eps", "knn_min",
    "knn_anchor", "knn_seed", "knn_bipartite",
    "weight_cmp", "weight_decay", "weight_pooling",
    "category_cmp", "category_alpha", "category_beta",
    "probe_radius", "bridge",
  }
  local prone_param_names = { "n_dims", "n_iter", "step", "mu", "theta", "neg_samples", "propagate" }
  local adj_samplers = M.build_samplers(adjacency_cfg, adj_param_names, global_dev)
  local prone_samplers = M.build_samplers(prone_cfg, prone_param_names, global_dev)

  local function make_adj_key (p)
    return str.format("%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s",
      key_int(p.knn), key_int(p.knn_alpha), key_str(p.knn_mutual), key_str(p.knn_mode),
      key_int(p.knn_cache), key_str(p.knn_cmp), key_float2(p.knn_cmp_alpha),
      key_float2(p.knn_cmp_beta), key_float2(p.knn_eps), key_int(p.knn_min),
      key_str(p.knn_anchor), key_str(p.knn_seed), key_str(p.knn_bipartite),
      key_str(p.weight_cmp), key_float2(p.weight_decay), key_str(p.weight_pooling),
      key_str(p.category_cmp), key_float2(p.category_alpha), key_float2(p.category_beta),
      key_int(p.probe_radius), key_str(p.bridge))
  end

  local function make_prone_key (adj_key, p)
    return str.format("%s|%s|%s|%s|%s|%s|%s|%s",
      adj_key, key_int(p.n_dims), key_int(p.n_iter), key_int(p.step),
      key_float2(p.mu), key_float2(p.theta), key_int(p.neg_samples), key_str(p.propagate))
  end

  local function make_select_key (prone_key, p)
    local thresh_str = p.threshold_params and p.threshold_params.method or "none"
    return str.format("%s|%s", prone_key, thresh_str)
  end

  local function make_eval_key (select_key, p)
    return str.format("%s|%s", select_key, p.ranking)
  end

  local function mem_kb ()
    return collectgarbage("count")
  end

  local seen = {}

  for round = 1, rounds do
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

  for adj_i = 1, (adj_fixed and 1 or adjacency_samples) do
    local adj_params = has_prebuilt_adj and { prebuilt = true } or M.sample_params(adj_samplers, adj_param_names, adjacency_cfg)
    local adj_key = has_prebuilt_adj and "prebuilt" or make_adj_key(adj_params)

    if seen[adj_key] then
      if each_cb then
        each_cb({
          event = "adjacency_cached",
          round = round,
          adj_sample = adj_i,
          adj_key = adj_key,
          params = { adjacency = adj_params },
        })
      end
    else
      seen[adj_key] = true
      sample_count = sample_count + 1

      local adj_mem_before = mem_kb()
      if each_cb then
        each_cb({
          event = "stage",
          stage = "adjacency",
          round = round,
          sample = sample_count,
          adj_sample = adj_i,
          adj_key = adj_key,
          is_final = false,
          params = { adjacency = adj_params },
          mem_kb = adj_mem_before,
        })
      end

      local adj_ids, adj_offsets, adj_neighbors, adj_weights
      if has_prebuilt_adj then
        adj_ids = prebuilt_adj_ids
        adj_offsets = prebuilt_adj_offsets
        adj_neighbors = prebuilt_adj_neighbors
        adj_weights = prebuilt_adj_weights
      else
        adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
          index = index,
          knn_index = knn_index,
          params = adj_params,
          seed_ids = seed_ids,
          seed_offsets = seed_offsets,
          seed_neighbors = seed_neighbors,
          each = adjacency_each,
        })
      end

      if each_cb then
        each_cb({
          event = "adjacency_result",
          round = round,
          adj_sample = adj_i,
          n_nodes = adj_ids:size(),
          n_edges = adj_neighbors:size(),
          params = { adjacency = adj_params },
          has_seeds = seed_ids ~= nil,
        })
      end

      for _ = 1, (prone_fixed and 1 or prone_samples) do
        local prone_params = M.sample_params(prone_samplers, prone_param_names, prone_cfg)
        local prone_key = make_prone_key(adj_key, prone_params)

        if not seen[prone_key] then
          seen[prone_key] = true

          local prone_mem_before = mem_kb()
          if each_cb then
            each_cb({
              event = "stage",
              stage = "prone",
              round = round,
              sample = sample_count,
              is_final = false,
              params = { adjacency = adj_params, prone = prone_params },
              mem_kb = prone_mem_before,
            })
          end

          local raw_model = M.build_prone_raw({
            adj_ids = adj_ids,
            adj_offsets = adj_offsets,
            adj_neighbors = adj_neighbors,
            adj_weights = adj_weights,
            params = prone_params,
            each = prone_each,
          })

          if each_cb then
            each_cb({
              event = "prone_result",
              n_nodes = raw_model.ids:size(),
              n_dims = raw_model.dims,
              params = { adjacency = adj_params, prone = prone_params },
            })
          end

          local raw_model_has_best = false

          for _ = 1, select_samples do
            local select_params = {}

            local threshold_params, threshold_fn = M.sample_threshold(prone_cfg.threshold)
            if threshold_params then
              select_params.threshold_params = threshold_params
            elseif threshold_fn then
              select_params.threshold_fn = threshold_fn
            end

            local select_key = make_select_key(prone_key, select_params)

            if not seen[select_key] then
              seen[select_key] = true

              if each_cb then
                each_cb({
                  event = "stage",
                  stage = "select",
                  round = round,
                  sample = sample_count,
                  is_final = false,
                  params = { adjacency = adj_params, prone = prone_params, select = select_params },
                  mem_kb = mem_kb(),
                })
              end

              local model = M.apply_threshold({
                raw_model = raw_model,
                threshold_params = select_params.threshold_params,
                threshold_fn = select_params.threshold_fn,
                bucket_size = bucket_size,
              })

              if each_cb then
                each_cb({
                  event = "select_result",
                  round = round,
                  params = { adjacency = adj_params, prone = prone_params, select = select_params },
                  selected_dims = model.dims,
                })
              end

              local model_is_best = false
              local consecutive_zeros = 0
              local max_consecutive_zeros = eval_cfg.max_consecutive_zeros or 3

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
                eval_params.query_ids = query_ids

                local eval_key = make_eval_key(select_key, eval_params)

                if not seen[eval_key] then
                  seen[eval_key] = true

                  local score, metrics = M.score_prone_eval({
                    model = model,
                    eval_params = eval_params,
                    expected = expected,
                  })

                  if each_cb then
                    each_cb({
                      event = "eval",
                      round = round,
                      sample = sample_count,
                      is_final = false,
                      params = { adjacency = adj_params, prone = prone_params, select = select_params, eval = eval_params },
                      score = score,
                      metrics = metrics,
                      mem_kb = mem_kb(),
                    })
                  end

                  if score <= 0 then
                    consecutive_zeros = consecutive_zeros + 1
                  else
                    consecutive_zeros = 0
                  end

                  round_samples = round_samples + 1

                  if score > round_best_score then
                    round_best_score = score
                    round_best_params = { adjacency = adj_params, prone = prone_params, select = select_params, eval = eval_params }
                  end

                  local current_cost = { adj_params.weight_decay or 0, prone_params.n_dims or 0, adj_neighbors:size() }
                  if is_preferred(score, current_cost, best_score, best_cost, preference_tolerance) then
                    round_improvements = round_improvements + 1
                    if best_model and best_model ~= model then
                      M.destroy_prone(best_model)
                    end
                    if best_adj_ids and best_adj_ids ~= adj_ids then
                      best_adj_ids:destroy()
                      best_adj_offsets:destroy()
                      best_adj_neighbors:destroy()
                      if best_adj_weights then best_adj_weights:destroy() end
                    end
                    best_score = score
                    best_cost = current_cost
                    best_params = { adjacency = adj_params, prone = prone_params, select = select_params, eval = eval_params }
                    best_model = model
                    best_model.raw_model = raw_model
                    best_metrics = metrics
                    best_adj_ids = adj_ids
                    best_adj_offsets = adj_offsets
                    best_adj_neighbors = adj_neighbors
                    best_adj_weights = adj_weights
                    model_is_best = true
                    raw_model_has_best = true
                  end
                end
              end

              if not model_is_best then
                M.destroy_prone(model)
              end
            end
          end

          if not raw_model_has_best then
            M.destroy_prone_raw(raw_model)
          end
          collectgarbage("collect")

          if each_cb then
            each_cb({
              event = "prone_done",
              mem_kb = mem_kb(),
              mem_before = prone_mem_before,
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

    if best_params then
      M.recenter(adj_samplers, adj_param_names, best_params.adjacency)
      M.recenter(prone_samplers, prone_param_names, best_params.prone)
    end
    local expected_samples = adjacency_samples * prone_samples * select_samples * eval_samples
    local skip_rate = 1 - (round_samples / expected_samples)
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor
    if skip_rate > 0.8 then
      adapt_factor = 1.3
    elseif round_samples < 3 then
      adapt_factor = 1.0
    else
      adapt_factor = M.success_factor(success_rate)
    end
    M.adapt_jitter(adj_samplers, adapt_factor)
    M.adapt_jitter(prone_samplers, adapt_factor)

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
        skip_rate = skip_rate,
        adapt_factor = adapt_factor,
        mem_kb = round_end_mem,
        mem_delta_kb = round_end_mem - round_start_mem,
      })
    end

  end

  if best_model then
    best_model.owns_base = true
    best_model.owns_raw_codes = true
    best_model.adj_ids = best_adj_ids
    best_model.adj_offsets = best_adj_offsets
    best_model.adj_neighbors = best_adj_neighbors
    best_model.adj_weights = best_adj_weights
  end

  collectgarbage("collect")
  return best_model, best_params, best_metrics
end

return M

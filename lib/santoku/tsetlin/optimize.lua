local tm = require("santoku.tsetlin.capi")
local spectral = require("santoku.tsetlin.spectral")
local hlth = require("santoku.tsetlin.hlth")
local evaluator = require("santoku.tsetlin.evaluator")
local cvec = require("santoku.cvec")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")
local utc = require("santoku.utc")

local M = {}

local function key_int (v)
  return v and tostring(num.floor(v + 0.5)) or "nil"
end

local function key_int8 (v)
  return v and tostring(num.floor((v + 4) / 8) * 8) or "nil"
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
          local lo = is_log and num.log(smin) or minv
          local hi = is_log and num.log(smax) or maxv
          local half_j = jitter * 0.5
          if c - half_j < lo then c = lo + half_j end
          if c + half_j > hi then c = hi - half_j end
          x = rand.fast_normal(c, jitter * jitter)
          if is_log then x = num.exp(x) - shift end
        else
          local r = rand.fast_random() / (rand.fast_max + 1)
          if is_log then
            x = num.exp(r * span + num.log(smin)) - shift
          else
            x = r * span + minv
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
        local idx = num.floor(rand.fast_random() / (rand.fast_max + 1) * #spec) + 1
        return spec[idx]
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

M.sample_params = function (samplers, param_names, base_cfg, use_exact_defaults, use_center)
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
        p[name] = s.sample(use_center and s.center or nil)
      else
        p[name] = s.sample()
      end
    end
  end
  return p
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
  local rounds = args.rounds or 6
  local trials = args.trials or 20
  local make_key = args.make_key
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local rerun_final = args.rerun_final ~= false
  local preference_tolerance = args.preference_tolerance or 1e-6
  local size_fn = args.size_fn or function() return 0 end
  local constrain_fn = args.constrain
  local best_score = -num.huge
  local best_size = nil
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if M.all_fixed(samplers) or rounds <= 0 or trials <= 0 then
    best_params = M.sample_params(samplers, param_names, nil, true)
    if constrain_fn then
      constrain_fn(best_params)
    end
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
      local params = M.sample_params(samplers, param_names, nil, false, r > 1)
      if constrain_fn then
        constrain_fn(params)
      end
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
          rounds = rounds,
          trial = t,
          trials = trials,
          is_final = false,
          global_best_score = best_score,
        })
        round_samples = round_samples + 1
        if each_cb then
          each_cb({
            event = "trial",
            round = r,
            trial = t,
            trials = trials,
            params = params,
            score = score,
            metrics = metrics,
            global_best_score = best_score,
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
        rounds = rounds,
        trials = trials,
        round_samples = round_samples,
        round_improvements = round_improvements,
        round_best_score = round_best_score,
        round_best_params = round_best_params,
        global_best_score = best_score,
        global_best_params = best_params,
        global_best_metrics = best_metrics,
        success_rate = success_rate,
        skip_rate = skip_rate,
        adapt_factor = adapt_factor,
      })
    end
    collectgarbage("collect")
  end

  if not skip_final and best_params and rerun_final then
    if best_result and cleanup_fn then
      cleanup_fn(best_result)
    end
    local _, final_metrics, final_result = trial_fn(best_params, { is_final = true })
    best_result = final_result
    best_metrics = final_metrics
  end

  return best_result, best_params, best_metrics

end

local function create_tm (args)
  return tm.create({
    features = args.features,
    outputs = args.outputs,
    clauses = 8,
    clause_tolerance = 8,
    clause_maximum = 8,
    target = 4,
    specificity = 1000,
    reusable = true,
  })
end

local function create_final_tm (args, params)
  return tm.create({
    features = args.features,
    outputs = args.outputs,
    clauses = params.clauses,
    clause_tolerance = params.clause_tolerance,
    clause_maximum = params.clause_maximum,
    target = params.target,
    specificity = params.specificity,
    reusable = true,
  })
end

local function train_tm_simple (tmobj, args, params, iterations)
  tmobj:train({
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
    codes = args.codes,
    targets = args.targets,
    iterations = iterations,
    grouped = args.grouped,
  })
end

local function train_tm_batched (tmobj, args, params, iterations, batch_size, patience, tolerance, metric_fn, each_cb, info)
  local best_score = -num.huge
  local last_score = -num.huge
  local last_metrics = nil
  local batches_since_improve = 0
  local checkpoint = (patience and patience > 0) and cvec.create(0) or nil
  local has_checkpoint = false
  local tol = tolerance or 1e-6
  local total_epochs = 0

  while total_epochs < iterations do
    local batch_iters = math.min(batch_size, iterations - total_epochs)
    tmobj:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      codes = args.codes,
      targets = args.targets,
      iterations = batch_iters,
      grouped = args.grouped,
    })
    total_epochs = total_epochs + batch_iters

    local score, metrics = metric_fn(tmobj, args)
    last_score = score
    last_metrics = metrics

    if each_cb then
      local cb_result = each_cb(tmobj, info.is_final, metrics, params, total_epochs,
        not info.is_final and info.round or nil,
        not info.is_final and info.trial or nil,
        not info.is_final and info.rounds or nil,
        not info.is_final and info.trials or nil,
        not info.is_final and info.global_best_score or nil,
        best_score)
      if cb_result == false then
        break
      end
    end

    if score > best_score + tol then
      best_score = score
      batches_since_improve = 0
      if checkpoint then
        tmobj:checkpoint(checkpoint)
        has_checkpoint = true
      end
    else
      batches_since_improve = batches_since_improve + 1
    end

    if patience and patience > 0 and batches_since_improve >= patience then
      break
    end
  end

  if checkpoint and has_checkpoint then
    tmobj:restore(checkpoint)
    last_score, last_metrics = metric_fn(tmobj, args)
  end

  return last_score, last_metrics
end

local function optimize_tm (args)

  local iters_search = args.search_iterations or 40
  local final_iters = args.final_iterations or 400
  local final_batch = args.final_batch or 10
  local final_patience = args.final_patience or 4
  local global_dev = args.search_dev or 0.2
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local param_names = { "clauses", "clause_tolerance", "clause_maximum", "target", "specificity" }
  local samplers = M.build_samplers(args, param_names, global_dev)

  local search_tm
  if not M.all_fixed(samplers) then
    search_tm = create_tm({
      features = args.features,
      outputs = args.outputs,
    })
  end

  local function constrain_tm_params (params)
    if params.target and params.clause_tolerance then
      local max_target = 8 * params.clause_tolerance
      if params.target > max_target then
        params.target = max_target
      end
      if params.target < 1 then
        params.target = 1
      end
    end
    if params.specificity and args.features then
      local max_specificity = 2 * args.features
      if params.specificity > max_specificity then
        params.specificity = max_specificity
      end
      if params.specificity < 1 then
        params.specificity = 1
      end
    end
  end

  local function search_trial_fn (params, info)
    constrain_tm_params(params)
    search_tm:reconfigure(params)
    local train_args = {
      codes = args.codes,
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      targets = args.targets,
      grouped = args.grouped,
    }
    train_tm_simple(search_tm, train_args, params, iters_search)
    local score, metrics = metric_fn(search_tm, train_args)
    if each_cb then
      each_cb(search_tm, false, metrics, params, iters_search,
        info.round, info.trial, info.rounds, info.trials,
        info.global_best_score, nil)
    end
    return score, metrics, nil
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    rounds = args.search_rounds or 6,
    trials = args.search_trials or 20,
    trial_fn = search_trial_fn,
    skip_final = true,
    preference_tolerance = args.preference_tolerance or 1e-6,
    size_fn = function(p) return { p.clauses or 0 } end,
    make_key = function (p)
      return str.format("%s|%s|%s|%s|%s",
        key_int8(p.clauses),
        key_int(p.clause_tolerance),
        key_int(p.clause_maximum),
        key_int(p.target),
        key_int(p.specificity))
    end,
  })

  if search_tm then
    search_tm:destroy()
  end

  constrain_tm_params(best_params)
  local final_tm = create_final_tm({
    features = args.features,
    outputs = args.outputs,
  }, best_params)
  local final_train_args = {
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
    targets = args.targets,
    grouped = args.grouped,
  }
  local _, final_metrics = train_tm_batched(final_tm, final_train_args, best_params, final_iters,
    final_batch, final_patience, args.early_tolerance, metric_fn, each_cb, { is_final = true })

  collectgarbage("collect")
  return final_tm, final_metrics, best_params
end

M.regressor = function (args)
  return optimize_tm(args)
end

M.destroy_spectral = function (model)
  if not model then return end
  if model.raw_codes then model.raw_codes:destroy() end
  if model.ids then model.ids:destroy() end
  if model.landmark_ids then model.landmark_ids:destroy() end
  if model.eigenvectors then model.eigenvectors:destroy() end
  if model.eigenvalues then model.eigenvalues:destroy() end
  if model.landmark_chol then model.landmark_chol:destroy() end
  if model.scales then model.scales:destroy() end
  if model.col_means then model.col_means:destroy() end
end

M.score_spectral_eval = function (args)
  local model = args.model
  local eval_params = args.eval_params
  local eval = args.eval
  local kernel_index = args.kernel_index
  local kernel_params = args.kernel_params

  local kernel_stats = nil
  if kernel_index and kernel_params then
    kernel_stats = evaluator.ranking_accuracy({
      kernel_index = kernel_index,
      kernel_decay = kernel_params.decay,
      kernel_bandwidth = kernel_params.bandwidth,
      eval_ids = eval.ids,
      eval_offsets = eval.offsets,
      eval_neighbors = eval.neighbors,
      eval_weights = eval.weights,
      ranking = eval_params.ranking,
    })
  end

  local raw_stats = evaluator.ranking_accuracy({
    raw_codes = model.raw_codes,
    ids = model.ids,
    eval_ids = eval.ids,
    eval_offsets = eval.offsets,
    eval_neighbors = eval.neighbors,
    eval_weights = eval.weights,
    ranking = eval_params.ranking,
    n_dims = model.spectral_dims or model.dims,
  })

  return raw_stats.score, {
    score = raw_stats.score,
    raw_score = raw_stats.score,
    kernel_score = kernel_stats and kernel_stats.score or nil,
    n_dims = model.dims,
    total_queries = raw_stats.total_queries,
  }
end

M.build_spectral_nystrom = function (args)
  local index = args.index
  local n_landmarks = args.n_landmarks or 0
  local n_dims = args.n_dims
  local decay = args.decay
  local bandwidth = args.bandwidth
  local trace_tol = args.trace_tol
  local each_cb = args.each
  local train_tokens = args.train_tokens
  local train_ids = args.train_ids

  local landmark_ids, landmark_chol, scales, actual_landmarks, trace_ratio =
    spectral.sample_landmarks({
      inv = index,
      n_landmarks = n_landmarks,
      decay = decay,
      bandwidth = bandwidth,
      trace_tol = trace_tol,
    })

  local effective_dims = n_dims or n_landmarks
  if effective_dims > actual_landmarks then
    effective_dims = actual_landmarks
  end

  local eigenvectors, eigenvalues, col_means = spectral.encode({
    chol = landmark_chol,
    n_samples = actual_landmarks,
    n_landmarks = actual_landmarks,
    n_dims = effective_dims,
    each = args.spectral_each,
  })

  if each_cb then
    each_cb({
      event = "spectral_result",
      n_dims = effective_dims,
      n_landmarks = actual_landmarks,
      trace_ratio = trace_ratio,
      eig_min = eigenvalues:min(),
      eig_max = eigenvalues:max(),
    })
  end

  local nystrom_encode, _ = hlth.nystrom_encoder({
    features_index = index,
    eigenvectors = eigenvectors,
    eigenvalues = eigenvalues,
    landmark_ids = landmark_ids,
    col_means = col_means,
    landmark_chol = landmark_chol,
    scales = scales,
    n_dims = effective_dims,
    decay = decay,
    bandwidth = bandwidth,
  })

  local raw_codes = nil
  local ids = nil
  if train_tokens then
    raw_codes = nystrom_encode(train_tokens)
    ids = train_ids
  end

  return {
    ids = ids,
    raw_codes = raw_codes,
    dims = effective_dims,
    spectral_dims = effective_dims,
    landmark_ids = landmark_ids,
    eigenvectors = eigenvectors,
    eigenvalues = eigenvalues,
    col_means = col_means,
    landmark_chol = landmark_chol,
    scales = scales,
    n_landmarks = actual_landmarks,
    decay = decay,
    bandwidth = bandwidth,
    nystrom_encode = nystrom_encode,
  }
end

M.spectral = function (args)
  local index = err.assert(args.index, "index required")
  local n_landmarks_cfg = args.n_landmarks
  local n_dims_cfg = args.n_dims
  if not n_landmarks_cfg and not n_dims_cfg then
    err.error("n_landmarks or n_dims required")
  end
  n_landmarks_cfg = n_landmarks_cfg or n_dims_cfg
  n_dims_cfg = n_dims_cfg or n_landmarks_cfg
  local decay_cfg = args.decay or 0.0
  local bandwidth_cfg = args.bandwidth or -1.0
  local trace_tol = args.trace_tol
  local each_cb = args.each
  local tolerance = args.tolerance or 1e-6
  local train_tokens = args.train_tokens
  local train_ids = args.train_ids

  local expected = args.expected and {
    ids = args.expected.ids,
    offsets = args.expected.offsets,
    neighbors = args.expected.neighbors,
    weights = args.expected.weights,
  } or nil

  local eval_cfg = args.eval
  local rounds = args.rounds or 6
  local samples_per_round = args.samples or 20
  local global_dev = args.search_dev or 0.2

  local function is_range(cfg)
    return type(cfg) == "table" and cfg.min ~= nil
  end

  local function get_fixed_val(cfg)
    if type(cfg) == "number" then return cfg end
    if type(cfg) == "table" then return cfg.def or cfg.min end
    return nil
  end

  local landmarks_is_range = is_range(n_landmarks_cfg)
  local dims_is_range = is_range(n_dims_cfg)
  local decay_is_range = is_range(decay_cfg)
  local bandwidth_is_range = is_range(bandwidth_cfg)
  local has_search = (landmarks_is_range or dims_is_range or decay_is_range or bandwidth_is_range) and expected and eval_cfg and rounds > 0

  if not has_search then
    local n_landmarks_val = get_fixed_val(n_landmarks_cfg)
    local n_dims_val = get_fixed_val(n_dims_cfg)
    n_landmarks_val = n_landmarks_val or n_dims_val
    n_dims_val = n_dims_val or n_landmarks_val
    if n_dims_val > n_landmarks_val then
      n_dims_val = n_landmarks_val
    end
    local decay_val = get_fixed_val(decay_cfg) or 0.0
    local bandwidth_val = get_fixed_val(bandwidth_cfg) or -1.0
    local params = { n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val, bandwidth = bandwidth_val }
    if each_cb then
      each_cb({ event = "sample", n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val, bandwidth = bandwidth_val })
    end
    local model = M.build_spectral_nystrom({
      index = index,
      n_landmarks = n_landmarks_val,
      n_dims = n_dims_val,
      decay = decay_val,
      bandwidth = bandwidth_val,
      trace_tol = trace_tol,
      each = each_cb,
      train_tokens = train_tokens,
      train_ids = train_ids,
    })
    local score = nil
    local metrics = nil
    if expected and eval_cfg then
      local eval_t0 = utc.time(true)
      score, metrics = M.score_spectral_eval({
        model = model,
        eval_params = eval_cfg,
        eval = expected,
        kernel_index = index,
        kernel_params = { decay = decay_val, bandwidth = bandwidth_val },
      })
      local eval_t1 = utc.time(true)
      if each_cb then
        each_cb({ event = "eval", n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val, bandwidth = bandwidth_val, score = score, metrics = metrics, elapsed = eval_t1 - eval_t0 })
      end
    end
    if each_cb then
      each_cb({ event = "done", best_params = params, best_score = score, best_metrics = metrics })
    end
    return model, params
  end

  local param_names = { "n_landmarks", "n_dims", "decay", "bandwidth" }
  local samplers = {
    n_landmarks = M.build_sampler(n_landmarks_cfg, global_dev),
    n_dims = M.build_sampler(n_dims_cfg, global_dev),
    decay = M.build_sampler(decay_cfg, global_dev),
    bandwidth = M.build_sampler(bandwidth_cfg, global_dev),
  }

  local function trial_fn (params, info)
    if each_cb then
      each_cb({
        event = "sample",
        round = info.round,
        rounds = info.rounds,
        trial = info.trial,
        trials = info.trials,
        n_landmarks = params.n_landmarks,
        n_dims = params.n_dims,
        decay = params.decay,
        bandwidth = params.bandwidth,
        global_best_score = info.global_best_score,
      })
    end
    local model = M.build_spectral_nystrom({
      index = index,
      n_landmarks = params.n_landmarks,
      n_dims = params.n_dims,
      decay = params.decay,
      bandwidth = params.bandwidth,
      trace_tol = trace_tol,
      each = each_cb,
      train_tokens = train_tokens,
      train_ids = train_ids,
    })
    local score, metrics = M.score_spectral_eval({
      model = model,
      eval_params = eval_cfg,
      eval = expected,
      kernel_index = index,
      kernel_params = { decay = params.decay, bandwidth = params.bandwidth },
    })
    return score, metrics, model
  end

  local best_model, best_params, best_metrics = M.search({
    param_names = param_names,
    samplers = samplers,
    trial_fn = trial_fn,
    rounds = rounds,
    trials = samples_per_round,
    preference_tolerance = tolerance,
    rerun_final = false,
    cleanup = M.destroy_spectral,
    size_fn = function (p)
      return { p.n_dims, p.n_landmarks, num.abs(p.decay), num.abs(p.bandwidth) }
    end,
    make_key = function (p)
      return str.format("%d_%d_%.2f_%.2f", p.n_landmarks, p.n_dims, p.decay, p.bandwidth)
    end,
    constrain = function (p)
      if p.n_dims > p.n_landmarks then
        p.n_dims = p.n_landmarks
      end
    end,
    each = each_cb and function (ev)
      if ev.event == "trial" then
        each_cb({
          event = "eval",
          round = ev.round,
          trial = ev.trial,
          trials = ev.trials,
          n_landmarks = ev.params.n_landmarks,
          n_dims = ev.params.n_dims,
          decay = ev.params.decay,
          bandwidth = ev.params.bandwidth,
          score = ev.score,
          metrics = ev.metrics,
          global_best_score = ev.global_best_score,
        })
      elseif ev.event == "round" then
        each_cb({
          event = "round_end",
          round = ev.round,
          rounds = ev.rounds,
          trials = ev.trials,
          round_samples = ev.round_samples,
          round_improvements = ev.round_improvements,
          round_best_score = ev.round_best_score,
          global_best_score = ev.global_best_score,
          best_params = ev.global_best_params,
          best_metrics = ev.global_best_metrics,
          success_rate = ev.success_rate,
          skip_rate = ev.skip_rate,
          adapt_factor = ev.adapt_factor,
        })
      end
    end or nil,
  })

  if each_cb then
    each_cb({ event = "done", best_params = best_params, best_score = best_metrics and best_metrics.score, best_metrics = best_metrics })
  end

  return best_model, best_params
end

M.rp = function (args)
  local rvec = require("santoku.rvec")

  local raw_codes = err.assert(args.raw_codes, "raw_codes required")
  local ids = err.assert(args.ids, "ids required")
  local n_samples = err.assert(args.n_samples, "n_samples required")
  local n_dims = err.assert(args.n_dims, "n_dims required")
  local eval_data = err.assert(args.eval, "eval required")
  local max_bits = args.max_bits or 512
  local ranking = args.ranking or "ndcg"
  local seed = args.seed or 12345
  local tolerance = args.tolerance or 0.01
  local each_cb = args.each

  local n_bits_levels = 0
  local b = 8
  while b <= max_bits do
    n_bits_levels = n_bits_levels + 1
    b = b * 2
  end
  local n_results = n_dims * n_bits_levels

  local scores_out = dvec.create(n_results)
  local dims_out = ivec.create(n_results)
  local bits_out = ivec.create(n_results)

  local selected_cols = ivec.create(n_dims)
  local truncated = dvec.create()
  local rp_codes = cvec.create()
  local rp_weights = dvec.create(max_bits * n_dims)

  local result_idx = 0
  for prefix_dims = 1, n_dims do
    selected_cols:setn(prefix_dims)
    selected_cols:fill_indices()
    raw_codes:mtx_select(selected_cols, nil, n_dims, truncated)

    local bits = 8
    while bits <= max_bits do
      local rp_encode, rp_n_bits = hlth.rp_encoder({
        n_dims = prefix_dims,
        rp_dims = bits,
        seed = seed,
        weights = rp_weights,
      })

      rp_codes:setn(0)
      rp_encode(truncated, rp_codes)

      local stats = evaluator.ranking_accuracy({
        codes = rp_codes,
        ids = ids,
        n_dims = rp_n_bits,
        eval_ids = eval_data.ids,
        eval_offsets = eval_data.offsets,
        eval_neighbors = eval_data.neighbors,
        eval_weights = eval_data.weights,
        ranking = ranking,
      })

      scores_out:set(result_idx, stats.score)
      dims_out:set(result_idx, prefix_dims)
      bits_out:set(result_idx, bits)
      result_idx = result_idx + 1

      if each_cb then
        each_cb({ dims = prefix_dims, bits = bits, score = stats.score })
      end

      bits = bits * 2
    end
  end

  selected_cols:destroy()
  truncated:destroy()
  rp_codes:destroy()
  rp_weights:destroy()

  local order = {}
  for i = 0, n_results - 1 do
    order[i + 1] = i
  end
  table.sort(order, function (a, b)
    local sa, sb = scores_out:get(a), scores_out:get(b)
    local ba, bb = bits_out:get(a), bits_out:get(b)
    local da, db = dims_out:get(a), dims_out:get(b)
    local band_a = math.floor(sa / tolerance)
    local band_b = math.floor(sb / tolerance)
    if band_a ~= band_b then return band_a > band_b end
    if da ~= db then return da < db end
    if ba ~= bb then return ba < bb end
    return false
  end)

  local ranks = rvec.create()
  for rank, idx in ipairs(order) do
    ranks:push(idx, rank - 1)
  end

  return ranks, scores_out, dims_out, bits_out
end

return M

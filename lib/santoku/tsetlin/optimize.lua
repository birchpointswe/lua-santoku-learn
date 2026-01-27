local tm = require("santoku.tsetlin.capi")
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

M.apply_itq = function (raw_codes, n_dims, return_params, binarize)
  binarize = binarize or "itq"
  if binarize == "sign" then
    local codes = itq.sign({ codes = raw_codes, n_dims = n_dims })
    return codes, nil, nil
  elseif binarize == "median" then
    local codes, medians = itq.median({ codes = raw_codes, n_dims = n_dims, return_params = return_params })
    return codes, medians, nil
  else
    return itq.itq({
      codes = raw_codes,
      n_dims = n_dims,
      iterations = 100,
      tolerance = 1e-8,
      return_params = return_params,
    })
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
      local params = M.sample_params(samplers, param_names, nil, false, r > 1)
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
      feat_offsets = args.feat_offsets,
      clauses = 8,
      clause_tolerance = 8,
      clause_maximum = 8,
      target = 4,
      specificity = 1000,
      include_bits = 1,
      reusable = true,
    })
  elseif typ == "regressor" then
    return tm.create("regressor", {
      features = args.features,
      outputs = args.outputs,
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
      feat_offsets = args.feat_offsets,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
      include_bits = params.include_bits,
    })
  elseif typ == "regressor" then
    return tm.create("regressor", {
      features = args.features,
      outputs = args.outputs,
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
  elseif typ == "regressor" then
    tmobj:train({
      samples = args.samples,
      problems = args.problems,
      targets = args.targets,
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
      outputs = args.outputs,
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
      targets = args.targets,
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
    outputs = args.outputs,
    individualized = args.individualized,
    feat_offsets = args.feat_offsets,
  }, best_params)
  local final_train_args = {
    sentences = args.sentences,
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
    targets = args.targets,
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

M.regressor = function (args)
  return optimize_tm(args, "regressor")
end

M.destroy_spectral = function (model)
  if not model then return end
  if model.index then model.index:destroy() end
  if model.codes then model.codes:destroy() end
  if model.raw_codes then model.raw_codes:destroy() end
  if model.ids then model.ids:destroy() end
  if model.landmark_ids then model.landmark_ids:destroy() end
  if model.eigenvectors then model.eigenvectors:destroy() end
  if model.eigenvalues then model.eigenvalues:destroy() end
  if model.chol then model.chol:destroy() end
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
      kernel_cmp = kernel_params.cmp,
      kernel_alpha = kernel_params.cmp_alpha,
      kernel_beta = kernel_params.cmp_beta,
      kernel_decay = kernel_params.decay,
      kernel_combine = kernel_params.combine,
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
    n_dims = model.dims,
  })

  local stats = evaluator.ranking_accuracy({
    codes = model.codes,
    index = model.index,
    ids = model.ids,
    eval_ids = eval.ids,
    eval_offsets = eval.offsets,
    eval_neighbors = eval.neighbors,
    eval_weights = eval.weights,
    ranking = eval_params.ranking,
    n_dims = model.dims,
  })

  return stats.score, {
    score = stats.score,
    raw_score = raw_stats.score,
    kernel_score = kernel_stats and kernel_stats.score or nil,
    n_dims = model.dims,
    total_queries = stats.total_queries,
  }
end

M.build_spectral_nystrom = function (args)
  local index = args.index
  local n_landmarks = args.n_landmarks
  local n_dims = args.n_dims
  local cmp = args.cmp
  local cmp_alpha = args.cmp_alpha
  local cmp_beta = args.cmp_beta
  local decay = args.decay
  local combine = args.combine
  local bucket_size = args.bucket_size
  local each_cb = args.each
  local binarize = args.binarize or "itq"

  local t0 = utc.time(true)

  if each_cb then
    each_cb({ event = "stage", stage = "landmarks", decay = decay })
  end

  local landmark_ids, doc_ids, chol, actual_landmarks =
    index:sample_landmarks(n_landmarks, cmp, cmp_alpha, cmp_beta, decay, combine)

  local n_samples = doc_ids:size()
  local t1 = utc.time(true)

  if each_cb then
    each_cb({
      event = "landmarks_result",
      n_samples = n_samples,
      n_landmarks = actual_landmarks,
      requested_landmarks = n_landmarks,
      elapsed = t1 - t0,
    })
  end

  if each_cb then
    each_cb({ event = "stage", stage = "spectral" })
  end

  local eigenvectors, eigenvalues = spectral.encode({
    chol = chol,
    n_samples = n_samples,
    n_landmarks = actual_landmarks,
    n_dims = n_dims,
    each = args.spectral_each,
  })

  local t2 = utc.time(true)

  if each_cb then
    each_cb({
      event = "spectral_result",
      n_dims = n_dims,
      eig_min = eigenvalues:min(),
      eig_max = eigenvalues:max(),
      elapsed = t2 - t1,
    })
  end

  if each_cb then
    each_cb({ event = "stage", stage = "lift" })
  end

  local col_means, raw_codes = hlth.nystrom_lift({
    chol = chol,
    eigenvectors = eigenvectors,
    eigenvalues = eigenvalues,
    n_samples = n_samples,
    n_landmarks = actual_landmarks,
    n_dims = n_dims,
  })

  local t3 = utc.time(true)

  if each_cb then
    each_cb({ event = "stage_done", stage = "lift", elapsed = t3 - t2 })
  end

  if each_cb then
    each_cb({ event = "stage", stage = "binarize", method = binarize })
  end

  local codes, itq_means_or_medians, itq_rotation = M.apply_itq(raw_codes, n_dims, true, binarize)

  local t4 = utc.time(true)

  if each_cb then
    each_cb({ event = "stage_done", stage = "binarize", method = binarize, elapsed = t4 - t3 })
  end

  if each_cb then
    each_cb({ event = "stage", stage = "index" })
  end

  local ann_index = ann.create({
    expected_size = n_samples,
    bucket_size = bucket_size,
    features = n_dims,
  })
  ann_index:add(codes, doc_ids)

  local t5 = utc.time(true)

  if each_cb then
    each_cb({ event = "stage_done", stage = "index", elapsed = t5 - t4, total = t5 - t0 })
  end

  return {
    ids = doc_ids,
    codes = codes,
    raw_codes = raw_codes,
    dims = n_dims,
    index = ann_index,
    landmark_ids = landmark_ids,
    eigenvectors = eigenvectors,
    eigenvalues = eigenvalues,
    col_means = col_means,
    binarize = binarize,
    itq_means = binarize == "itq" and itq_means_or_medians or nil,
    itq_rotation = itq_rotation,
    median_thresholds = binarize == "median" and itq_means_or_medians or nil,
    chol = chol,
    n_landmarks = actual_landmarks,
  }
end

M.spectral = function (args)
  local index = err.assert(args.index, "index required")
  local n_landmarks_cfg = err.assert(args.n_landmarks, "n_landmarks required")
  local n_dims_cfg = err.assert(args.n_dims, "n_dims required")
  local cmp = args.cmp or "jaccard"
  local cmp_alpha = args.cmp_alpha or 0.5
  local cmp_beta = args.cmp_beta or 0.5
  local decay_cfg = args.decay or 0.0
  local combine = args.combine or "weighted_avg"
  local bucket_size = args.bucket_size
  local each_cb = args.each
  local tolerance = args.tolerance or 1e-6
  local binarize = args.binarize or "itq"

  local expected = args.expected and {
    ids = args.expected.ids,
    offsets = args.expected.offsets,
    neighbors = args.expected.neighbors,
    weights = args.expected.weights,
  } or nil

  local eval_cfg = args.eval
  local rounds = args.rounds or 1
  local samples_per_round = args.samples or 4
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
  local has_search = (landmarks_is_range or dims_is_range or decay_is_range) and expected and eval_cfg and rounds > 0

  if not has_search then
    local n_landmarks_val = get_fixed_val(n_landmarks_cfg)
    local n_dims_val = get_fixed_val(n_dims_cfg)
    local decay_val = get_fixed_val(decay_cfg) or 0.0
    local params = { n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val }
    if each_cb then
      each_cb({ event = "sample", n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val })
    end
    local model = M.build_spectral_nystrom({
      index = index,
      n_landmarks = n_landmarks_val,
      n_dims = n_dims_val,
      cmp = cmp,
      cmp_alpha = cmp_alpha,
      cmp_beta = cmp_beta,
      decay = decay_val,
      combine = combine,
      bucket_size = bucket_size,
      binarize = binarize,
      each = each_cb,
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
        kernel_params = { cmp = cmp, cmp_alpha = cmp_alpha, cmp_beta = cmp_beta, decay = decay_val, combine = combine },
      })
      local eval_t1 = utc.time(true)
      if each_cb then
        each_cb({ event = "eval", n_landmarks = n_landmarks_val, n_dims = n_dims_val, decay = decay_val, score = score, metrics = metrics, elapsed = eval_t1 - eval_t0 })
      end
    end
    if each_cb then
      each_cb({ event = "done", best_params = params, best_score = score, best_metrics = metrics })
    end
    return model, params
  end

  local landmarks_sampler = M.build_sampler(n_landmarks_cfg, global_dev)
  local dims_sampler = M.build_sampler(n_dims_cfg, global_dev)
  local decay_sampler = M.build_sampler(decay_cfg, global_dev)

  local best_score = -num.huge
  local best_cost = nil
  local best_model = nil
  local best_params = nil
  local best_metrics = nil
  local seen = {}

  for round = 1, rounds do
    if each_cb then
      each_cb({ event = "round_start", round = round, rounds = rounds })
    end

    local round_best_score = -num.huge
    local round_improvements = 0
    local round_samples = 0

    for _ = 1, samples_per_round do
      local use_center = round > 1
      local n_landmarks_val = landmarks_sampler.sample(use_center and landmarks_sampler.center or nil)
      local n_dims_val = dims_sampler.sample(use_center and dims_sampler.center or nil)
      local decay_val = decay_sampler.sample(use_center and decay_sampler.center or nil)

      if n_dims_val > n_landmarks_val then
        n_dims_val = n_landmarks_val
      end

      local cfg_key = str.format("%d_%d_%.2f", n_landmarks_val, n_dims_val, decay_val)

      if not seen[cfg_key] then
        seen[cfg_key] = true
        round_samples = round_samples + 1

        if each_cb then
          each_cb({
            event = "sample",
            round = round,
            n_landmarks = n_landmarks_val,
            n_dims = n_dims_val,
            decay = decay_val,
          })
        end

        local model = M.build_spectral_nystrom({
          index = index,
          n_landmarks = n_landmarks_val,
          n_dims = n_dims_val,
          cmp = cmp,
          cmp_alpha = cmp_alpha,
          cmp_beta = cmp_beta,
          decay = decay_val,
          combine = combine,
          bucket_size = bucket_size,
          binarize = binarize,
          each = each_cb,
        })

        local eval_t0 = utc.time(true)
        local score, metrics = M.score_spectral_eval({
          model = model,
          eval_params = eval_cfg,
          eval = expected,
          kernel_index = index,
          kernel_params = { cmp = cmp, cmp_alpha = cmp_alpha, cmp_beta = cmp_beta, decay = decay_val, combine = combine },
        })
        local eval_t1 = utc.time(true)

        if each_cb then
          each_cb({
            event = "eval",
            round = round,
            n_landmarks = n_landmarks_val,
            n_dims = n_dims_val,
            decay = decay_val,
            score = score,
            metrics = metrics,
            elapsed = eval_t1 - eval_t0,
          })
        end

        if score > round_best_score then
          round_best_score = score
        end

        local cost = { n_dims_val, n_landmarks_val }
        if is_preferred(score, cost, best_score, best_cost, tolerance) then
          round_improvements = round_improvements + 1
          if best_model then
            M.destroy_spectral(best_model)
          end
          best_score = score
          best_cost = cost
          best_model = model
          best_params = {
            n_landmarks = n_landmarks_val,
            n_dims = n_dims_val,
            decay = decay_val,
          }
          best_metrics = metrics
        else
          M.destroy_spectral(model)
        end
      end
    end

    if best_params then
      landmarks_sampler.center = best_params.n_landmarks
      dims_sampler.center = best_params.n_dims
      decay_sampler.center = best_params.decay
    end

    local skip_rate = (samples_per_round - round_samples) / samples_per_round
    local success_rate = round_samples > 0 and (round_improvements / round_samples) or 0
    local adapt_factor
    if skip_rate > 0.8 then
      adapt_factor = 1.3
    elseif round_samples < 3 then
      adapt_factor = 1.0
    else
      adapt_factor = M.success_factor(success_rate)
    end
    if landmarks_sampler.adapt then landmarks_sampler.adapt(adapt_factor) end
    if dims_sampler.adapt then dims_sampler.adapt(adapt_factor) end
    if decay_sampler.adapt then decay_sampler.adapt(adapt_factor) end

    if each_cb then
      each_cb({
        event = "round_end",
        round = round,
        rounds = rounds,
        round_best_score = round_best_score,
        global_best_score = best_score,
        best_params = best_params,
        best_metrics = best_metrics,
        success_rate = success_rate,
        skip_rate = skip_rate,
        adapt_factor = adapt_factor,
      })
    end

    collectgarbage("collect")
  end

  if each_cb then
    each_cb({ event = "done", best_params = best_params, best_score = best_score, best_metrics = best_metrics })
  end

  return best_model, best_params
end

return M

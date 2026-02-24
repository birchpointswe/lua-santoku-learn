local tm = require("santoku.learn.regressor")
local spectral = require("santoku.learn.spectral")
local evaluator = require("santoku.learn.evaluator")
local csr = require("santoku.learn.csr")
local gp = require("santoku.learn.gp")

local cvec = require("santoku.cvec")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")

local M = {}

local function lhs_sample (n, d)
  local grid = {}
  for j = 1, d do
    local perm = {}
    for i = 1, n do perm[i] = i - 1 end
    for i = n, 2, -1 do
      local k = num.floor(rand.fast_random() / (rand.fast_max + 1) * i) + 1
      perm[i], perm[k] = perm[k], perm[i]
    end
    grid[j] = perm
  end
  local pts = {}
  for i = 1, n do
    local row = {}
    for j = 1, d do
      row[j] = (grid[j][i] + rand.fast_random() / (rand.fast_max + 1)) / n
    end
    pts[i] = row
  end
  return pts
end

local function round_to_pow2 (x)
  local log2x = num.log(x) / num.log(2)
  return num.pow(2, num.floor(log2x + 0.5))
end

local function spec_defaults (spec, defs)
  if type(spec) ~= "table" then return spec end
  local s = {}
  for k, v in pairs(defs) do s[k] = v end
  for k, v in pairs(spec) do s[k] = v end
  return s
end

local function cap_spec_max (spec, cap)
  if spec == nil then return nil end
  if type(spec) == "number" then
    return spec > cap and cap or spec
  end
  if type(spec) == "table" and spec.min ~= nil then
    local s = {}
    for k, v in pairs(spec) do s[k] = v end
    if s.max > cap then s.max = cap end
    if s.min > cap then s.min = cap end
    if s.def and s.def > cap then s.def = cap end
    return s
  end
  return spec
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
    local is_int = not not spec.int
    local is_pow2 = not not spec.pow2
    local is_log = is_pow2 or not not spec.log
    local round_to = spec.round
    -- For log scale with non-positive min, shift range so shifted_min = 1
    local shift = (is_log and minv <= 0) and (1 - minv) or 0
    local smin = minv + shift
    local smax = maxv + shift
    local span = is_log and (num.log(smax) - num.log(smin)) or (maxv - minv)
    local init_jitter = (spec.dev or global_dev or 1.0) * span
    local jitter = init_jitter
    local log_smin = is_log and num.log(smin) or 0
    local log_span = is_log and (num.log(smax) - num.log(smin)) or 0
    local lin_span = maxv - minv
    return {
      type = "range",
      center = spec.def,
      normalize = function (x)
        if lin_span == 0 then return 0.5 end
        if is_log then
          return (num.log(x + shift) - log_smin) / log_span
        elseif is_int then
          return (x - minv + 0.5) / (maxv - minv + 1)
        else
          return (x - minv) / lin_span
        end
      end,
      denormalize = function (u)
        local x
        if is_log then
          x = num.exp(u * log_span + log_smin) - shift
          if is_pow2 then x = round_to_pow2(x)
          elseif round_to then x = num.floor(x / round_to + 0.5) * round_to
          elseif is_int then x = num.floor(x + 0.5) end
        elseif is_int then
          x = num.floor(u * (maxv - minv + 1) + minv)
        else
          x = u * lin_span + minv
          if is_pow2 then x = round_to_pow2(x)
          elseif round_to then x = num.floor(x / round_to + 0.5) * round_to end
        end
        if x < minv then x = minv elseif x > maxv then x = maxv end
        return x
      end,
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

M.search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local trials = args.trials or 120
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local rerun_final = args.rerun_final ~= false
  local constrain_fn = args.constrain
  local cost_fn = args.cost_fn
  local cost_beta = args.cost_beta or 0.0
  local n_candidates = args.n_candidates or 500
  local n_hyper_restarts = args.n_hyper_restarts or 20
  local best_score = -num.huge
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if M.all_fixed(samplers) or trials <= 0 then
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

  local search_dims = {}
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s and s.type == "range" and s.normalize then
      search_dims[#search_dims + 1] = name
    end
  end
  local n_dims = #search_dims
  local n_initial = num.min(2 * n_dims + 1, trials)
  local X_obs = n_dims > 0 and dvec.create() or nil
  local Y_obs = n_dims > 0 and dvec.create() or nil

  local lhs_pts = n_dims > 0 and lhs_sample(n_initial, n_dims) or nil

  for t = 1, trials do
    local params

    if n_dims == 0 or t <= n_initial then
      params = {}
      if lhs_pts then
        local pt = lhs_pts[t]
        for i, name in ipairs(search_dims) do
          params[name] = samplers[name].denormalize(pt[i])
        end
      end
      for _, name in ipairs(param_names) do
        if params[name] == nil then
          local s = samplers[name]
          if s then
            if s.type == "fixed" then
              params[name] = s.center
            else
              params[name] = s.sample()
            end
          end
        end
      end
    else
      local cand_pts = lhs_sample(n_candidates, n_dims)
      local cand_flat = dvec.create()
      local cand_costs = (cost_fn and cost_beta > 0) and dvec.create() or nil
      for i = 1, n_candidates do
        local pt = cand_pts[i]
        local p = {}
        for j, name in ipairs(search_dims) do
          p[name] = samplers[name].denormalize(pt[j])
        end
        if constrain_fn then constrain_fn(p) end
        for _, name in ipairs(search_dims) do
          cand_flat:push(samplers[name].normalize(p[name]))
        end
        if cand_costs then cand_costs:push(cost_fn(p)) end
      end
      local ei = gp.suggest(X_obs, Y_obs, n_dims, cand_flat, n_candidates, n_hyper_restarts)
      if cand_costs then
        local _, cmax = cand_costs:max()
        if cmax > 0 then cand_costs:scale(1.0 / cmax) end
        ei:scalev(cand_costs:clamp(1e-12, 1.0):pow(-cost_beta))
      end
      local _, best_c = ei:max()
      params = {}
      for i, name in ipairs(search_dims) do
        params[name] = samplers[name].denormalize(cand_flat:get(best_c * n_dims + i - 1))
      end
      for _, name in ipairs(param_names) do
        if params[name] == nil then
          local s = samplers[name]
          if s then
            if s.type == "fixed" then
              params[name] = s.center
            else
              params[name] = s.sample()
            end
          end
        end
      end
    end

    if constrain_fn then constrain_fn(params) end

    local phase = (n_dims == 0 or t <= n_initial) and "lhs" or "gp"
    local score, metrics, result = trial_fn(params, {
      trial = t,
      trials = trials,
      is_final = false,
      global_best_score = best_score,
      phase = phase,
    })

    if n_dims > 0 then
      for _, name in ipairs(search_dims) do
        X_obs:push(samplers[name].normalize(params[name]))
      end
      Y_obs:push(score)
    end

    local new_best = score > best_score
    if each_cb then
      each_cb({
        event = "trial",
        trial = t,
        trials = trials,
        params = params,
        score = score,
        metrics = metrics,
        global_best_score = best_score,
        is_new_best = new_best,
        phase = phase,
      })
    end
    if new_best then
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

  collectgarbage("collect")

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

local function create_tm (args, max_features)
  local opts = {
    features = max_features,
    outputs = args.outputs,
    state = args.state or 8,
    clauses = 1,
    clause_tolerance = 8,
    clause_maximum = 8,
    target = 4,
    specificity = 1000,
    reusable = true,
  }
  if args.n_tokens then
    opts.n_tokens = args.n_tokens
  end
  if args.class_batch then
    opts.class_batch = args.class_batch
  end
  return tm.create(opts)
end

local function create_final_tm (args, params, max_features)
  local opts = {
    features = params.features or max_features,
    outputs = args.outputs,
    state = args.state or 8,
    clauses = params.clauses,
    clause_tolerance = params.clause_tolerance,
    clause_maximum = params.clause_maximum,
    target = params.target,
    specificity = params.specificity,
    reusable = true,
  }
  if args.n_tokens then
    opts.n_tokens = args.n_tokens
    opts.absorb_threshold = params.absorb_threshold
    opts.absorb_maximum = params.absorb_maximum
    opts.absorb_insert = params.absorb_insert
    opts.absorb_ranking_limit = params.absorb_ranking_limit
  end
  if args.class_batch then
    opts.class_batch = args.class_batch
  end
  if params.per_class_tolerances then
    opts.per_class_tolerances = params.per_class_tolerances
    opts.per_class_maximums = params.per_class_maximums
    opts.per_class_spec_thresholds = params.per_class_spec_thresholds
    opts.per_class_targets = params.per_class_targets
  end
  return tm.create(opts)
end

local function train_tm_simple (tmobj, args, _, iterations)
  local t = {
    samples = args.samples,
    problems = args.problems,
    sol_offsets = args.sol_offsets,
    sol_neighbors = args.sol_neighbors,
    codes = args.codes,
    targets = args.targets,
    iterations = iterations,
  }
  if args.csc_offsets then
    t.csc_offsets = args.csc_offsets
    t.csc_indices = args.csc_indices
    t.absorb_ranking = args.absorb_ranking
    t.absorb_ranking_offsets = args.absorb_ranking_offsets
    t.absorb_ranking_global = args.absorb_ranking_global
    t.problems = nil
  end
  tmobj:train(t)
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
    local t = {
      samples = args.samples,
      problems = args.problems,
      sol_offsets = args.sol_offsets,
      sol_neighbors = args.sol_neighbors,
      codes = args.codes,
      targets = args.targets,
      iterations = batch_iters,
    }
    if args.csc_offsets then
      t.csc_offsets = args.csc_offsets
      t.csc_indices = args.csc_indices
      t.absorb_ranking = args.absorb_ranking
      t.absorb_ranking_offsets = args.absorb_ranking_offsets
      t.absorb_ranking_global = args.absorb_ranking_global
      t.problems = nil
    end
    tmobj:train(t)
    total_epochs = total_epochs + batch_iters

    local score, metrics = metric_fn(tmobj, args)
    last_score = score
    last_metrics = metrics

    if each_cb then
      local cb_result = each_cb({
        tm = tmobj,
        is_final = info.is_final,
        metrics = metrics,
        params = params,
        epoch = total_epochs,
        trial = not info.is_final and info.trial or nil,
        trials = not info.is_final and info.trials or nil,
        global_best_score = not info.is_final and info.global_best_score or nil,
        best_epoch_score = best_score,
        phase = not info.is_final and info.phase or nil,
      })
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

local function compute_per_class_ivec (base, alpha, weights, n, apply_fn)
  if not weights or alpha == 0 then return nil end
  local w_min, w_max = weights:min(), weights:max()
  local result = ivec.create(n)
  for i = 0, n - 1 do
    local w_norm = (w_max > w_min) and ((weights:get(i) - w_min) / (w_max - w_min)) or 0.5
    local val = base * num.exp(alpha * (w_norm - 0.5))
    result:set(i, apply_fn(val))
  end
  return result
end

local function select_ranking_segments (ranking, offsets, dim_ids)
  if not ranking or not offsets then return ranking, offsets end
  local n_dims = dim_ids:size()
  local indices = ivec.create()
  local new_offsets = ivec.create(n_dims + 1)
  local pos = 0
  for i = 0, n_dims - 1 do
    local c = dim_ids:get(i)
    local seg_start = offsets:get(c)
    local seg_end = offsets:get(c + 1)
    new_offsets:set(i, pos)
    for j = seg_start, seg_end - 1 do
      indices:push(j)
    end
    pos = pos + (seg_end - seg_start)
  end
  new_offsets:set(n_dims, pos)
  return ivec.create():copy(ranking, indices), new_offsets
end

local function optimize_tm (args)

  local iters_search = args.search_iterations or 40
  local final_iters = args.final_iterations or 400
  local final_batch = args.final_batch or 10
  local final_patience = args.final_patience or 4
  local global_dev = args.search_dev or 0.2
  local metric_fn = args.search_metric
  if not metric_fn and args.targets then
    metric_fn = function (t, targs)
      local input = targs.tokens
        and { tokens = targs.tokens, n_samples = targs.samples }
        or targs.problems
      local score, mae = t:regress_mae(input, targs.samples, targs.targets)
      return score, { mean = mae }
    end
  end
  if not metric_fn and args.codes then
    metric_fn = function (t, targs)
      local input = targs.tokens
        and { tokens = targs.tokens, n_samples = targs.samples }
        or targs.problems
      local acc = t:encode_hamming(input, targs.samples, targs.codes)
      return acc, { accuracy = acc }
    end
  end
  if not metric_fn then
    err.error("search_metric required (or provide targets for default regressor metric)")
  end
  local each_cb = args.each
  local search_subsample_samples = args.search_subsample_samples or args.search_subsample
  local search_subsample_targets = args.search_subsample_targets
  local sparse = not not args.n_tokens

  local max_features
  if type(args.features) == "table" and args.features.max then
    max_features = args.features.max
  else
    max_features = args.features
  end

  local param_names = { "features", "clauses", "clause_maximum_fraction", "clause_tolerance_fraction", "target_fraction", "specificity_fraction" }
  if sparse then
    param_names[#param_names + 1] = "absorb_threshold"
    param_names[#param_names + 1] = "absorb_maximum_fraction"
    param_names[#param_names + 1] = "absorb_insert_offset"
    param_names[#param_names + 1] = "absorb_ranking_fraction"
  end
  if args.output_weights then
    local alpha_def = { min = -3, max = 3, def = 0 }
    if not args.alpha_specificity then args.alpha_specificity = alpha_def end
    if not args.alpha_target then args.alpha_target = alpha_def end
    if not args.alpha_tolerance then args.alpha_tolerance = alpha_def end
    if not args.alpha_maximum then args.alpha_maximum = alpha_def end
    param_names[#param_names + 1] = "alpha_specificity"
    param_names[#param_names + 1] = "alpha_target"
    param_names[#param_names + 1] = "alpha_tolerance"
    param_names[#param_names + 1] = "alpha_maximum"
  end

  local input_bits = 2 * max_features
  args.clause_maximum_fraction = spec_defaults(args.clause_maximum_fraction,
    { min = 0.001, max = 1.0, log = true })
  args.clause_tolerance_fraction = spec_defaults(args.clause_tolerance_fraction,
    { min = 0.01, max = 1.0 })
  args.target_fraction = spec_defaults(args.target_fraction,
    { min = 0.01, max = 2.0 })
  args.specificity_fraction = spec_defaults(args.specificity_fraction,
    { min = 0.0005, max = 0.5, log = true })
  if sparse then
    local m = (args.state or 8) - 1
    args.absorb_maximum_fraction = spec_defaults(args.absorb_maximum_fraction,
      { min = 0.001, max = 1.0, log = true })
    args.absorb_threshold = spec_defaults(args.absorb_threshold,
      { min = 0, max = 2 ^ (m + 1) - 1, int = true })
    args.absorb_insert_offset = spec_defaults(args.absorb_insert_offset,
      { min = 1, max = 2 ^ m - 1, int = true })
    args.absorb_ranking_fraction = spec_defaults(args.absorb_ranking_fraction,
      { min = 0.01, max = 1.0, log = true })
    args.absorb_threshold = cap_spec_max(args.absorb_threshold, 2 ^ (m + 1) - 1)
    args.absorb_insert_offset = cap_spec_max(args.absorb_insert_offset, 2 ^ m - 1)
  end

  local samplers = M.build_samplers(args, param_names, global_dev)

  local search_n = args.samples
  local search_ids
  if search_subsample_samples and search_subsample_samples < 1.0 then
    search_n = num.floor(args.samples * search_subsample_samples)
    if args.stratify_offsets and args.stratify_neighbors and args.stratify_labels then
      search_ids = csr.stratified_sample(
        args.stratify_offsets, args.stratify_neighbors,
        args.samples, args.stratify_labels, search_n):asc()
      search_n = search_ids:size()
    else
      search_ids = ivec.create(args.samples):fill_indices():shuffle():setn(search_n):asc()
    end
  end

  local search_outputs = args.outputs
  local search_dim_ids
  if search_subsample_targets and (args.targets or args.codes) then
    local n
    if search_subsample_targets >= 1 then
      n = num.min(num.floor(search_subsample_targets), args.outputs)
    else
      n = num.max(1, num.floor(args.outputs * search_subsample_targets))
    end
    if n < args.outputs then
      search_outputs = n
      search_dim_ids = ivec.create(search_outputs)
      for i = 0, search_outputs - 1 do
        search_dim_ids:set(i, num.floor(i * args.outputs / search_outputs))
      end
    end
  end

  local search_weights = args.output_weights
  if search_dim_ids and args.output_weights then
    search_weights = dvec.create():copy(args.output_weights, search_dim_ids)
  end

  local search_data
  if sparse then
    local search_tokens = args.tokens
    if search_ids then
      search_tokens = ivec.create()
      args.tokens:bits_select(nil, search_ids, args.n_tokens, search_tokens)
    end
    local search_csc_offsets, search_csc_indices
    if search_ids then
      search_csc_offsets, search_csc_indices = csr.to_csc(search_tokens, search_n, args.n_tokens)
    else
      search_csc_offsets = args.csc_offsets
      search_csc_indices = args.csc_indices
    end
    local search_ranking = args.absorb_ranking
    local search_ranking_offsets = args.absorb_ranking_offsets
    if search_dim_ids and args.absorb_ranking_offsets then
      search_ranking, search_ranking_offsets = select_ranking_segments(
        args.absorb_ranking, args.absorb_ranking_offsets, search_dim_ids)
    end
    search_data = {
      samples = search_n,
      tokens = search_tokens,
      csc_offsets = search_csc_offsets,
      csc_indices = search_csc_indices,
      absorb_ranking = search_ranking,
      absorb_ranking_offsets = search_ranking_offsets,
      absorb_ranking_global = args.absorb_ranking_global,
    }
  else
    local search_problems = args.problems
    if search_ids then
      search_problems = cvec.create()
      local search_cvec_features = args.cvec_features or (args.features * 2)
      args.problems:bits_select(nil, search_ids, search_cvec_features, search_problems)
    end
    search_data = {
      samples = search_n,
      problems = search_problems,
    }
  end

  if args.targets then
    local t = args.targets
    if search_ids then
      local tmp = dvec.create()
      t:mtx_select(nil, search_ids, args.outputs, tmp)
      t = tmp
    end
    if search_dim_ids then
      local tmp = dvec.create()
      t:mtx_select(search_dim_ids, nil, args.outputs, tmp)
      t = tmp
    end
    search_data.targets = t
  elseif args.sol_offsets then
    if search_ids then
      local new_off = ivec.create(search_n + 1)
      local new_nbr = ivec.create()
      local pos = 0
      for i = 0, search_n - 1 do
        local sid = search_ids:get(i)
        local lo = args.sol_offsets:get(sid)
        local hi = args.sol_offsets:get(sid + 1)
        new_off:set(i, pos)
        if hi > lo then new_nbr:copy(args.sol_neighbors, lo, hi, pos) end
        pos = pos + hi - lo
      end
      new_off:set(search_n, pos)
      search_data.sol_offsets = new_off
      search_data.sol_neighbors = new_nbr
    else
      search_data.sol_offsets = args.sol_offsets
      search_data.sol_neighbors = args.sol_neighbors
    end
  elseif args.codes then
    local c = args.codes
    if search_ids then
      local tmp = cvec.create()
      c:bits_select(nil, search_ids, args.outputs, tmp)
      c = tmp
    end
    if search_dim_ids then
      local tmp = cvec.create()
      c:bits_select(search_dim_ids, nil, args.outputs, tmp)
      c = tmp
    end
    search_data.codes = c
  end

  local search_tm
  if not M.all_fixed(samplers) then
    search_tm = create_tm({
      outputs = search_outputs,
      state = args.state,
      n_tokens = args.n_tokens,
      class_batch = args.class_batch,
    }, max_features)
  end

  local function constrain_tm_params (params, weights, outputs)
    local feat = params.features or max_features
    local input_bits = feat and 2 * feat or nil
    if params.clause_maximum_fraction and input_bits then
      local f = params.clause_maximum_fraction
      if f < 0 then f = 0 end
      if f > 1 then f = 1 end
      params.clause_maximum_fraction = f
      params.clause_maximum = num.max(1, num.floor(f * input_bits + 0.5))
    end
    if params.clause_tolerance_fraction and params.clause_maximum then
      local f = params.clause_tolerance_fraction
      if f < 0 then f = 0 end
      if f > 1 then f = 1 end
      params.clause_tolerance_fraction = f
      params.clause_tolerance = num.max(1, num.floor(f * params.clause_maximum + 0.5))
    end
    if params.target_fraction and params.clause_tolerance then
      local f = params.target_fraction
      if f < 0.01 then f = 0.01 end
      params.target_fraction = f
      params.target = num.max(1, num.floor(f * 8 * params.clause_tolerance + 0.5))
    end
    if params.specificity_fraction then
      local f = params.specificity_fraction
      if f < 0.0005 then f = 0.0005 end
      if f > 0.5 then f = 0.5 end
      params.specificity_fraction = f
      params.specificity = num.max(1, num.floor(1.0 / f + 0.5))
    end
    if params.absorb_maximum_fraction and feat then
      local f = params.absorb_maximum_fraction
      if f < 0 then f = 0 end
      if f > 1 then f = 1 end
      params.absorb_maximum_fraction = f
      params.absorb_maximum = num.max(1, num.floor(f * feat + 0.5))
    end
    if params.absorb_ranking_fraction and args.n_tokens then
      local f = params.absorb_ranking_fraction
      if f < 0.01 then f = 0.01 end
      if f > 1 then f = 1 end
      params.absorb_ranking_fraction = f
      params.absorb_ranking_limit = num.max(1, num.floor(f * args.n_tokens + 0.5))
    end
    if params.absorb_threshold or params.absorb_insert_offset then
      local m = (args.state or 8) - 1
      local max_excl = 2 ^ m - 1
      if params.absorb_threshold and params.absorb_threshold >= 2 * max_excl + 1 then
        params.absorb_threshold = 2 * max_excl
      end
      if params.absorb_insert_offset then
        params.absorb_insert = num.min(params.absorb_threshold + params.absorb_insert_offset, max_excl)
      end
    end
    if weights then
      local n = outputs or weights:size()
      local features = feat
      local clamp_int = function (v) return num.max(1, num.floor(v + 0.5)) end
      params.per_class_tolerances = compute_per_class_ivec(
        params.clause_tolerance, params.alpha_tolerance or 0, weights, n, clamp_int)
      params.per_class_maximums = compute_per_class_ivec(
        params.clause_maximum, params.alpha_maximum or 0, weights, n, clamp_int)
      params.per_class_spec_thresholds = compute_per_class_ivec(
        params.specificity, params.alpha_specificity or 0, weights, n, function (v)
          local s = num.max(1, v)
          return num.floor((2.0 * features) / s)
        end)
      params.per_class_targets = compute_per_class_ivec(
        params.target, params.alpha_target or 0, weights, n, clamp_int)
      if params.per_class_tolerances and params.per_class_maximums then
        for i = 0, n - 1 do
          local tol = params.per_class_tolerances:get(i)
          local mx = params.per_class_maximums:get(i)
          if tol > mx then
            params.per_class_tolerances:set(i, mx)
            tol = mx
          end
          local tgt = params.per_class_targets and params.per_class_targets:get(i) or params.target
          if tgt > 8 * tol then
            if params.per_class_targets then params.per_class_targets:set(i, 8 * tol) end
          end
        end
      end
    end
  end

  local function search_trial_fn (params, info)
    constrain_tm_params(params, search_weights, search_outputs)
    search_tm:reconfigure(params)
    train_tm_simple(search_tm, search_data, params, iters_search)
    local score, metrics = metric_fn(search_tm, search_data)
    if each_cb then
      each_cb({
        tm = search_tm,
        is_final = false,
        metrics = metrics,
        params = params,
        epoch = iters_search,
        trial = info.trial,
        trials = info.trials,
        global_best_score = info.global_best_score,
        phase = info.phase,
      })
    end
    return score, metrics, nil
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    trials = args.search_trials or 120,
    trial_fn = search_trial_fn,
    skip_final = true,
    cost_fn = args.cost_fn or function (p) return (p.features or max_features) * (p.clauses or 1) end,
    cost_beta = args.cost_beta,
    constrain = function (p) constrain_tm_params(p) end,
  })

  if search_tm then
    search_tm:destroy()
  end

  constrain_tm_params(best_params, args.output_weights, args.outputs)
  local final_tm = create_final_tm(args, best_params, max_features)
  local final_train_args = {
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    sol_offsets = args.sol_offsets,
    sol_neighbors = args.sol_neighbors,
    targets = args.targets,
  }
  if sparse then
    final_train_args.csc_offsets = args.csc_offsets
    final_train_args.csc_indices = args.csc_indices
    final_train_args.absorb_ranking = args.absorb_ranking
    final_train_args.absorb_ranking_offsets = args.absorb_ranking_offsets
    final_train_args.absorb_ranking_global = args.absorb_ranking_global
    final_train_args.tokens = args.tokens
    final_train_args.problems = nil
  end
  local _, final_metrics = train_tm_batched(final_tm, final_train_args, best_params, final_iters,
    final_batch, final_patience, args.early_tolerance, metric_fn, each_cb, { is_final = true })

  collectgarbage("collect")
  return final_tm, final_metrics, best_params
end

M.regressor = function (args)
  return optimize_tm(args)
end

M.spectral = function (args)
  local index = err.assert(args.index, "index required")
  local decay = args.decay or 0.0
  local each_cb = args.each
  local score_fn = args.score_fn or function (enc) return enc:trace_ratio() end

  local param_names = { "n_landmarks", "n_dims" }
  local samplers = M.build_samplers(args, param_names)

  local function trial_fn (params, info)
    local raw_codes, ids, encoder, eigenvalues =
      spectral.encode({
        inv = index,
        n_landmarks = params.n_landmarks,
        n_dims = params.n_dims,
        decay = decay,
      })
    local effective_dims = encoder and encoder:dims() or 0
    local actual_landmarks = encoder and encoder:n_landmarks() or 0
    local score = encoder and score_fn(encoder) or 0
    local trace_ratio = encoder and encoder:trace_ratio() or 0
    local result = {
      ids = ids,
      raw_codes = raw_codes,
      dims = effective_dims,
      landmark_ids = encoder and encoder:landmark_ids() or nil,
      n_landmarks = actual_landmarks,
      decay = decay,
      encoder = encoder,
      eigenvalues = eigenvalues,
    }
    if each_cb then
      each_cb({
        event = "spectral_result",
        n_landmarks = actual_landmarks, n_dims = effective_dims,
        n_embedded = ids and ids:size() or 0,
        trace_ratio = trace_ratio, decay = decay,
      })
    end
    return score, { trace_ratio = trace_ratio }, result
  end

  local result, best_params, best_metrics = M.search({
    param_names = param_names,
    samplers = samplers,
    trials = args.search_trials or 30,
    trial_fn = trial_fn,
    each = each_cb,
    cost_fn = args.cost_fn or function (p) return (p.n_landmarks or 1) * (p.n_dims or 1) end,
    cost_beta = args.cost_beta,
  })

  return result, best_params, best_metrics
end

M.ridge = function (args)
  local ridge = require("santoku.learn.ridge")
  local param_names = { "lambda", "propensity_a", "propensity_b" }
  local samplers = M.build_samplers(args, param_names)
  local k = args.k or 32
  local each_cb = args.each
  local score_fn = args.score_fn or function (ret) return ret.thresh.macro_f1 end
  local ga = {
    n_samples = args.n_samples,
    n_labels = args.n_labels,
    n_dims = args.n_dims,
    codes = args.codes,
    label_offsets = args.label_offsets,
    label_neighbors = args.label_neighbors,
  }
  local gram = ridge.precompute(ga)
  local full_data = {
    codes = args.codes, n_samples = args.n_samples,
    expected_offsets = args.expected_offsets, expected_neighbors = args.expected_neighbors,
  }
  local search_data = full_data
  local search_subsample = args.search_subsample
  if search_subsample and search_subsample < 1.0 then
    local search_n = num.floor(args.n_samples * search_subsample)
    local search_ids
    if args.stratify_offsets and args.stratify_neighbors and args.stratify_labels then
      search_ids = csr.stratified_sample(
        args.stratify_offsets, args.stratify_neighbors,
        args.n_samples, args.stratify_labels, search_n):asc()
      search_n = search_ids:size()
    else
      search_ids = ivec.create(args.n_samples):fill_indices():shuffle():setn(search_n):asc()
    end
    local function subsample_csr(offsets, neighbors, ids, n)
      local new_off = ivec.create(n + 1)
      local new_nbr = ivec.create()
      local pos = 0
      for i = 0, n - 1 do
        local sid = ids:get(i)
        local lo, hi = offsets:get(sid), offsets:get(sid + 1)
        new_off:set(i, pos)
        if hi > lo then
          new_nbr:copy(neighbors, lo, hi, pos)
        end
        pos = pos + hi - lo
      end
      new_off:set(n, pos)
      return new_off, new_nbr
    end
    local s_eoff, s_enbr = subsample_csr(args.expected_offsets, args.expected_neighbors, search_ids, search_n)
    search_data = { n_samples = search_n, expected_offsets = s_eoff, expected_neighbors = s_enbr }
    local s_codes = dvec.create(search_n * args.n_dims)
    args.codes:mtx_select(nil, search_ids, args.n_dims, s_codes)
    search_data.codes = s_codes
  end
  local wb
  local function trial_fn (params, info)
    local r; r, wb = ridge.create({
      gram = gram,
      lambda = params.lambda,
      propensity_a = params.propensity_a,
      propensity_b = params.propensity_b,
      w_buf = wb,
    })
    local data = info.is_final and full_data or search_data
    local pred_off, pred_nbr, pred_scores = r:label(data.codes, data.n_samples, k)
    local _, oracle, thresh = evaluator.retrieval_ks({
      pred_offsets = pred_off,
      pred_neighbors = pred_nbr,
      pred_scores = pred_scores,
      expected_offsets = data.expected_offsets,
      expected_neighbors = data.expected_neighbors,
    })
    local ret = { oracle = oracle, thresh = thresh }
    return score_fn(ret), ret, info.is_final and r or nil
  end
  local result, best_params, best_metrics = M.search({
    param_names = param_names,
    samplers = samplers,
    trials = args.search_trials or 30,
    trial_fn = trial_fn,
    each = each_cb,
    cost_fn = args.cost_fn or function () return 1 end,
    cost_beta = args.cost_beta,
  })
  return result, best_params, best_metrics
end

return M

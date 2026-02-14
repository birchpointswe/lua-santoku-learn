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
local utc = require("santoku.utc")

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

local function key_int (v)
  return v and tostring(num.floor(v + 0.5)) or "nil"
end

local function key_float2 (v)
  return v and str.format("%.2f", v) or "nil"
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
  local make_key = args.make_key
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local rerun_final = args.rerun_final ~= false
  local preference_tolerance = args.preference_tolerance or 1e-6
  local size_fn = args.size_fn or function() return 0 end
  local constrain_fn = args.constrain
  local n_candidates = args.n_candidates or 500
  local n_hyper_restarts = args.n_hyper_restarts or 20
  local best_score = -num.huge
  local best_size = nil
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

  local seen = {}

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
      end
      local ei = gp.suggest(X_obs, Y_obs, n_dims, cand_flat, n_candidates, n_hyper_restarts)
      local best_c = 0
      local best_ei = ei:get(0)
      for i = 1, n_candidates - 1 do
        local v = ei:get(i)
        if v > best_ei then
          best_ei = v
          best_c = i
        end
      end
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

      if each_cb then
        each_cb({
          event = "trial",
          trial = t,
          trials = trials,
          params = params,
          score = score,
          metrics = metrics,
          global_best_score = best_score,
          phase = phase,
        })
      end
      local current_size = size_fn(params)
      if is_preferred(score, current_size, best_score, best_size, preference_tolerance) then
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

local function create_tm (args)
  local opts = {
    features = args.features,
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
  return tm.create(opts)
end

local function create_final_tm (args, params)
  local opts = {
    features = args.features,
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
    opts.absorb_interval = params.absorb_interval
    opts.absorb_threshold = params.absorb_threshold
    opts.absorb_maximum = params.absorb_maximum
    opts.absorb_insert = params.absorb_insert
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
    solutions = args.solutions,
    codes = args.codes,
    targets = args.targets,
    iterations = iterations,
    grouped = args.grouped,
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
      solutions = args.solutions,
      codes = args.codes,
      targets = args.targets,
      iterations = batch_iters,
      grouped = args.grouped,
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
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each
  local search_subsample_samples = args.search_subsample_samples or args.search_subsample
  local search_subsample_targets = args.search_subsample_targets
  local sparse = not not args.n_tokens

  local param_names = { "clauses", "clause_maximum", "clause_tolerance_fraction", "target_fraction", "specificity" }
  if sparse then
    param_names[#param_names + 1] = "absorb_interval"
    param_names[#param_names + 1] = "absorb_threshold"
    param_names[#param_names + 1] = "absorb_maximum"
    param_names[#param_names + 1] = "absorb_insert_offset"
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

  local input_bits = 2 * args.features
  args.clause_maximum = cap_spec_max(args.clause_maximum, input_bits)
  args.specificity = cap_spec_max(args.specificity, input_bits)
  if sparse then
    local m = (args.state or 8) - 1
    args.absorb_maximum = cap_spec_max(args.absorb_maximum, args.features)
    args.absorb_threshold = cap_spec_max(args.absorb_threshold, 2 ^ m - 2)
    args.absorb_insert_offset = cap_spec_max(args.absorb_insert_offset, 2 ^ m - 1)
  end

  local samplers = M.build_samplers(args, param_names, global_dev)

  local search_n = args.samples
  local search_ids
  if search_subsample_samples and search_subsample_samples < 1.0 then
    search_n = num.floor(args.samples * search_subsample_samples)
    search_ids = ivec.create(args.samples):fill_indices():shuffle():setn(search_n):asc()
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
    search_weights = dvec.create(search_outputs)
    for i = 0, search_outputs - 1 do
      search_weights:set(i, args.output_weights:get(search_dim_ids:get(i)))
    end
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
      local search_cvec_features = args.cvec_features or ((args.grouped and args.outputs or 1) * args.features * 2)
      args.problems:bits_select(nil, search_ids, search_cvec_features, search_problems)
    end
    search_data = {
      samples = search_n,
      problems = search_problems,
      grouped = args.grouped,
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
  elseif args.solutions then
    if search_ids then
      search_data.solutions = ivec.create():copy(args.solutions, search_ids)
    else
      search_data.solutions = args.solutions
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
      features = args.features,
      outputs = search_outputs,
      state = args.state,
      n_tokens = args.n_tokens,
    })
  end

  local function constrain_tm_params (params, weights, outputs)
    local input_bits = args.features and 2 * args.features or nil
    if input_bits then
      if params.clause_maximum and params.clause_maximum > input_bits then
        params.clause_maximum = input_bits
      end
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
    if params.specificity and args.features then
      local max_specificity = 2 * args.features
      if params.specificity > max_specificity then
        params.specificity = max_specificity
      end
      if params.specificity < 1 then
        params.specificity = 1
      end
    end
    if params.absorb_maximum and args.features then
      if params.absorb_maximum > args.features then
        params.absorb_maximum = args.features
      end
    end
    if params.absorb_threshold or params.absorb_insert_offset then
      local m = (args.state or 8) - 1
      local max_excl = 2 ^ m - 1
      if params.absorb_threshold and params.absorb_threshold >= max_excl then
        params.absorb_threshold = max_excl - 1
      end
      if params.absorb_insert_offset then
        params.absorb_insert = num.min(params.absorb_threshold + params.absorb_insert_offset, max_excl)
      end
    end
    if weights then
      local n = outputs or weights:size()
      local features = args.features
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
    preference_tolerance = args.preference_tolerance or 1e-6,
    constrain = function (p) constrain_tm_params(p) end,
    size_fn = function(p) return { p.clauses or 0 } end,
    make_key = function (p)
      local k = str.format("%s|%s|%s|%s|%s",
        key_int(p.clauses),
        key_int(p.clause_tolerance),
        key_int(p.clause_maximum),
        key_int(p.target),
        key_float2(p.specificity))
      if p.absorb_interval then
        k = k .. "|" .. key_int(p.absorb_interval) .. "|" .. key_int(p.absorb_threshold) .. "|" .. key_int(p.absorb_maximum) .. "|" .. key_int(p.absorb_insert)
      end
      if p.alpha_specificity then
        k = k .. "|" .. key_float2(p.alpha_specificity)
            .. "|" .. key_float2(p.alpha_target)
            .. "|" .. key_float2(p.alpha_tolerance)
            .. "|" .. key_float2(p.alpha_maximum)
      end
      return k
    end,
  })

  if search_tm then
    search_tm:destroy()
  end

  constrain_tm_params(best_params, args.output_weights, args.outputs)
  local final_tm = create_final_tm(args, best_params)
  local final_train_args = {
    codes = args.codes,
    samples = args.samples,
    problems = args.problems,
    solutions = args.solutions,
    targets = args.targets,
    grouped = args.grouped,
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

M.destroy_spectral = function (model)
  if not model then return end
  if model.raw_codes then model.raw_codes:destroy() end
  if model.ids then model.ids:destroy() end
  if model.landmark_ids then model.landmark_ids:destroy() end
  if model.eigenvectors then model.eigenvectors:destroy() end
  if model.eigenvalues then model.eigenvalues:destroy() end
  if model.landmark_chol then model.landmark_chol:destroy() end
  if model.col_means then model.col_means:destroy() end
end

M.score_spectral_eval = function (args)
  local model = args.model
  local kernel_index = args.kernel_index

  local kernel_stats = nil
  if kernel_index and args.kernel_decay then
    kernel_stats = evaluator.ranking_accuracy({
      kernel_index = kernel_index,
      kernel_decay = args.kernel_decay,
      kernel_bandwidth = args.kernel_bandwidth,
      eval_ids = args.eval_ids,
      eval_offsets = args.eval_offsets,
      eval_neighbors = args.eval_neighbors,
      eval_weights = args.eval_weights,
      ranking = args.ranking,
    })
  end

  local raw_stats = evaluator.ranking_accuracy({
    raw_codes = model.raw_codes,
    ids = model.ids,
    eval_ids = args.eval_ids,
    eval_offsets = args.eval_offsets,
    eval_neighbors = args.eval_neighbors,
    eval_weights = args.eval_weights,
    ranking = args.ranking,
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
  local landmarks_index = args.landmarks_index or index
  local each_cb = args.each
  local decay = args.decay
  local bandwidth = args.bandwidth

  local raw_codes, ids, encoder, eigenvalues =
    spectral.encode({
      inv = index,
      landmarks_inv = landmarks_index ~= index and landmarks_index or nil,
      n_landmarks = args.n_landmarks or 0,
      n_dims = args.n_dims or args.n_landmarks or 0,
      decay = decay,
      bandwidth = bandwidth,
      trace_tol = args.trace_tol,
    })

  local effective_dims = encoder and encoder:dims() or 0
  local landmark_ids = encoder and encoder:landmark_ids() or nil
  local actual_landmarks = encoder and encoder:n_landmarks() or 0
  local trace_ratio = encoder and encoder:trace_ratio() or 0.0

  if each_cb then
    each_cb({
      event = "spectral_result",
      n_dims = effective_dims,
      n_landmarks = actual_landmarks,
      trace_ratio = trace_ratio,
    })
  end

  return {
    ids = ids,
    raw_codes = raw_codes,
    dims = effective_dims,
    spectral_dims = effective_dims,
    landmark_ids = landmark_ids,
    n_landmarks = actual_landmarks,
    decay = decay,
    bandwidth = bandwidth,
    encoder = encoder,
    eigenvalues = eigenvalues,
  }
end

M.spectral = function (args)
  local index = err.assert(args.index, "index required")
  local landmarks_index = args.landmarks_index
  local n_landmarks = args.n_landmarks
  local n_dims = args.n_dims
  if not n_landmarks and not n_dims then
    err.error("n_landmarks or n_dims required")
  end
  n_landmarks = n_landmarks or n_dims
  n_dims = n_dims or n_landmarks
  if n_dims > n_landmarks then
    n_dims = n_landmarks
  end
  local decay = args.decay or 0.0
  local bandwidth = args.bandwidth or -1.0
  local each_cb = args.each
  local params = { n_landmarks = n_landmarks, n_dims = n_dims, decay = decay, bandwidth = bandwidth }
  if each_cb then
    each_cb({ event = "sample", n_landmarks = n_landmarks, n_dims = n_dims, decay = decay, bandwidth = bandwidth })
  end
  local model = M.build_spectral_nystrom({
    index = index,
    landmarks_index = landmarks_index,
    n_landmarks = n_landmarks,
    n_dims = n_dims,
    decay = decay,
    bandwidth = bandwidth,
    trace_tol = args.trace_tol,
    each = each_cb,
    train_tokens = args.train_tokens,
    train_ids = args.train_ids,
  })
  local score = nil
  local metrics = nil
  if args.expected_ids and args.ranking then
    local eval_t0 = utc.time(true)
    score, metrics = M.score_spectral_eval({
      model = model,
      ranking = args.ranking,
      eval_ids = args.expected_ids,
      eval_offsets = args.expected_offsets,
      eval_neighbors = args.expected_neighbors,
      eval_weights = args.expected_weights,
      kernel_index = index,
      kernel_decay = decay,
      kernel_bandwidth = bandwidth,
    })
    local eval_t1 = utc.time(true)
    if each_cb then
      each_cb({ event = "eval", n_landmarks = n_landmarks, n_dims = n_dims, decay = decay, bandwidth = bandwidth, score = score, metrics = metrics, elapsed = eval_t1 - eval_t0 })
    end
  end
  if each_cb then
    each_cb({ event = "done", best_params = params, best_score = score, best_metrics = metrics })
  end
  return model, params
end

return M

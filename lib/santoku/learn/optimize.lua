local gp = require("santoku.learn.gp")

local dvec = require("santoku.dvec")
local num = require("santoku.num")
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

local build_sampler = function (spec, global_dev)
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
    }
  end
  if type(spec) == "table" and #spec > 0 then
    local k = #spec
    local val_to_idx = {}
    for i = 1, k do
      val_to_idx[spec[i]] = i - 1
    end
    return {
      type = "range",
      center = spec.def or spec[1],
      normalize = function (x)
        local idx = val_to_idx[x] or 0
        return (idx + 0.5) / k
      end,
      denormalize = function (u)
        local idx = num.floor(u * k)
        if idx < 0 then idx = 0 end
        if idx >= k then idx = k - 1 end
        return spec[idx + 1]
      end,
      sample = function ()
        local idx = num.floor(rand.fast_random() / (rand.fast_max + 1) * k) + 1
        return spec[idx]
      end,
    }
  end
  err.error("Bad hyper-parameter specification", spec)
end

local build_samplers = function (args, param_names, global_dev)
  local samplers = {}
  for _, pname in ipairs(param_names) do
    samplers[pname] = build_sampler(args[pname], global_dev)
  end
  return samplers
end

local sample_params = function (samplers, param_names, base_cfg, use_exact_defaults)
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
      else
        p[name] = s.sample()
      end
    end
  end
  return p
end

local all_fixed = function (samplers)
  for _, s in pairs(samplers) do
    if s and s.type ~= "fixed" then
      return false
    end
  end
  return true
end

local search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local trials = args.trials or 120
  local each_cb = args.each
  local skip_final = args.skip_final
  local constrain_fn = args.constrain
  local cost_fn = args.cost_fn
  local cost_beta = args.cost_beta or 0.0
  local n_candidates = args.n_candidates or 500
  local n_hyper_restarts = args.n_hyper_restarts or 20
  local best_score = -num.huge
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if all_fixed(samplers) or trials <= 0 then
    best_params = sample_params(samplers, param_names, nil, true)
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

  local seed = 2166136261
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s and s.center ~= nil then
      local v = s.normalize and s.normalize(s.center) or (type(s.center) == "number" and s.center or 0)
      seed = (seed * 16777619 + num.floor(v * 2147483647)) % 4294967296
    end
  end
  rand.fast_seed(seed)

  local lhs_pts = n_dims > 0 and lhs_sample(n_initial, n_dims) or nil
  if lhs_pts then
    local def_pt = {}
    for i, name in ipairs(search_dims) do
      local s = samplers[name]
      def_pt[i] = s.center ~= nil and s.normalize(s.center) or 0.5
    end
    table.insert(lhs_pts, 1, def_pt)
    n_initial = n_initial + 1
  end
  local cand_flat = n_dims > 0 and dvec.create(n_candidates * n_dims) or nil
  local cand_costs = (n_dims > 0 and cost_fn and cost_beta > 0) and dvec.create(n_candidates) or nil
  local ei_buf

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
      for i = 1, n_candidates do
        local pt = cand_pts[i]
        local p = {}
        for j, name in ipairs(search_dims) do
          p[name] = samplers[name].denormalize(pt[j])
        end
        if constrain_fn then constrain_fn(p) end
        local base = (i - 1) * n_dims
        for j, name in ipairs(search_dims) do
          cand_flat:set(base + j - 1, samplers[name].normalize(p[name]))
        end
        if cand_costs then cand_costs:set(i - 1, cost_fn(p)) end
      end
      ei_buf = gp.suggest(X_obs, Y_obs, n_dims, cand_flat, n_candidates, n_hyper_restarts, ei_buf)
      if cand_costs then
        local _, cmax = cand_costs:max()
        if cmax > 0 then cand_costs:scale(1.0 / cmax) end
        ei_buf:scalev(cand_costs:clamp(1e-12, 1.0):pow(-cost_beta))
      end
      local _, best_c = ei_buf:max()
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
      best_score = score
      best_params = params
      best_result = result
      best_metrics = metrics
    end
  end

  collectgarbage("collect")

  if not skip_final and best_params then
    local _, final_metrics, final_result = trial_fn(best_params, { is_final = true })
    best_result = final_result
    best_metrics = final_metrics
  end

  return best_result, best_params, best_metrics

end

M.ridge = function (args)
  local ridge = require("santoku.learn.ridge")
  local dense = args.val_targets ~= nil
  local gram = args.gram
  if not gram then
    local train_codes = err.assert(args.train_codes, "train_codes required")
    local n_samples = err.assert(args.n_samples, "n_samples required")
    local n_dims = err.assert(args.n_dims, "n_dims required")
    gram = ridge.gram({
      codes = train_codes, n_samples = n_samples, n_dims = n_dims,
      label_offsets = args.label_offsets, label_neighbors = args.label_neighbors,
      label_values = args.label_values, n_labels = args.n_labels,
      targets = args.targets, n_targets = args.n_targets,
    })
  end
  local param_names
  if dense then param_names = { "lambda" }
  else param_names = { "lambda", "propensity_a", "propensity_b" } end
  args.lambda = spec_defaults(args.lambda, { min = 0, max = 4, def = 1.0 })
  args.propensity_a = spec_defaults(args.propensity_a, { min = 0, max = 4.0, def = 0.5 })
  args.propensity_b = spec_defaults(args.propensity_b, { min = 0, max = 8.0, def = 1.5 })
  local samplers = build_samplers(args, param_names)
  local k = not dense and (args.k or 32) or nil
  if all_fixed(samplers) or not args.search_trials or args.search_trials <= 0 then
    local params = sample_params(samplers, param_names, nil, true)
    local r = ridge.create({
      gram = gram, lambda = params.lambda,
      propensity_a = not dense and params.propensity_a or nil,
      propensity_b = not dense and params.propensity_b or nil,
    })
    return r, params
  end
  gram:prepare(args.val_codes, args.val_n_samples)
  local trial_fn = args.trial_fn
  if not trial_fn then
    if dense then
      trial_fn = function (g, params)
        local mae, nmae = g:regress_accuracy(params.lambda, nil, nil, args.val_targets)
        return -mae, { mae = mae, nmae = nmae }
      end
    else
      trial_fn = function (g, params)
        local f1, p, r = g:label_accuracy(params.lambda, k,
          params.propensity_a, params.propensity_b,
          args.val_expected_offsets, args.val_expected_neighbors)
        return f1, { f1 = f1, precision = p, recall = r }
      end
    end
  end
  local _, best_params = search({
    param_names = param_names, samplers = samplers,
    trials = args.search_trials or 30,
    trial_fn = function (params) return trial_fn(gram, params) end,
    each = args.each, skip_final = true,
  })
  local r = ridge.create({
    gram = gram, lambda = best_params.lambda,
    propensity_a = not dense and best_params.propensity_a or nil,
    propensity_b = not dense and best_params.propensity_b or nil,
  })
  return r, best_params
end

M.krr = function (args)
  local spectral = require("santoku.learn.spectral")
  local ridge = require("santoku.learn.ridge")
  local dense = args.val_targets ~= nil
  local kernel_spec = args.kernel or "cosine"
  local kernels = type(kernel_spec) == "table" and kernel_spec or { kernel_spec }
  local param_names
  if dense then param_names = { "kernel", "lambda" }
  else param_names = { "kernel", "lambda", "propensity_a", "propensity_b" } end
  args.kernel = kernels
  args.lambda = spec_defaults(args.lambda, { min = 0, max = 4, def = 1.0 })
  args.propensity_a = spec_defaults(args.propensity_a, { min = 0, max = 4.0, def = 0.5 })
  args.propensity_b = spec_defaults(args.propensity_b, { min = 0, max = 8.0, def = 1.5 })
  local samplers = build_samplers(args, param_names)
  local do_search = not all_fixed(samplers) and args.search_trials and args.search_trials > 0
  local k = not dense and (args.k or 32) or nil
  local use_tile = not do_search and not dense and args.label_tile_size and args.label_tile_size > 0
  local spectral_args = {
    offsets = args.offsets, tokens = args.tokens, values = args.values,
    n_tokens = args.n_tokens, n_samples = args.n_samples,
    codes = args.codes, d_input = args.d_input,
    bits = args.bits, d_bits = args.d_bits,
    n_landmarks = args.n_landmarks, trace_tol = args.trace_tol,
    label_offsets = args.label_offsets, label_neighbors = args.label_neighbors,
    label_values = args.label_values, n_labels = args.n_labels,
    targets = args.targets, n_targets = args.n_targets,
  }
  if use_tile then
    local params = sample_params(samplers, param_names, nil, true)
    spectral_args.kernel = params.kernel
    spectral_args.label_tile_size = args.label_tile_size
    spectral_args.lambda = params.lambda
    spectral_args.propensity_a = params.propensity_a
    spectral_args.propensity_b = params.propensity_b
    local _, sp_enc, tiled = spectral.encode(spectral_args)
    local r = ridge.create({
      W = tiled.W, intercept = tiled.intercept,
      n_dims = tiled.n_dims, n_labels = tiled.n_labels,
    })
    local val_codes
    if args.val_encode then
      val_codes = args.val_encode(sp_enc)
    else
      val_codes = sp_enc:encode({
        offsets = args.val_offsets, tokens = args.val_tokens,
        values = args.val_values, n_samples = args.val_n_samples,
      })
    end
    return sp_enc, r, val_codes, params
  end
  local kernel_data = {}
  local encode_kernels = do_search and kernels or { samplers.kernel.center or kernels[1] }
  for _, kname in ipairs(encode_kernels) do
    spectral_args.kernel = kname
    local _, sp_enc, gram = spectral.encode(spectral_args)
    local val_codes
    if args.val_encode then
      val_codes = args.val_encode(sp_enc)
    else
      val_codes = sp_enc:encode({
        offsets = args.val_offsets, tokens = args.val_tokens,
        values = args.val_values, n_samples = args.val_n_samples,
      })
    end
    gram:prepare(val_codes, args.val_n_samples)
    kernel_data[kname] = { sp_enc = sp_enc, gram = gram, val_codes = val_codes }
  end
  collectgarbage("collect")
  if not do_search then
    local params = sample_params(samplers, param_names, nil, true)
    local kd = kernel_data[params.kernel]
    local r = ridge.create({
      gram = kd.gram, lambda = params.lambda,
      propensity_a = not dense and params.propensity_a or nil,
      propensity_b = not dense and params.propensity_b or nil,
    })
    return kd.sp_enc, r, kd.val_codes, params
  end
  local trial_fn = args.trial_fn
  if not trial_fn then
    if dense then
      trial_fn = function (g, params)
        local mae, nmae = g:regress_accuracy(params.lambda, nil, nil, args.val_targets)
        return -mae, { mae = mae, nmae = nmae }
      end
    else
      trial_fn = function (g, params)
        local f1, p, r = g:label_accuracy(params.lambda, k,
          params.propensity_a, params.propensity_b,
          args.val_expected_offsets, args.val_expected_neighbors)
        return f1, { f1 = f1, precision = p, recall = r }
      end
    end
  end
  local _, best_params = search({
    param_names = param_names, samplers = samplers,
    trials = args.search_trials or 30,
    trial_fn = function (params)
      local kd = kernel_data[params.kernel]
      return trial_fn(kd.gram, params)
    end,
    each = args.each, skip_final = true,
  })
  local best_kd = kernel_data[best_params.kernel]
  local r = ridge.create({
    gram = best_kd.gram, lambda = best_params.lambda,
    propensity_a = not dense and best_params.propensity_a or nil,
    propensity_b = not dense and best_params.propensity_b or nil,
  })
  return best_kd.sp_enc, r, best_kd.val_codes, best_params
end

M.gfm = function (args)
  local gfm = require("santoku.learn.gfm")
  local g = gfm.create({ n_labels = args.n_labels })
  local best_f1 = g:calibrate({
    offsets = args.val_offsets,
    neighbors = args.val_neighbors,
    scores = args.val_scores,
    n_samples = args.val_n_samples,
    expected_offsets = args.val_expected_offsets,
    expected_neighbors = args.val_expected_neighbors,
  })
  return g, { f1 = best_f1 }
end

return M

local gp = require("santoku.learn.gp")

local dvec = require("santoku.dvec")
local num = require("santoku.num")
local err = require("santoku.error")
local rand = require("santoku.random")

local M = {}

M.plateaus = function (curve, n, klo, khi)
  klo = klo or 2
  khi = khi or #curve
  local m = khi - klo + 1
  if n >= m then local out = {}; for k = klo, khi do out[#out + 1] = k end; return out end
  local s1, s2 = { [0] = 0 }, { [0] = 0 }
  for i = 1, m do local v = curve[klo + i - 1]; s1[i] = s1[i - 1] + v; s2[i] = s2[i - 1] + v * v end
  local function ssd (a, b)
    local c = b - a + 1; local sm = s1[b] - s1[a - 1]
    return (s2[b] - s2[a - 1]) - sm * sm / c
  end
  local cost, back = {}, {}
  for c = 1, n do cost[c] = {}; back[c] = {} end
  for i = 1, m do cost[1][i] = ssd(1, i); back[1][i] = 1 end
  for c = 2, n do
    for i = c, m do
      local bc, bj = math.huge, c
      for j = c - 1, i - 1 do
        local cc = cost[c - 1][j] + ssd(j + 1, i)
        if cc < bc then bc, bj = cc, j + 1 end
      end
      cost[c][i], back[c][i] = bc, bj
    end
  end
  local cuts, i = {}, m
  for c = n, 1, -1 do
    local start = back[c][i]
    cuts[#cuts + 1] = klo + start - 1
    i = start - 1
  end
  table.sort(cuts)
  return cuts
end

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
  if spec == nil then return defs end
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

  if args.reseed ~= false then
    local seed = 2166136261
    for _, name in ipairs(param_names) do
      local s = samplers[name]
      if s and s.center ~= nil then
        local v = s.normalize and s.normalize(s.center) or (type(s.center) == "number" and s.center or 0)
        seed = (seed * 16777619 + num.floor(v * 2147483647)) % 4294967296
      end
    end
    rand.fast_seed(seed)
  end

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
  local ei_buf

  -- top up any params not covered by the search dims (fixed -> center, else fresh sample)
  local function fill_rest (params)
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
      fill_rest(params)
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
      end
      ei_buf = gp.suggest(X_obs, Y_obs, n_dims, cand_flat, n_candidates, n_hyper_restarts, ei_buf)
      local _, best_c = ei_buf:max()
      params = {}
      for i, name in ipairs(search_dims) do
        params[name] = samplers[name].denormalize(cand_flat:get(best_c * n_dims + i - 1))
      end
      fill_rest(params)
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

-- All trials score through the SAME decide core used at deployment, fed by the gram's pure
-- producers (regress -> dense scores; label -> tiled top-k P). No gram-internal metric, no
-- per-trial ridge build. Dense regression has no decision, so it uses the nmae helper.
local function default_trial_fn (args, dense, metric, k)
  local fvec = require("santoku.fvec")
  if dense then
    local eval = require("santoku.learn.evaluator")
    local sbuf = fvec.create()
    return function (g, params)
      local s = g:regress(params.lambda, nil, nil, sbuf)
      local m = eval.regress_accuracy(s, args.val_targets)
      return -m.mean, { mae = m.mean, nmae = m.nmae }
    end
  end
  local decide = require("santoku.learn.decide")
  local nl, vn = args.n_labels, args.val_n_samples
  if metric == "span" then
    local cand, gold = args.val_cand, args.val_gold
    local n_docs = cand:offsets():size() - 1
    local sbuf = fvec.create()
    local probe = decide.create({ n_labels = nl, span = true, reject = args.reject })
    return function (g, params)
      local s = g:regress(params.lambda, params.propensity_a, params.propensity_b, sbuf)
      local f1 = probe:calibrate({ scores = s, n_samples = n_docs, cand = cand, gold = gold })
      return f1, { span_f1 = f1 }
    end
  end
  if metric == "single" then
    local sbuf = fvec.create()
    local probe = decide.create({ n_labels = nl, single = true })
    return function (g, params)
      local s = g:regress(params.lambda, params.propensity_a, params.propensity_b, sbuf)
      local macro = probe:calibrate({ scores = s, n_samples = vn, expected = args.val_y })
      return macro, { macro_f1 = macro }
    end
  end
  -- multilabel: gram tiled top-k producer -> P csr -> decide multilabel calibrate
  local csr = require("santoku.csr")
  local ivec = require("santoku.ivec")
  local off, lbl, sco = ivec.create(), ivec.create(), fvec.create()
  local probe = decide.create({ n_labels = nl })
  return function (g, params)
    local o, l, s = g:label(params.lambda, k, params.propensity_a, params.propensity_b, off, lbl, sco)
    local P = csr.create({ offsets = o, neighbors = l, values = s, n_cols = nl })
    local f1, p, r = probe:calibrate({ pred = P, n_samples = vn, expected = args.val_y })
    return f1, { f1 = f1, precision = p, recall = r }
  end
end

local function add_label_params (param_names, args, dense)
  args.lambda = spec_defaults(args.lambda, { min = 1e-4, max = 16, log = true, def = 1.0 })
  param_names[#param_names + 1] = "lambda"
  if not dense and (args.n_labels or 0) > 1 then
    args.propensity_a = spec_defaults(args.propensity_a, { min = 0, max = 8.0, def = 0.5 })
    args.propensity_b = spec_defaults(args.propensity_b, { min = 0, max = 16.0, def = 1.5 })
    param_names[#param_names + 1] = "propensity_a"
    param_names[#param_names + 1] = "propensity_b"
  end
end

local function decode_mode (args, dense)
  if dense or args.trial_fn ~= nil then return false, nil end
  if args.val_cand then return true, "span" end
  if (args.n_labels or 0) > 1 and args.val_y then
    local eo = args.val_y:offsets()
    for i = 0, args.val_n_samples - 1 do
      if eo:get(i + 1) - eo:get(i) ~= 1 then return true, "multilabel" end
    end
    return true, "single"
  end
  return true, "multilabel"
end

local function bundle_decider (r, val_codes, args, mode, k)
  if mode == "span" then
    local scores = r:regress(val_codes)
    return M.decide({ n_labels = args.n_labels, reject = args.reject, val_scores = scores,
      val_n_samples = args.val_cand:offsets():size() - 1,
      val_cand = args.val_cand, val_gold = args.val_gold })
  end
  if mode == "single" then
    local scores = r:regress(val_codes)
    return M.decide({ n_labels = args.n_labels, val_scores = scores,
      val_n_samples = args.val_n_samples, val_expected = args.val_y })
  end
  local P = r:label(val_codes, k)
  return M.decide({ n_labels = args.n_labels, val_pred = P,
    val_n_samples = args.val_n_samples, val_expected = args.val_y })
end

M.ridge = function (args)
  local ridge = require("santoku.learn.ridge")
  -- Object API: train_codes/val_codes are mtx codes, y is a labels csr, targets a dvec.
  if args.y and not args.n_labels then args.n_labels = select(2, args.y:shape()) end
  if args.val_codes and not args.val_n_samples then args.val_n_samples = (args.val_codes:shape()) end
  local dense = args.val_targets ~= nil
  local param_names = {}
  add_label_params(param_names, args, dense)
  local samplers = build_samplers(args, param_names)
  local k = not dense and (args.k or 32) or nil
  local want_decode, mode = decode_mode(args, dense)
  local locked = all_fixed(samplers) or not args.search_trials or args.search_trials <= 0
  local locked_params = locked and sample_params(samplers, param_names, nil, true) or nil
  local gram = args.gram
  if not gram then
    local gram_args = {
      x = err.assert(args.train_codes, "train_codes (mtx) required"),
      y = args.y,
      targets = args.targets, n_targets = args.n_targets,
    }
    if locked then
      gram_args.solve_lambda = locked_params.lambda
      if not dense then
        gram_args.solve_propensity_a = locked_params.propensity_a
        gram_args.solve_propensity_b = locked_params.propensity_b
      end
    end
    gram = ridge.gram(gram_args)
  end
  local baked = locked and args.gram == nil
  local function finish (r, params, solve)
    if args.each then args.each({ event = "done", params = params, solve = solve }) end
    local decider, decider_metrics
    if want_decode then
      decider, decider_metrics = bundle_decider(r, args.val_codes, args, mode, k)
    end
    return r, params, decider, decider_metrics
  end
  if locked then
    local params = locked_params
    local r = ridge.create({
      gram = gram, lambda = params.lambda,
      propensity_a = not dense and params.propensity_a or nil,
      propensity_b = not dense and params.propensity_b or nil,
    })
    return finish(r, params, baked and "cholesky" or "eigen")
  end
  gram:prepare(args.val_codes, args.val_n_samples)
  local trial_fn = args.trial_fn or default_trial_fn(args, dense, mode == "multilabel" and "fmeasure" or mode, k)
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
  return finish(r, best_params, "eigen")
end

M.krr = function (args)
  local spectral = require("santoku.learn.spectral")
  local ridge = require("santoku.learn.ridge")
  local fvec = require("santoku.fvec")
  -- Object API: x/val_x (csr features or mtx codes), y/val_y (labels csr) flow as objects;
  -- spectral.encode and the decode helpers consume them directly. Only scalars are read here.
  if args.x ~= nil then
    local r, c = args.x:shape()
    args.n_samples = args.n_samples or r
    if args.x.neighbors then args.n_tokens = args.n_tokens or c
    else args.d_input = args.d_input or c end
  end
  if args.y ~= nil then
    local _, c = args.y:shape()
    args.n_labels = args.n_labels or c
  end
  if args.val_x ~= nil then
    args.val_n_samples = args.val_n_samples or (args.val_x:shape())
  end
  -- val_y stays an object; decode helpers consume it directly (expected = args.val_y)
  local dense = args.val_targets ~= nil
  local kernel_spec = args.kernel or "cosine"
  local kernels = type(kernel_spec) == "table" and kernel_spec or { kernel_spec }
  args.kernel = kernels
  local families = {}
  for _, kn in ipairs(kernels) do
    if kn == "cosine" then families.cosine = true
    elseif kn == "arccos" or kn == "arccos1" then families.arccos = true
    else families.matern = true end
  end
  -- Canonical kernel hyperparam specs, shared by the locked and search paths. gamma is a log
  -- range; nu/order/depth/tangent are categorical. In every case `def` only warm-starts the
  -- search (the center), it does not lock; pass a scalar to lock or a list to prune/reorder.
  local function cat_spec (v, deflist)
    if v == nil then return deflist end
    if type(v) ~= "table" then return v end
    if #v > 0 then return v end
    local s = {}
    for i = 1, #deflist do s[i] = deflist[i] end
    s.def = v.def
    return s
  end
  args.gamma = spec_defaults(args.gamma, { min = 1e-2, max = 16, log = true, def = 1.0 })
  args.nu = cat_spec(args.nu, { 3, 0, 1, 2 })
  -- arccos grid pruned to the live region: order 0/5/6 and depth 3 never won anywhere (depth 3
  -- collapses; order 0 = step is always worst; orders 5/6 plateau). 16-cell grid (4x2x2).
  args.order = cat_spec(args.order, { 1, 2, 3, 4 })
  args.depth = cat_spec(args.depth, { 1, 2 })
  args.tangent = cat_spec(args.tangent, { 0, 1 })
  local kernel_samplers = build_samplers(args, { "nu", "gamma", "order", "depth", "tangent" })
  local do_search = args.search_trials and args.search_trials > 0
  local inner_names = {}
  add_label_params(inner_names, args, dense)
  if args.extra then
    for _, e in ipairs(args.extra) do
      inner_names[#inner_names + 1] = e.name
      args[e.name] = spec_defaults({ min = e.min, max = e.max, log = e.log, def = e.def },
        { min = 0, max = 1, def = 0.5 })
    end
  end
  local inner_samplers = build_samplers(args, inner_names)
  local inner_trials = args.inner_trials or 80
  local k = not dense and (args.k or 32) or nil
  local tiled = not dense
  local tile_labels = tiled and (args.tile_labels or 1024) or nil
  local want_decode, mode = decode_mode(args, dense)
  local spectral_args = {
    x = args.x, y = args.y,
    n_tokens = args.n_tokens, n_samples = args.n_samples,
    d_input = args.d_input,
    n_landmarks = args.n_landmarks, trace_tol = args.trace_tol,
    chol_block = args.chol_block,
    n_labels = args.n_labels,
    targets = args.targets, n_targets = args.n_targets,
  }
  -- Auto scratch: allocate the tiled buffers ONCE and reuse them across every build_kd, so a search does
  -- not re-alloc (and the OS re-fault) the big n_landmarks*n_samples chol buffer per trial -- that churn,
  -- not just the size, is what hurts. The caller passes chol_buf/pqty_buf/w_buf ONLY to control storage
  -- (e.g. fvec.mmap_create to keep them off RAM, or to persist); nil => we reuse one RAM buffer here.
  local pqty_auto, w_auto
  if tiled then
    spectral_args.tile_labels = tile_labels
    spectral_args.tile_samples = args.tile_samples
    spectral_args.chol_buf = args.chol_buf or fvec.create(args.n_landmarks * args.n_samples)
    pqty_auto = args.pqty_buf or fvec.create(args.n_landmarks * (args.n_labels or 1))
    w_auto = args.w_buf or fvec.create(args.n_landmarks * (args.n_labels or 1))
  end
  local seed = args.seed or 1
  local function val_encode (sp_enc)
    if args.val_encode then return args.val_encode(sp_enc) end
    return sp_enc:encode(args.val_x)
  end
  -- solve == nil: search path (gram left unsolved, prepared for per-trial scoring). solve set:
  -- locked path (lambda/propensity baked into the cholesky during encode, no prepare needed).
  local function build_kd (spec, solve)
    collectgarbage("collect")
    spectral_args.kernel = spec.kernel
    spectral_args.gamma = spec.gamma
    spectral_args.nu = spec.nu
    spectral_args.order = spec.order
    spectral_args.depth = spec.depth
    spectral_args.tangent = spec.tangent
    spectral_args.solve_lambda = solve and solve.lambda or nil
    spectral_args.solve_propensity_a = solve and not dense and solve.propensity_a or nil
    spectral_args.solve_propensity_b = solve and not dense and solve.propensity_b or nil
    if tiled then
      local lbl = spec.label or spec.kernel
      spectral_args.pqty_buf = type(pqty_auto) == "function"
        and pqty_auto(lbl) or pqty_auto
    end
    rand.fast_seed(seed)
    local _, sp_enc, gram = spectral.encode(spectral_args)
    local val_codes = val_encode(sp_enc)
    if not solve then gram:prepare(val_codes, args.val_n_samples) end
    return { sp_enc = sp_enc, gram = gram, val_codes = val_codes }
  end
  local function finish (kd, params, solve)
    local r = ridge.create({
      gram = kd.gram, lambda = params.lambda,
      propensity_a = not dense and params.propensity_a or nil,
      propensity_b = not dense and params.propensity_b or nil,
      w_buf = tiled and w_auto or nil,
      tile_labels = tiled and tile_labels or nil,
    })
    if args.each then args.each({ event = "done", params = params, emb_d = kd.sp_enc:dims(),
      solve = solve }) end
    local decider, decider_metrics
    if want_decode then
      decider, decider_metrics = bundle_decider(r, kd.val_codes, args, mode, k)
    end
    return kd.sp_enc, r, kd.val_codes, params, decider, decider_metrics
  end
  if not do_search then
    local kname = kernels.def or kernels[1]
    local spec = { kernel = kname }
    if kname == "matern" then
      spec.nu = kernel_samplers.nu.center
      spec.gamma = kernel_samplers.gamma.center
    elseif kname == "rbf" then
      spec.gamma = kernel_samplers.gamma.center
    elseif kname == "arccos" then
      spec.order = kernel_samplers.order.center
      spec.depth = kernel_samplers.depth.center
      spec.tangent = kernel_samplers.tangent.center
    end
    local lp = sample_params(inner_samplers, inner_names, nil, true)
    local kd = build_kd(spec, lp)
    local params = { kernel = kname, gamma = spec.gamma, nu = spec.nu,
      order = spec.order, depth = spec.depth, tangent = spec.tangent }
    for _, n in ipairs(inner_names) do params[n] = lp[n] end
    return finish(kd, params, "cholesky")
  end
  local trial_fn = args.trial_fn or default_trial_fn(args, dense, mode == "multilabel" and "fmeasure" or mode, k)
  local matern_trials = args.matern_trials or args.search_trials or 0
  local arccos_trials = args.arccos_trials or args.search_trials or 0
  -- arccos is a small discrete grid (order x depth x tangent); never run (or count) more trials
  -- than unique cells -- the rest are just dedup-cache hits.
  if families.arccos then
    local function grid_size (v) return type(v) == "table" and #v or 1 end
    local cells = grid_size(args.order) * grid_size(args.depth) * grid_size(args.tangent)
    if arccos_trials > cells then arccos_trials = cells end
  end
  local btot = (families.cosine and 1 or 0)
    + (families.matern and matern_trials or 0)
    + (families.arccos and arccos_trials or 0)
  local best_params, best_score = nil, -num.huge
  local bi = 0
  local function eval_kd (kd, base)
    local _, ib = search({
      param_names = inner_names, samplers = inner_samplers, trials = inner_trials,
      trial_fn = function (p) return trial_fn(kd.gram, p, kd) end,
      skip_final = true, reseed = false,
    })
    for kk, vv in pairs(base) do ib[kk] = vv end
    local sc, sm = trial_fn(kd.gram, ib, kd)
    bi = bi + 1
    if args.each then
      args.each({ event = "trial", phase = "kernel", trial = bi, trials = btot,
        params = ib, score = sc, metrics = sm, emb_d = kd.sp_enc:dims(),
        global_best_score = best_score, is_new_best = sc > best_score })
    end
    if sc > best_score then best_score, best_params = sc, ib end
    return sc, sm
  end
  if families.cosine then
    eval_kd(build_kd({ kernel = "cosine", label = "cosine" }), { kernel = "cosine" })
  end
  if families.matern then
    local m_samplers = { nu = kernel_samplers.nu, gamma = kernel_samplers.gamma }
    search({
      param_names = { "nu", "gamma" }, samplers = m_samplers,
      trials = matern_trials, skip_final = true,
      trial_fn = function (gp)
        local kd = build_kd({ kernel = "matern", nu = gp.nu, gamma = gp.gamma, label = "matern" })
        return eval_kd(kd, { kernel = "matern", nu = gp.nu, gamma = gp.gamma })
      end,
    })
  end
  if families.arccos then
    local a_samplers = { order = kernel_samplers.order, depth = kernel_samplers.depth,
      tangent = kernel_samplers.tangent }
    local seen = {}
    search({
      param_names = { "order", "depth", "tangent" }, samplers = a_samplers,
      trials = arccos_trials, skip_final = true,
      trial_fn = function (gp)
        local sig = gp.order .. ":" .. gp.depth .. ":" .. gp.tangent
        local hit = seen[sig]
        if hit then return hit[1], hit[2] end
        local kd = build_kd({ kernel = "arccos", order = gp.order, depth = gp.depth,
          tangent = gp.tangent, label = "arccos" })
        local sc, sm = eval_kd(kd, { kernel = "arccos", order = gp.order,
          depth = gp.depth, tangent = gp.tangent })
        seen[sig] = { sc, sm }
        return sc, sm
      end,
    })
  end
  local best_kd = build_kd(best_params)
  return finish(best_kd, best_params, "eigen")
end

M.decide = function (args)
  local decide = require("santoku.learn.decide")
  if args.val_cand ~= nil then
    local g = decide.create({ n_labels = args.n_labels, span = true, reject = args.reject })
    local f1, precision, recall = g:calibrate({
      scores = args.val_scores, n_samples = args.val_n_samples,
      cand = args.val_cand, gold = args.val_gold,
    })
    return g, { span_f1 = f1, precision = precision, recall = recall, f1 = f1 }
  end
  local single = args.val_pred == nil
  local g = decide.create({ n_labels = args.n_labels, single = single })
  if single then
    local macro_f1, accuracy = g:calibrate({
      scores = args.val_scores,
      n_samples = args.val_n_samples,
      expected = args.val_expected,
    })
    return g, { macro_f1 = macro_f1, accuracy = accuracy }
  end
  local best_f1, precision, recall = g:calibrate({
    pred = args.val_pred,
    n_samples = args.val_n_samples,
    expected = args.val_expected,
  })
  return g, { f1 = best_f1, precision = precision, recall = recall }
end

return M

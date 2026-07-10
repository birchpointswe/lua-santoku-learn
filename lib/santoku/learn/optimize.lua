local gp = require("santoku.learn.gp")

local dvec = require("santoku.dvec")
local num = require("santoku.num")
local err = require("santoku.error")
local rand = require("santoku.random")
local utc = require("santoku.utc")

local M = {}

local SEARCH = setmetatable({}, { __tostring = function () return "optimize.SEARCH" end })
M.SEARCH = SEARCH

local tt
local function tick (name) return tt and tt(name) or nil end
local function tock (stop) if stop then stop() end end
local function prof_add_raw (name, dt)
  if not tt then return end
  local stats = tt()
  local e = stats[name]; if not e then e = { time = 0, count = 0 }; stats[name] = e end
  e.time = e.time + dt; e.count = e.count + 1
end

local function make_gp_model (n_dims, cap, restarts)
  local X, Y, N = dvec.create(), dvec.create(), dvec.create()
  local state, ls = dvec.create(), dvec.create()
  local ei = nil
  local n, ls_ready = 0, false
  local function subset ()
    if n <= cap then return X, Y, N end
    local half = num.floor(cap / 2)
    local order = Y:argsort_desc()
    local keep, seen = {}, {}
    for i = 0, half - 1 do local ix = order:get(i); keep[#keep + 1] = ix; seen[ix] = true end
    local i = n - 1
    while #keep < cap and i >= 0 do
      if not seen[i] then keep[#keep + 1] = i; seen[i] = true end
      i = i - 1
    end
    local gX, gY, gN = dvec.create(), dvec.create(), dvec.create()
    for _, ix in ipairs(keep) do
      for j = 0, n_dims - 1 do gX:push(X:get(ix * n_dims + j)) end
      gY:push(Y:get(ix)); gN:push(N:get(ix))
    end
    return gX, gY, gN
  end
  return {
    observe = function (row, y, noise)
      for j = 1, n_dims do X:push(row[j]) end
      Y:push(y); N:push(noise)
      n = n + 1
    end,
    suggest = function (cand_flat, n_candidates)
      local gX, gY, gN = subset()
      ei = gp.suggest(gX, gY, n_dims, cand_flat, n_candidates, restarts, ei, state, ls, gN)
      ls_ready = true
      return ei
    end,
    lengthscales = function () return ls_ready and ls or nil end,
  }
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
  if spec == nil or spec == SEARCH then return defs end
  if type(spec) ~= "table" then return spec end
  local s = {}
  for k, v in pairs(defs) do s[k] = v end
  for k, v in pairs(spec) do s[k] = v end
  return s
end

local function veclen (t)
  local n = 0
  for k in pairs(t) do if type(k) == "number" and k > n then n = k end end
  return n
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
      center = spec.def,
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

local DEF_DRAW_OFFSET = 1013904223
local function def_uniform (seed, name)
  local s = (seed + DEF_DRAW_OFFSET) % 2147483647
  if s == 0 then s = 1 end
  for i = 1, #name do
    s = (s + name:byte(i)) % 2147483647
    s = (s * 48271) % 2147483647
  end
  s = (s * 48271) % 2147483647
  return s / 2147483647
end

local function seed_center (s, name, seed)
  if s and s.type == "range" and s.center == nil and s.denormalize then
    s.center = s.denormalize(def_uniform(seed, name))
  end
end

-- Helmert ILR (isometric log-ratio): N log-values on the zero-sum hyperplane <-> N-1 orthonormal
-- coords. Used for gauge (scales) so the search drops the redundant uniform-scale direction (killed
-- by L2-norm) and works in a full-rank, geomean-1 space.
local function ilr_forward (y)
  local N = #y
  local z = {}
  for k = 1, N - 1 do
    local m = 0
    for i = 1, k do m = m + y[i] end
    z[k] = math.sqrt(k / (k + 1)) * (m / k - y[k + 1])
  end
  return z
end

local function ilr_inverse (z, N)
  local y = {}
  for i = 1, N do y[i] = 0 end
  for k = 1, N - 1 do
    local a = math.sqrt(1 / (k * (k + 1)))
    local zk = z[k]
    for i = 1, k do y[i] = y[i] + zk * a end
    y[k + 1] = y[k + 1] - zk * math.sqrt(k / (k + 1))
  end
  return y
end

local build_samplers = function (args, param_names, global_dev, seed)
  local samplers = {}
  for _, pname in ipairs(param_names) do
    local s = build_sampler(args[pname], global_dev)
    if seed ~= nil then seed_center(s, pname, seed) end
    samplers[pname] = s
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
  local n_hyper_restarts = args.n_hyper_restarts or 10
  local gp_cap = args.gp_obs_cap or 96
  local ptag = args.prof_tag or "?"
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
  local n_initial = num.min(math.max(4, 2 * math.ceil(math.log(n_dims + 1) / math.log(2))), trials)
  local gp_model = n_dims > 0 and make_gp_model(n_dims, gp_cap, n_hyper_restarts) or nil

  if args.reseed ~= false then
    local seed = 2166136261 % 2147483647
    for _, name in ipairs(param_names) do
      local s = samplers[name]
      if s and s.center ~= nil then
        local v = s.normalize and s.normalize(s.center) or (type(s.center) == "number" and s.center or 0)
        seed = (seed + num.floor(v * 2147483646)) % 2147483647
        seed = (seed * 48271) % 2147483647
      end
    end
    rand.fast_seed(seed)
  end

  local lhs_pts = n_dims > 0 and lhs_sample(n_initial, n_dims) or nil
  if lhs_pts then
    local def_pt = {}
    for i, name in ipairs(search_dims) do
      local s = samplers[name]
      def_pt[i] = s.normalize(s.center)
    end
    table.insert(lhs_pts, 1, def_pt)
    n_initial = n_initial + 1
  end
  local cand_flat = n_dims > 0 and dvec.create(n_candidates * n_dims) or nil
  local tr_L, tr_succ, tr_fail = 0.8, 0, 0
  local TR_LMAX, TR_LMIN, TR_SUCC, TR_FAIL = 1.6, 0.5 ^ 7, 3, math.max(4, n_dims)
  local have_success = false
  local held_fail = {}

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

    local explore = n_dims == 0 or t <= n_initial or not have_success
    if explore then
      params = {}
      local pt = lhs_pts and lhs_pts[t]
      if pt then
        for i, name in ipairs(search_dims) do
          params[name] = samplers[name].denormalize(pt[i])
        end
      end
      fill_rest(params)
    else
      local inc, half = {}, {}
      local gmean
      local ls_buf = gp_model.lengthscales()
      if ls_buf then
        local logsum = 0.0
        for j = 1, n_dims do logsum = logsum + num.log(ls_buf:get(j - 1)) end
        gmean = num.exp(logsum / n_dims)
      end
      for j, name in ipairs(search_dims) do
        inc[j] = samplers[name].normalize(best_params[name])
        local w = ls_buf and (0.5 * tr_L * (ls_buf:get(j - 1) / gmean)) or (0.5 * tr_L)
        half[j] = w < 1e-3 and 1e-3 or w
      end
      local pmask = num.min(1.0, 20.0 / n_dims)
      local rmax = rand.fast_max + 1
      local function box (j)
        local lo = inc[j] - half[j]; if lo < 0 then lo = 0 end
        local hi = inc[j] + half[j]; if hi > 1 then hi = 1 end
        return lo + (hi - lo) * (rand.fast_random() / rmax)
      end
      local t_cg = tick("cand_gen/" .. ptag)
      for i = 1, n_candidates do
        local p = {}
        local any = false
        for j, name in ipairs(search_dims) do
          local u = inc[j]
          if rand.fast_random() / rmax < pmask then any = true; u = box(j) end
          p[name] = samplers[name].denormalize(u)
        end
        if not any then
          local j = num.floor(rand.fast_random() / rmax * n_dims) + 1
          p[search_dims[j]] = samplers[search_dims[j]].denormalize(box(j))
        end
        if constrain_fn then constrain_fn(p) end
        local base = (i - 1) * n_dims
        for j, name in ipairs(search_dims) do
          cand_flat:set(base + j - 1, samplers[name].normalize(p[name]))
        end
      end
      tock(t_cg)
      local t_gp = tick("gp.suggest/" .. ptag)
      local ei_buf = gp_model.suggest(cand_flat, n_candidates)
      tock(t_gp)
      local _, best_c = ei_buf:max()
      params = {}
      for i, name in ipairs(search_dims) do
        params[name] = samplers[name].denormalize(cand_flat:get(best_c * n_dims + i - 1))
      end
      fill_rest(params)
    end

    if constrain_fn then constrain_fn(params) end

    local phase = explore and "lhs" or "gp"
    local score, metrics, result = trial_fn(params, {
      trial = t,
      trials = trials,
      is_final = false,
      global_best_score = best_score,
      phase = phase,
    })

    if n_dims > 0 then
      local failed = metrics and metrics.failed
      local row = {}
      for i, name in ipairs(search_dims) do
        row[i] = samplers[name].normalize(params[name])
      end
      if failed and not have_success then
        held_fail[#held_fail + 1] = row
      else
        if not failed and not have_success then
          have_success = true
          for _, hrow in ipairs(held_fail) do
            gp_model.observe(hrow, score - 1, 0)
          end
          held_fail = {}
        end
        local nz = (metrics and metrics.fold_var) or 0
        local sc = score < 0 and -score or score
        if sc < 1 then sc = 1 end
        local nf = (1e-3 * sc) ^ 2
        gp_model.observe(row, score, nz > nf and nz or nf)
      end
    end

    local new_best = score > best_score
    if new_best then
      best_score = score
      best_params = params
      best_result = result
      best_metrics = metrics
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
        is_new_best = new_best,
        phase = phase,
      })
    end
    if phase == "gp" then
      if new_best then tr_succ = tr_succ + 1; tr_fail = 0
      else tr_fail = tr_fail + 1; tr_succ = 0 end
      if tr_succ >= TR_SUCC then tr_L = num.min(tr_L * 2, TR_LMAX); tr_succ = 0 end
      if tr_fail >= TR_FAIL then
        tr_L = tr_L * 0.5; tr_fail = 0
        if tr_L < TR_LMIN then tr_L = 0.8 end
      end
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

local function default_trial_fn (args, dense, metric, k)
  local ridge = require("santoku.learn.ridge")
  local function mk_ridge (kd)
    return ridge.create({ gram = kd.gram, w_buf = args.ridge_w_buf })
  end
  if dense then
    local eval = require("santoku.learn.evaluator")
    return function (kd)
      local r = mk_ridge(kd)
      local tr = tick("regress"); local s = r:regress(kd.val_codes); tock(tr)
      local td = tick("decide"); local m = eval.regress_accuracy(s, args.val_targets); tock(td)
      return 1 - m.nmae, { nmae = m.nmae }
    end
  end
  local decide = require("santoku.learn.decide")
  local nl, vn = args.n_labels, args.val_n_samples
  if metric == "span" then
    local cand, gold = args.val_cand, args.val_gold
    local probe = decide.create({ n_labels = nl, span = true, reject = args.reject })
    return function (kd)
      local r = mk_ridge(kd)
      local tr = tick("regress"); local s = r:regress(kd.val_codes); tock(tr)
      local td = tick("decide")
      local f1 = probe:calibrate({ scores = s, n_samples = cand:offsets():size() - 1, cand = cand, gold = gold })
      tock(td)
      return f1, { span_f1 = f1, offset = probe:offset() }
    end
  end
  if metric == "single" then
    local probe = decide.create({ n_labels = nl, single = true })
    return function (kd)
      local r = mk_ridge(kd)
      local tr = tick("regress"); local s = r:regress(kd.val_codes); tock(tr)
      local td = tick("decide"); local _, m = probe:score({ scores = s, n_samples = vn, expected = args.val_y }); tock(td)
      return m.accuracy, { macro_f1 = m.macro_f1, accuracy = m.accuracy }
    end
  end
  local probe = decide.create({ n_labels = nl })
  return function (kd)
    local r = mk_ridge(kd)
    local tr = tick("regress"); local P = r:label(kd.val_codes, k); tock(tr)
    local td = tick("decide")
    local f1, p, rc = probe:calibrate({ pred = P, n_samples = vn, expected = args.val_y })
    tock(td)
    return f1, { f1 = f1, precision = p, recall = rc, offset = probe:offset() }
  end
end

local function decode_mode (args, dense)
  if dense then return false, nil end
  if args.cand or (args.fold_val_cand and args.fold_val_cand[1]) then return true, "span" end
  local y = args.y
  local n = args.n_samples
  if (args.n_labels or 0) > 1 and y then
    local eo = y:offsets()
    for i = 0, n - 1 do
      if eo:get(i + 1) - eo:get(i) ~= 1 then return true, "multilabel" end
    end
    return true, "single"
  end
  return true, "multilabel"
end

local REBUILD_KNOBS = {
  { key = "scales", defaults = { min = 0.01, max = 100, log = true }, gauge = true },
  { key = "exponent", defaults = { min = 0, max = 8 } },
}

M.krr = function (args)
  local spectral = require("santoku.learn.spectral")
  local ridge = require("santoku.learn.ridge")
  local fvec = require("santoku.fvec")
  err.assert(args.n_landmarks, "n_landmarks required")
  tt = args.verbose and utc.ticktock() or nil
  local function prof_emit ()
    if tt and args.each then
      local stats, total = tt()
      args.each({ event = "profile", stats = stats, total = total })
    end
    tt = nil
  end
  if not args.rebuild then
    if args.pool_blocks then args = require("santoku.learn.util").fold_blocks(args)
    elseif args.pool_codes then args = require("santoku.learn.util").fold_dense(args) end
  end
  local function resolve_knob (spec)
    if spec == SEARCH then return nil end
    if type(spec) ~= "table" then return spec end
    if spec[1] ~= nil then
      local v = {}
      for i = 1, veclen(spec) do
        local e = spec[i]
        if e == nil or e == SEARCH then v[i] = false
        elseif type(e) == "table" then v[i] = e.def or e.max or e.min
        else v[i] = e end
      end
      return v
    end
    if type(spec.def) == "table" then
      local v = {}
      for i = 1, veclen(spec.def) do
        local d = spec.def[i]
        if d == nil or d == SEARCH then v[i] = false else v[i] = d end
      end
      return v
    end
    return spec.def or spec.max or spec.min
  end
  local function resolve_params ()
    local p, any = {}, false
    for _, kdef in ipairs(REBUILD_KNOBS) do
      if args[kdef.key] ~= nil then p[kdef.key] = resolve_knob(args[kdef.key]); any = true end
    end
    if not any then return nil end
    return p
  end
  if args.rebuild and args.x == nil then
    local rb = args.rebuild(resolve_params())
    args.x = rb.x
    args.blocks = rb.blocks
    if rb.blocks then
      args.n_samples = args.n_samples or rb.n_samples
    end
  end
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
  local dense = args.pool_targets ~= nil
  local kernel_spec = args.kernel or "cosine"
  local kernels = type(kernel_spec) == "table" and kernel_spec or { kernel_spec }
  args.kernel = kernels
  local families = {}
  for _, kn in ipairs(kernels) do
    if kn == "cosine" then families.cosine = true
    elseif kn == "arccos" then families.arccos = true
    else families.matern = true end
  end
  local function cat_spec (v, deflist)
    if v == nil or v == SEARCH then return deflist end
    if type(v) ~= "table" then return v end
    if #v > 0 then return v end
    local s = {}
    for i = 1, #deflist do s[i] = deflist[i] end
    s.def = v.def
    return s
  end
  args.gamma = spec_defaults(args.gamma, { min = 1e-2, max = 16, log = true })
  args.nu = cat_spec(args.nu, { 3, 0, 1, 2 })
  args.order = cat_spec(args.order, { 1, 2, 3, 4 })
  args.depth = cat_spec(args.depth, { 1, 2 })
  args.tangent = cat_spec(args.tangent, { 0, 1 })
  local seed = args.seed or 4
  local kernel_samplers = build_samplers(args, { "nu", "gamma", "order", "depth", "tangent" }, nil, seed)
  local do_search = args.search_trials and args.search_trials > 0
  local lcb_kappa = args.lcb_kappa or 1
  local decode_offset = args.decode_offset
  if decode_offset == SEARCH then decode_offset = nil
  elseif type(decode_offset) == "table" then decode_offset = (not do_search) and decode_offset.def or nil end
  args.lambda = spec_defaults(args.lambda, { min = 1e-7, max = 8, log = true })
  local label_names = { "lambda" }
  if args.extra then
    for _, e in ipairs(args.extra) do
      label_names[#label_names + 1] = e.name
      args[e.name] = spec_defaults({ min = e.min, max = e.max, log = e.log, def = e.def },
        { min = 0, max = 1 })
    end
  end
  local label_samplers = build_samplers(args, label_names, nil, seed)
  local k = not dense and (args.k or 32) or nil
  local tiled = not dense
  local tile_labels = tiled and (args.tile_labels or 1024) or nil
  local want_decode, mode = decode_mode(args, dense)
  local spectral_args = {
    x = args.x, y = args.y,
    blocks = args.blocks,
    n_tokens = args.n_tokens, n_samples = args.n_samples,
    d_input = args.d_input,
    n_labels = args.n_labels,
    targets = args.targets, n_targets = args.n_targets,
  }
  local w_auto
  local cv_folds = (args.folds or 1) > 1
  local nl_cap = args.n_labels or args.n_targets or 1
  local xtx_shared = args.xtx_buf or fvec.create(args.n_landmarks * args.n_landmarks)
  local xty_shared = args.xty_buf or fvec.create(args.n_landmarks * nl_cap)
  spectral_args.xtx_buf = xtx_shared
  spectral_args.xty_buf = xty_shared
  local w_shared = args.spectral_w_buf or fvec.create(args.n_landmarks * nl_cap)
  spectral_args.w_buf = w_shared
  if tiled then
    spectral_args.tile_labels = tile_labels
    w_auto = args.w_buf or fvec.create(args.n_landmarks * (args.n_labels or 1))
  end
  local lm_bufs, lm_bufs_created
  if math.max(args.landmark_rounds or 1, args.search_landmark_rounds or 1) > 1 then
    lm_bufs = args.landmark_bufs
    if not lm_bufs then
      local pn = args.n_samples or args.pool_n
      err.assert(pn, "krr: landmark rounds > 1 requires n_samples")
      local kn = args.n_landmarks * pn
      local fn = args.n_landmarks * args.n_landmarks
      if args.landmark_buf_path then
        local pfx = args.landmark_buf_path
        lm_bufs = {
          k = fvec.mmap_create(pfx .. ".lmk.bin", kn),
          y = fvec.mmap_create(pfx .. ".lmy.bin", kn),
          fac = fvec.mmap_create(pfx .. ".lmfac.bin", fn),
        }
      else
        lm_bufs = { k = fvec.create(kn), y = fvec.create(kn), fac = fvec.create(fn) }
      end
      lm_bufs_created = true
    end
    spectral_args.lmk_buf = lm_bufs.k
    spectral_args.lmy_buf = lm_bufs.y
    spectral_args.lmfac_buf = lm_bufs.fac
  end
  local proj_shared, sims_shared, row_shared
  if do_search then
    args.ridge_w_buf = args.ridge_w_buf or fvec.create(args.n_landmarks * (args.n_labels or 1))
    proj_shared = fvec.create(args.n_landmarks * args.n_landmarks)
    sims_shared = fvec.create(4096 * args.n_landmarks)
    row_shared = fvec.create(128 * args.n_landmarks)
    spectral_args.proj_buf = proj_shared
    spectral_args.sims_buf = sims_shared
    spectral_args.row_buf = row_shared
  end
  local val_out_shared, val_out_reuse = {}, true
  local fixed_lms = do_search
  local search_m = args.search_landmarks or args.n_landmarks
  local lm_slots = {}
  local xtx_slots, xty_slots = {}, {}
  local function release_enc_scratch ()
    spectral_args.proj_buf = args.enc_chol_buf
    spectral_args.sims_buf = nil; spectral_args.row_buf = nil
    proj_shared = nil; sims_shared = nil; row_shared = nil -- luacheck: ignore
    val_out_shared = nil; val_out_reuse = false
    fixed_lms = false; lm_slots = {}
    xtx_slots = {}; xty_slots = {}
    spectral_args.landmarks = nil
    spectral_args.xtx_buf = xtx_shared
    spectral_args.xty_buf = xty_shared
    collectgarbage("collect")
  end
  local function release_cv ()
    if not args.xtx_buf then xtx_shared:destroy() end
    if not args.xty_buf then xty_shared:destroy() end
    if lm_bufs_created then
      lm_bufs.k:destroy(); lm_bufs.y:destroy(); lm_bufs.fac:destroy()
    end
  end
  local cur_val_blocks, cur_val_x = nil, nil
  local function val_encode (sp_enc, slot)
    if not (cur_val_blocks or cur_val_x) then return nil end
    local out = val_out_reuse and val_out_shared[slot] or nil
    local vc = cur_val_blocks and sp_enc:encode({ blocks = cur_val_blocks }, out)
      or sp_enc:encode(cur_val_x, out)
    if val_out_reuse then val_out_shared[slot] = vc end
    return vc
  end
  local function build_kd (spec, solve, fold)
    local tgc = tick("gc"); collectgarbage("collect"); tock(tgc)
    if args.rebuild and spec.params ~= nil then
      local trb = tick("rebuild"); local rb = args.rebuild(spec.params, fold); tock(trb)
      spectral_args.x = rb.x
      spectral_args.blocks = rb.blocks
      cur_val_blocks = rb.val_blocks
      cur_val_x = rb.val_x
      spectral_args.colscale = rb.colscale
    end
    spectral_args.y = (fold and args.fold_y) and args.fold_y[fold] or args.y
    spectral_args.targets = (fold and args.fold_targets) and args.fold_targets[fold] or args.targets
    spectral_args.kernel = spec.kernel
    spectral_args.gamma = spec.gamma
    spectral_args.nu = spec.nu
    spectral_args.order = spec.order
    spectral_args.depth = spec.depth
    spectral_args.tangent = spec.tangent
    spectral_args.solve_lambda = solve and solve.lambda or nil
    spectral_args.strata = (fold and args.fold_strata) and args.fold_strata[fold] or args.pool_strata
    if fixed_lms then
      local slot = fold or 0
      local rounds = args.search_landmark_rounds or 1
      if rounds > 1 then
        spectral_args.landmarks = nil
        spectral_args.n_landmarks = search_m
        spectral_args.landmark_rounds = rounds
        spectral_args.landmark_seed = seed + 1000 * (slot + 1)
      else
        if lm_slots[slot] == nil then
          lm_slots[slot] = spectral.uniform_landmarks(spectral_args, search_m, seed + 1000 * (slot + 1))
        end
        spectral_args.landmarks = lm_slots[slot]
        spectral_args.landmark_rounds = nil
      end
      if xtx_slots[slot] == nil then
        xtx_slots[slot] = fvec.create(search_m * search_m)
        xty_slots[slot] = fvec.create(search_m * nl_cap)
      end
      spectral_args.xtx_buf = xtx_slots[slot]
      spectral_args.xty_buf = xty_slots[slot]
    else
      local rounds = args.landmark_rounds or 1
      if rounds > 1 then
        spectral_args.landmarks = nil
        spectral_args.n_landmarks = args.n_landmarks
        spectral_args.landmark_rounds = rounds
        spectral_args.landmark_seed = seed + 1000 * ((fold or 0) + 1)
      else
        -- final: random (uniform) landmarks at full n_landmarks (matches the random-trial regime)
        spectral_args.landmarks = spectral.uniform_landmarks(spectral_args, args.n_landmarks, seed + 1000 * ((fold or 0) + 1))
        spectral_args.landmark_rounds = nil
      end
    end
    local tse = tick("spectral.encode"); local _, sp_enc, gram = spectral.encode(spectral_args); tock(tse)
    if tt and spectral_args.enc_phases then
      for name, dt in pairs(spectral_args.enc_phases) do prof_add_raw("~spectral/" .. name, dt) end
      spectral_args.enc_phases = nil
    end
    local tve = tick("val_encode"); local val_codes = val_encode(sp_enc, fold or 0); tock(tve)
    return { sp_enc = sp_enc, gram = gram, val_codes = val_codes }
  end
  local function oof_applicable ()
    if decode_offset ~= nil then return false end
    if not (want_decode and cv_folds) then return false end
    if mode == "span" then return args.fold_val_cand and args.fold_val_cand[1] ~= nil end
    if mode == "multilabel" then return args.fold_val_y and args.fold_val_y[1] ~= nil end
    return false
  end
  local function oof_decider (params, kds)
    local ivec = require("santoku.ivec")
    local nf = args.folds or 1
    local spec = not kds and { kernel = params.kernel, gamma = params.gamma, nu = params.nu,
      order = params.order, depth = params.depth, tangent = params.tangent,
      params = params.params or params } or nil
    local function fold_kd (f) return kds and kds[f] or build_kd(spec, params, f) end
    if mode == "span" then
      local spans = require("santoku.spans")
      local pooled_s = fvec.create()
      local pooled_cand, pooled_gold
      local fold_s = {}
      local function clone_spans (S)
        local o, s, e, t = S:offsets(), S:col("s"), S:col("e"), S:col("ty")
        local no, ns, ne, nt = ivec.create(), ivec.create(), ivec.create(), ivec.create()
        no:copy(o); ns:copy(s); ne:copy(e); nt:copy(t)
        return spans.create({ offsets = no, s = ns, e = ne, ty = nt })
      end
      for f = 1, nf do
        local kd = fold_kd(f)
        local r = ridge.create({ gram = kd.gram })
        local tr = tick("regress")
        local s = r:regress(kd.val_codes)
        tock(tr)
        fold_s[f] = s
        pooled_s:copy(s)
        local fc, fg = args.fold_val_cand[f], args.fold_val_gold[f]
        if not pooled_cand then pooled_cand, pooled_gold = clone_spans(fc), clone_spans(fg)
        else pooled_cand:append(clone_spans(fc)); pooled_gold:append(clone_spans(fg)) end
      end
      local decider, m = M.decide({ n_labels = args.n_labels, reject = args.reject, val_scores = pooled_s,
        val_n_samples = pooled_cand:offsets():size() - 1, val_cand = pooled_cand, val_gold = pooled_gold })
      local scs, fms
      if kds then
        scs, fms = {}, {}
        for f = 1, nf do
          local f1, fm = decider:score({ scores = fold_s[f],
            n_samples = args.fold_val_cand[f]:offsets():size() - 1,
            cand = args.fold_val_cand[f], gold = args.fold_val_gold[f] })
          scs[f] = f1; fms[f] = fm
        end
      end
      return decider, m, scs, fms
    end
    local fold_P = {}
    local P, Y
    for f = 1, nf do
      local kd = fold_kd(f)
      local r = ridge.create({ gram = kd.gram })
      local tr = tick("regress")
      local Pf = r:label(kd.val_codes, k)
      tock(tr)
      fold_P[f] = Pf
      local Yf = args.fold_val_y[f]
      if f == 1 then
        P, Y = Pf:clone(), Yf:clone()
      else
        P:append(Pf); Y:append(Yf)
      end
    end
    local decider, m = M.decide({ n_labels = args.n_labels, val_pred = P,
      val_n_samples = select(1, Y:shape()), val_expected = Y })
    local scs, fms
    if kds then
      scs, fms = {}, {}
      for f = 1, nf do
        local f1, fm = decider:score({ pred = fold_P[f], n_samples = args.fold_val_n[f],
          expected = args.fold_val_y[f] })
        scs[f] = f1; fms[f] = fm
      end
    end
    return decider, m, scs, fms
  end
  local function finish (kd, params, solve)
    local r = ridge.create({
      gram = kd.gram,
      w_buf = tiled and w_auto or nil,
    })
    if args.each then args.each({ event = "done", params = params, emb_d = kd.sp_enc:dims(),
      solve = solve }) end
    local decider, decider_metrics
    if want_decode then
      if decode_offset ~= nil and mode ~= "single" then
        decider = require("santoku.learn.decide").create({ n_labels = args.n_labels,
          span = mode == "span", reject = args.reject, offset = decode_offset })
      elseif oof_applicable() then
        decider, decider_metrics = oof_decider(params)
      elseif mode == "single" then
        decider = require("santoku.learn.decide").create({ n_labels = args.n_labels, single = true })
      end
    end
    local codes_or_deploy, bake_blocks
    if args.bake_external then
      bake_blocks = function (ext) return args.bake_external(ext, params) end
      codes_or_deploy = function (ext, out)
        return kd.sp_enc:encode({ blocks = bake_blocks(ext) }, out)
      end
    else
      codes_or_deploy = function (x, out) return kd.sp_enc:encode(x, out) end
    end
    return kd.sp_enc, r, codes_or_deploy, params, decider, decider_metrics, bake_blocks
  end
  local function center_spec ()
    local kname = kernels.def or kernels[1]
    local base = { kernel = kname }
    if kname == "matern" then
      base.nu = kernel_samplers.nu.center
      base.gamma = kernel_samplers.gamma.center
    elseif kname == "arccos" then
      base.order = kernel_samplers.order.center
      base.depth = kernel_samplers.depth.center
      base.tangent = kernel_samplers.tangent.center
    end
    return kname, base
  end
  local function spec_with (base, p)
    local spec = { params = p }
    for kk, vv in pairs(base) do spec[kk] = vv end
    return spec
  end
  if not do_search then
    local _, base = center_spec()
    local rparams = resolve_params()
    local spec = spec_with(base, rparams)
    local lp = sample_params(label_samplers, label_names, nil, true)
    release_enc_scratch()
    local kd = build_kd(spec, lp)
    local params = {}
    for kk, vv in pairs(base) do params[kk] = vv end
    if rparams then for kk, vv in pairs(rparams) do params[kk] = vv end end
    for _, n in ipairs(label_names) do params[n] = lp[n] end
    local sp_enc, r, vcodes, params_out, decider, dmetrics, bake = finish(kd, params, "cholesky")
    release_cv()
    prof_emit()
    return sp_enc, r, vcodes, params_out, decider, dmetrics, bake
  end
  local nfolds = args.folds or 1
  err.assert(not do_search or nfolds > 1, "krr: search requires folds > 1 (all-CV; no external dev set)")
  local fold_trial_fns
  if nfolds > 1 then
    fold_trial_fns = {}
    for f = 1, nfolds do
      fold_trial_fns[f] = default_trial_fn({
        val_y = args.fold_val_y and args.fold_val_y[f] or nil,
        val_cand = args.fold_val_cand and args.fold_val_cand[f] or nil,
        val_gold = args.fold_val_gold and args.fold_val_gold[f] or nil,
        reject = args.reject,
        val_n_samples = args.fold_val_n[f],
        val_targets = args.fold_val_targets and args.fold_val_targets[f] or nil,
        n_labels = args.n_labels,
        ridge_w_buf = args.ridge_w_buf,
      }, dense, mode == "multilabel" and "fmeasure" or mode, k)
    end
  end
  local function run_folds (spec, lam)
    local kds = {}
    for f = 1, nfolds do
      kds[f] = build_kd(spec, nil, nfolds == 1 and nil or f)
    end
    local meta = { dims = kds[1].sp_enc:dims() }
    local ts = tick("~fold_solve")
    local mean, agg, scs = 0.0, {}, {}
    if decode_offset == nil and nfolds > 1 and (mode == "span" or mode == "multilabel") then
      for f = 1, nfolds do kds[f].gram:solve(lam) end
      local decider, m, foldsc, foldms = oof_decider(nil, kds)
      for f = 1, nfolds do scs[f] = foldsc[f]; mean = mean + scs[f] end
      mean = mean / nfolds
      if foldms then
        for f = 1, nfolds do
          local fm = foldms[f]
          if fm then for kk, vv in pairs(fm) do if type(vv) == "number" then agg[kk] = (agg[kk] or 0) + vv end end end
        end
        for kk, vv in pairs(agg) do agg[kk] = vv / nfolds end
      else
        agg = m
      end
      agg.offset = decider:offset()
    else
      for f = 1, #kds do
        local kd = kds[f]
        kd.gram:solve(lam)
        local sc, m = fold_trial_fns[f](kd)
        scs[f] = sc; mean = mean + sc
        if m then for kk, vv in pairs(m) do if type(vv) == "number" then agg[kk] = (agg[kk] or 0) + vv end end end
      end
      mean = mean / #kds
      for kk, vv in pairs(agg) do agg[kk] = vv / #kds end
    end
    if nfolds > 1 then
      local ss = 0.0
      for f = 1, nfolds do local dd = scs[f] - mean; ss = ss + dd * dd end
      agg.fold_var = ss / (nfolds - 1) / nfolds
    end
    tock(ts)
    return mean, agg, meta
  end
  local matern_trials = args.matern_trials or args.search_trials or 0
  local arccos_trials = args.arccos_trials or args.search_trials or 0
  local km_trials = (families.matern and matern_trials)
    or (families.cosine and (args.search_trials or 30) or 0)
  local btot = (km_trials or 0)
    + (families.arccos and arccos_trials or 0)
  local n_blocks = (args.pool_blocks and #args.pool_blocks) or 1
  local rebuild_knobs = {}
  for _, kdef in ipairs(REBUILD_KNOBS) do
    local spec = args[kdef.key]
    if spec ~= nil then
      local knob = { key = kdef.key, gauge = kdef.gauge, names = {}, samplers = {} }
      if spec == SEARCH then
        local dv = {}
        for i = 1, n_blocks do dv[i] = SEARCH end
        spec = { def = dv }
        args[kdef.key] = spec
      end
      if type(spec) == "table" and spec[1] == nil and type(spec.def) == "table" then
        local vec = {}
        for i = 1, veclen(spec.def) do
          local d = spec.def[i]
          if d == nil or d == false then
            vec[i] = false
          elseif d == SEARCH then
            vec[i] = { min = spec.min, max = spec.max, log = spec.log, pow2 = spec.pow2 }
          else
            vec[i] = { min = spec.min, max = spec.max, log = spec.log, pow2 = spec.pow2, def = d }
          end
        end
        spec = vec
        args[kdef.key] = vec
      end
      if type(spec) == "table" and spec[1] ~= nil and kdef.gauge then
        knob.kind = "gauge"
        knob.n = #spec
        knob.active = {}
        local defs, all_def = {}, true
        for i = 1, #spec do
          if spec[i] ~= false then
            knob.active[#knob.active + 1] = i
            local sd = spec_defaults(spec[i], kdef.defaults)
            local d = (type(sd) == "table" and sd.def) or (type(sd) == "number" and sd) or nil
            defs[#knob.active] = d
            if type(d) ~= "number" then all_def = false end
          end
        end
        local M = #knob.active
        knob.box_min = kdef.defaults.min
        knob.box_max = kdef.defaults.max
        -- box that matches the original independent per-scale log-uniform spread (Var(log s)=L^2/3),
        -- so the gauge-free prior isn't wider than what the per-block box implied.
        local c = math.log(kdef.defaults.max or 100) / math.sqrt(1 - 1 / math.max(M, 2))
        local centers
        if all_def and M > 1 then
          local y = {}
          for j = 1, M do y[j] = math.log(defs[j]) end
          centers = ilr_forward(y)
        end
        for k = 1, M - 1 do
          local nm = kdef.key .. "_z" .. k
          args[nm] = { min = -c, max = c, def = centers and centers[k] }
          knob.names[#knob.names + 1] = nm
          knob.samplers[nm] = build_samplers(args, { nm }, nil, seed)[nm]
        end
      elseif type(spec) == "table" and spec[1] ~= nil then
        knob.kind = "vector"
        knob.layout = {}
        local sds = {}
        for i = 1, #spec do
          if spec[i] ~= false then sds[i] = spec_defaults(spec[i], kdef.defaults) end
        end
        for i = 1, #spec do
          if spec[i] == false then
            knob.layout[i] = false
          else
            local nm = kdef.key .. i
            args[nm] = sds[i]
            knob.layout[i] = nm
            knob.names[#knob.names + 1] = nm
            knob.samplers[nm] = build_samplers(args, { nm }, nil, seed)[nm]
          end
        end
      else
        knob.kind = "scalar"
        args[kdef.key] = spec_defaults(spec, kdef.defaults)
        knob.names[1] = kdef.key
        knob.samplers[kdef.key] = build_samplers(args, { kdef.key }, nil, seed)[kdef.key]
      end
      rebuild_knobs[#rebuild_knobs + 1] = knob
    end
  end
  local has_knobs = #rebuild_knobs > 0
  local function knob_value (knob, gp)
    if knob.kind == "scalar" then return gp[knob.key] end
    if knob.kind == "gauge" then
      local v = {}
      for i = 1, knob.n do v[i] = false end
      local M = #knob.active
      if M == 1 then
        v[knob.active[1]] = 1.0
      elseif M > 1 then
        local z = {}
        for k = 1, M - 1 do z[k] = gp[knob.names[k]] end
        local y = ilr_inverse(z, M)
        for j = 1, M do
          local s = math.exp(y[j])
          if s < knob.box_min then s = knob.box_min elseif s > knob.box_max then s = knob.box_max end
          v[knob.active[j]] = s
        end
      end
      return v
    end
    local v = {}
    for i = 1, #knob.layout do
      local nm = knob.layout[i]
      if nm then v[i] = gp[nm] else v[i] = false end
    end
    return v
  end
  local function params_of (gp)
    local p = {}
    for _, knob in ipairs(rebuild_knobs) do p[knob.key] = knob_value(knob, gp) end
    for _, knob in ipairs(rebuild_knobs) do
      if knob.gauge then
        local v = p[knob.key]
        local logsum, cnt = 0, 0
        for i = 1, #v do
          if type(v[i]) == "number" and v[i] > 0 then logsum = logsum + math.log(v[i]); cnt = cnt + 1 end
        end
        if cnt > 0 then
          local gm = math.exp(logsum / cnt)
          for i = 1, #v do if type(v[i]) == "number" then v[i] = v[i] / gm end end
          if knob.bmin and knob.bmax and knob.bmin > 0 then
            local lo, hi
            for i = 1, #v do
              local x = v[i]
              if type(x) == "number" and x > 0 then
                lo = (lo == nil or x < lo) and x or lo
                hi = (hi == nil or x > hi) and x or hi
              end
            end
            if hi and (hi > knob.bmax or lo < knob.bmin) then
              local c = math.sqrt((hi * lo) / (knob.bmax * knob.bmin))
              if hi / c <= knob.bmax and lo / c >= knob.bmin then
                for i = 1, #v do if type(v[i]) == "number" then v[i] = v[i] / c end end
              end
            end
          end
        end
      end
    end
    return p
  end
  local function base_with (base, p)
    base.params = p
    for kk, vv in pairs(p) do base[kk] = vv end
    return base
  end
  local function params_sig (p)
    local parts = {}
    for _, knob in ipairs(rebuild_knobs) do
      local v = p[knob.key]
      if type(v) == "table" then
        local s = {}
        for i = 1, #v do s[i] = v[i] == false and "-" or tostring(v[i]) end
        parts[#parts + 1] = table.concat(s, ",")
      else
        parts[#parts + 1] = tostring(v)
      end
    end
    return table.concat(parts, ";")
  end
  local function with_knobs (names)
    local out = {}
    for i = 1, #names do out[i] = names[i] end
    if has_knobs then
      for _, knob in ipairs(rebuild_knobs) do
        for _, nm in ipairs(knob.names) do out[#out + 1] = nm end
      end
    end
    for _, nm in ipairs(label_names) do out[#out + 1] = nm end
    return out
  end
  local function merge_knob_samplers (base)
    for _, knob in ipairs(rebuild_knobs) do
      for nm, s in pairs(knob.samplers) do base[nm] = s end
    end
    for nm, s in pairs(label_samplers) do base[nm] = s end
    return base
  end
  local best_params, best_lcb = nil, -num.huge
  local worst_score = nil
  local bi = 0
  local function eval_kd (spec, base, gp)
    local t0 = utc.time(true)
    local ib = {}
    for kk, vv in pairs(base) do ib[kk] = vv end
    for _, n in ipairs(label_names) do ib[n] = gp[n] end
    local ok, sc, sm, meta = pcall(run_folds, spec, gp.lambda)
    if not ok or sc == nil then
      sc, sm, meta = (worst_score or -1e18), { failed = true }, meta or {}
    else
      worst_score = worst_score and num.min(worst_score, sc) or sc
    end
    local lcb = sc - lcb_kappa * math.sqrt((sm and sm.fold_var) or 0)
    local dt = utc.time(true) - t0
    bi = bi + 1
    local improved = lcb > best_lcb
    if improved then best_lcb = lcb; best_params = ib end
    if args.each then
      args.each({ event = "trial", phase = "kernel", trial = bi, trials = btot,
        params = ib, score = sc, metrics = sm, emb_d = meta.dims, gpbo = dt,
        best = best_lcb, is_new_best = improved })
    end
    return sc, sm
  end
  local t_ks = tick("~kernel_search")
  local fam_runs = {}
  if families.matern and families.cosine then
    local kfam_vals, kseen = {}, {}
    for _, kn in ipairs(kernels) do
      local fam
      if kn == "cosine" then fam = "cosine"
      elseif kn ~= "arccos" then fam = "matern" end
      if fam and not kseen[fam] then kseen[fam] = true; kfam_vals[#kfam_vals + 1] = fam end
    end
    args.kfam = kfam_vals
    fam_runs[#fam_runs + 1] = {
      names = { "kfam", "nu", "gamma" },
      samplers = { kfam = build_samplers(args, { "kfam" }, nil, seed).kfam,
        nu = kernel_samplers.nu, gamma = kernel_samplers.gamma },
      trials = matern_trials, tag = "kernel",
      base_of = function (gp)
        if gp.kfam == "cosine" then return { kernel = "cosine" } end
        return { kernel = "matern", nu = gp.nu, gamma = gp.gamma }
      end,
    }
  elseif families.matern then
    fam_runs[#fam_runs + 1] = {
      names = { "nu", "gamma" },
      samplers = { nu = kernel_samplers.nu, gamma = kernel_samplers.gamma },
      trials = matern_trials, tag = "kernel",
      base_of = function (gp) return { kernel = "matern", nu = gp.nu, gamma = gp.gamma } end,
    }
  elseif families.cosine then
    fam_runs[#fam_runs + 1] = {
      names = {},
      samplers = {},
      trials = has_knobs and args.search_trials or (args.search_trials or 30), tag = "kernel",
      base_of = function () return { kernel = "cosine" } end,
    }
  end
  if families.arccos then
    fam_runs[#fam_runs + 1] = {
      names = { "order", "depth", "tangent" },
      samplers = { order = kernel_samplers.order, depth = kernel_samplers.depth,
        tangent = kernel_samplers.tangent },
      trials = arccos_trials, tag = "arccos", memo = true,
      base_of = function (gp)
        return { kernel = "arccos", order = gp.order, depth = gp.depth, tangent = gp.tangent }
      end,
    }
  end
  for _, run in ipairs(fam_runs) do
    local seen = run.memo and {} or nil
    search({
      param_names = with_knobs(run.names), samplers = merge_knob_samplers(run.samplers),
      trials = run.trials, skip_final = true, prof_tag = run.tag,
      trial_fn = function (gp)
        local p = params_of(gp)
        local base = run.base_of(gp)
        local spec = spec_with(base, p)
        if not seen then
          return eval_kd(spec, base_with(base, p), gp)
        end
        local sig = tostring(base.order) .. ":" .. tostring(base.depth) .. ":"
          .. tostring(base.tangent) .. ":" .. params_sig(p)
        for _, n in ipairs(label_names) do sig = sig .. ":" .. tostring(gp[n]) end
        local hit = seen[sig]
        if hit then return hit[1], hit[2] end
        local sc, sm = eval_kd(spec, base_with(base, p), gp)
        seen[sig] = { sc, sm }
        return sc, sm
      end,
    })
  end
  tock(t_ks)
  if not best_params then
    local _, base = center_spec()
    local gp = {}
    for _, n in ipairs(label_names) do local s = label_samplers[n]; gp[n] = s and s.center end
    local rk = {}
    for _, knob in ipairs(rebuild_knobs) do
      for _, nm in ipairs(knob.names) do local s = knob.samplers[nm]; rk[nm] = s and s.center end
    end
    local p = params_of(rk)
    eval_kd(spec_with(base, p), base_with(base, p), gp)
  end
  local t_fin = tick("~finalize")
  release_enc_scratch()
  local best_kd = build_kd(best_params, best_params)
  local sp_enc, r, vcodes, params_out, decider, dmetrics, bake = finish(best_kd, best_params, "cholesky")
  tock(t_fin)
  release_cv()
  prof_emit()
  return sp_enc, r, vcodes, params_out, decider, dmetrics, bake
end

M.decide = function (args)
  local decide = require("santoku.learn.decide")
  if args.val_cand ~= nil then
    local g = decide.create({ n_labels = args.n_labels, span = true, reject = args.reject })
    local td = tick("decide")
    local f1, precision, recall = g:calibrate({
      scores = args.val_scores, n_samples = args.val_n_samples,
      cand = args.val_cand, gold = args.val_gold,
    })
    tock(td)
    return g, { span_f1 = f1, precision = precision, recall = recall, f1 = f1 }
  end
  local single = args.val_pred == nil
  local g = decide.create({ n_labels = args.n_labels, single = single })
  if single then
    local td = tick("decide")
    local macro_f1, accuracy = g:calibrate({
      scores = args.val_scores,
      n_samples = args.val_n_samples,
      expected = args.val_expected,
    })
    tock(td)
    return g, { macro_f1 = macro_f1, accuracy = accuracy }
  end
  local td = tick("decide")
  local best_f1, precision, recall = g:calibrate({
    pred = args.val_pred,
    n_samples = args.val_n_samples,
    expected = args.val_expected,
  })
  tock(td)
  return g, { f1 = best_f1, precision = precision, recall = recall }
end

return M

local num = require("santoku.num")
local err = require("santoku.error")
local capi = require("santoku.learn.optimize.capi")

local M = {}

local function fold_std (scores, mean, nf)
  if nf < 2 then return 0.0 end
  local s2 = 0.0
  for f = 1, nf do local d = scores[f] - mean; s2 = s2 + d * d end
  return num.sqrt(s2 / (nf - 1))
end

local function spec_defaults (spec, defs)
  if spec == nil then return defs end
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

local build_sampler = function (spec)
  if spec == nil then
    return nil
  end
  if type(spec) == "number" or type(spec) == "boolean" or type(spec) == "string" then
    return { type = "fixed", center = spec }
  end
  if type(spec) == "table" and spec.min ~= nil and spec.max ~= nil then
    local minv, maxv = spec.min, spec.max
    local is_log = not not spec.log
    local log_smin = is_log and num.log(minv) or 0
    local log_span = is_log and (num.log(maxv) - num.log(minv)) or 0
    local lin_span = maxv - minv
    return {
      type = "range",
      center = spec.def,
      normalize = function (x)
        if lin_span == 0 then return 0.5 end
        if is_log then
          return (num.log(x) - log_smin) / log_span
        else
          return (x - minv) / lin_span
        end
      end,
      denormalize = function (u)
        local x
        if is_log then
          x = num.exp(u * log_span + log_smin)
        else
          x = u * lin_span + minv
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

local function ilr_forward (y)
  local N = #y
  local z = {}
  for k = 1, N - 1 do
    local m = 0
    for i = 1, k do m = m + y[i] end
    z[k] = num.sqrt(k / (k + 1)) * (m / k - y[k + 1])
  end
  return z
end

local function ilr_inverse (z, N)
  local y = {}
  for i = 1, N do y[i] = 0 end
  for k = 1, N - 1 do
    local a = num.sqrt(1 / (k * (k + 1)))
    local zk = z[k]
    for i = 1, k do y[i] = y[i] + zk * a end
    y[k + 1] = y[k + 1] - zk * num.sqrt(k / (k + 1))
  end
  return y
end

local build_samplers = function (args, param_names, seed)
  local samplers = {}
  for _, pname in ipairs(param_names) do
    local s = build_sampler(args[pname])
    seed_center(s, pname, seed)
    samplers[pname] = s
  end
  return samplers
end

local center_params = function (samplers, param_names)
  local p = {}
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s then p[name] = s.center end
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

local cmaes_search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local trials = args.trials or 120
  local best_score = -num.huge
  local best_params = nil

  if all_fixed(samplers) or trials <= 0 then
    return center_params(samplers, param_names)
  end

  local search_dims = {}
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s and s.type == "range" then
      search_dims[#search_dims + 1] = name
    end
  end
  local n = #search_dims

  do
    local seed = 2166136261 % 2147483647
    for _, name in ipairs(param_names) do
      local s = samplers[name]
      if s and s.center ~= nil then
        local v = s.normalize and s.normalize(s.center) or (type(s.center) == "number" and s.center or 0)
        seed = (seed + num.floor(v * 2147483646)) % 2147483647
        seed = (seed * 48271) % 2147483647
      end
    end
    capi.seed(seed)
  end

  local function fill_rest (params)
    for _, name in ipairs(param_names) do
      local s = samplers[name]
      if s and s.type == "fixed" then params[name] = s.center end
    end
  end

  local function uniform () return capi.uniform() end

  local def_pt = {}
  for i, name in ipairs(search_dims) do
    def_pt[i] = samplers[name].normalize(samplers[name].center)
  end

  local eval_idx = 0
  local function evaluate (u)
    local params = {}
    local viol = 0.0
    for i, name in ipairs(search_dims) do
      local ui = u[i]
      local ci = ui
      if ci < 0 then ci = 0 elseif ci > 1 then ci = 1 end
      local d = ui - ci
      viol = viol + d * d
      params[name] = samplers[name].denormalize(ci)
    end
    fill_rest(params)
    eval_idx = eval_idx + 1
    local score, metrics = trial_fn(params, {
      trial = eval_idx, trials = trials, global_best_score = best_score, phase = "cmaes",
    })
    local failed = metrics and metrics.failed
    local feasible = (viol == 0.0) and not failed
    if (not failed) and (score > best_score) then
      best_score = score
      best_params = params
    end
    return -score, viol, feasible
  end

  local function cma_run (m0, lambda, sigma0, run_cap, eval_mean)

    local cma = capi.cma(n, lambda, sigma0, m0)
    local run_evals = 0

    if eval_mean and eval_idx < trials then
      local u0 = {}
      for i = 1, n do u0[i] = m0[i] end
      evaluate(u0)
      run_evals = run_evals + 1
    end

    while true do

      if eval_idx + lambda > trials then break end
      if run_evals + lambda > run_cap then break end
      local cand = cma:ask()
      local fs, feas, viols = {}, {}, {}
      for k = 1, lambda do
        local f, viol, feasible = evaluate(cand[k])
        fs[k] = f
        viols[k] = viol
        feas[k] = feasible
        run_evals = run_evals + 1
      end
      local stop = cma:tell(fs, feas, viols)
      if stop then break end
      if run_evals >= run_cap then break end
      if eval_idx >= trials then break end
    end

    return run_evals
  end

  local lambda0 = 4 + num.floor(3 * num.log(n))
  if lambda0 < 4 then lambda0 = 4 end
  local sigma0 = 0.3
  local evals_large = 0
  local evals_small = 0
  local i_large = 0

  local function rand_mean ()
    local mm = {}
    for i = 1, n do mm[i] = uniform() end
    return mm
  end

  local explore_cap = num.floor(trials / 2)
  if explore_cap < lambda0 then explore_cap = trials end
  do
    local used = cma_run(def_pt, lambda0, sigma0, explore_cap, true)
    evals_large = evals_large + used
  end
  while eval_idx < explore_cap do
    local remaining = explore_cap - eval_idx
    if remaining < 2 then break end
    local used
    if evals_small < evals_large then

      local u = uniform()
      local up = uniform()
      local lam = num.floor(lambda0 * (0.5 * 2 ^ i_large) ^ (u * u))
      if lam < 2 then lam = 2 end
      if lam > remaining then lam = remaining end
      local sig = sigma0 * 10 ^ (-2 * up)
      local cap = num.floor(num.max(1, evals_large) / 2)
      if cap < lam then cap = lam end
      if cap > remaining then cap = remaining end
      used = cma_run(rand_mean(), lam, sig, cap, false)
      evals_small = evals_small + used
    else

      i_large = i_large + 1
      local lam = lambda0 * 2 ^ i_large
      if lam > remaining then lam = remaining end
      used = cma_run(rand_mean(), lam, sigma0, remaining, false)
      evals_large = evals_large + used
    end
    if used == 0 then break end
  end

  local sigma_ref = 0.1
  while eval_idx < trials do
    if not best_params then break end
    local seed = {}
    for i, name in ipairs(search_dims) do
      seed[i] = samplers[name].normalize(best_params[name])
    end
    local used = cma_run(seed, lambda0, sigma_ref, trials, false)
    sigma_ref = sigma_ref * 0.5
    if used == 0 or sigma_ref < 1e-3 then break end
  end

  return best_params

end

local function default_trial_fn (args, dense, metric, k)
  local ridge = require("santoku.learn.ridge")
  local function mk_ridge (kd)
    kd.ridge = kd.ridge or ridge.create({ gram = kd.gram })
    return kd.ridge
  end
  if dense then
    local eval = require("santoku.learn.evaluator")
    return function (kd)
      local r = mk_ridge(kd)
      local s = r:regress(kd.val_codes)
      local m = eval.regress_accuracy(s, args.val_targets)
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
      local s = r:regress(kd.val_codes)
      local f1 = probe:calibrate({ scores = s, n_samples = cand:offsets():size() - 1, cand = cand, gold = gold })
      return f1, { span_f1 = f1, offset = probe:offset() }
    end
  end
  if metric == "single" then
    local probe = decide.create({ n_labels = nl, single = true })
    return function (kd)
      local r = mk_ridge(kd)
      local s = r:regress(kd.val_codes)
      local _, m = probe:score({ scores = s, n_samples = vn, expected = args.val_y })
      return m.accuracy, { macro_f1 = m.macro_f1, accuracy = m.accuracy }
    end
  end
  local probe = decide.create({ n_labels = nl })
  return function (kd)
    local r = mk_ridge(kd)
    local P = r:label(kd.val_codes, k)
    local f1, p, rc = probe:calibrate({ pred = P, n_samples = vn, expected = args.val_y })
    return f1, { f1 = f1, precision = p, recall = rc, offset = probe:offset() }
  end
end

local function decode_mode (args, dense)
  if dense then return false, nil end
  if args.cand or (args.fold_split and args.fold_split.val_cand) then return true, "span" end
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
  { key = "scales", defaults = { min = 0.01, max = 1000, log = true }, gauge = true },
  { key = "exponent", defaults = { min = 0, max = 8 } },
}

M.krr = function (args)
  local spectral = require("santoku.learn.spectral")
  local ridge = require("santoku.learn.ridge")
  local fvec = require("santoku.fvec")
  err.assert(args.n_landmarks, "n_landmarks required")
  args.folds = args.folds or 5
  if not args.rebuild then
    if args.pool_blocks then args = require("santoku.learn.util").fold_blocks(args)
    elseif args.pool_codes then args = require("santoku.learn.util").fold_dense(args) end
  end
  if args.each == nil then
    args.each = require("santoku.learn.util").make_ridge_log(require("santoku.utc").stopwatch())
  end
  local function resolve_knob (spec)
    if type(spec) ~= "table" then return spec end
    if spec[1] ~= nil then
      local v = {}
      for i = 1, veclen(spec) do
        local e = spec[i]
        if e == nil then v[i] = false
        elseif type(e) == "table" then v[i] = e.def or e.max or e.min
        else v[i] = e end
      end
      return v
    end
    if type(spec.def) == "table" then
      local v = {}
      for i = 1, veclen(spec.def) do
        local d = spec.def[i]
        if d == nil then v[i] = false else v[i] = d end
      end
      return v
    end
    return spec.def or spec.max or spec.min
  end

  local function resolve_params ()
    local p = {}
    for _, kdef in ipairs(REBUILD_KNOBS) do
      if args[kdef.key] ~= nil then p[kdef.key] = resolve_knob(args[kdef.key]) end
    end
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
    else families.matern = true end
  end
  err.assert(not (families.cosine and families.matern), "krr: kernel mixes cosine and matern; pick one family")
  local function cat_spec (v, deflist)
    if v == nil then return deflist end
    if type(v) ~= "table" then return v end
    if #v > 0 then return v end
    local s = {}
    for i = 1, #deflist do s[i] = deflist[i] end
    s.def = v.def
    return s
  end
  args.gamma = spec_defaults(args.gamma, { min = 1e-2, max = 16, log = true })
  args.nu = cat_spec(args.nu, { 3, 0, 1, 2 })
  local seed = 5
  local kernel_samplers = build_samplers(args, { "nu", "gamma" }, seed)

  local strials = args.search_trials or 0
  local do_search = strials > 1
  local frozen = strials == 0

  local decode_offset = args.decode_offset
  if type(decode_offset) == "table" then
    decode_offset = frozen and decode_offset.def or nil
  elseif not frozen then decode_offset = nil end
  args.lambda = spec_defaults(args.lambda, { min = 1e-7, max = 8, log = true })

  if do_search and type(args.lambda) == "table" and args.lambda.search ~= nil then
    args.lambda.def = args.lambda.search  -- luacheck: ignore
  end
  local label_names = { "lambda" }
  local label_samplers = build_samplers(args, label_names, seed)
  local k = not dense and (args.k or 32) or nil
  local tiled = not dense
  local want_decode, mode = decode_mode(args, dense)
  local use_oof = decode_offset == nil and (mode == "span" or mode == "multilabel")

  local spectral_args = {
    x = args.x, y = args.y,
    blocks = args.blocks,
    n_tokens = args.n_tokens, n_samples = args.n_samples,
    d_input = args.d_input,
    n_labels = args.n_labels,
    targets = args.targets, n_targets = args.n_targets,
  }
  local nl_cap = args.n_labels or args.n_targets or 1

  local xtx_shared, xty_shared
  local function ensure_shared_bufs ()
    xtx_shared = xtx_shared or fvec.create(args.n_landmarks * args.n_landmarks)
    xty_shared = xty_shared or fvec.create(args.n_landmarks * nl_cap)
    spectral_args.xtx_buf = xtx_shared
    spectral_args.xty_buf = xty_shared
  end
  local w_shared = fvec.create(args.n_landmarks * nl_cap)
  if tiled then
    spectral_args.tile_labels = 1024
  end
  local proj_shared, sims_shared, row_shared
  if do_search then
    proj_shared = fvec.create(args.n_landmarks * args.n_landmarks)
    sims_shared = fvec.create(4096 * args.n_landmarks)
    row_shared = fvec.create(128 * args.n_landmarks)
    spectral_args.proj_buf = proj_shared
    spectral_args.sims_buf = sims_shared
    spectral_args.row_buf = row_shared
  end
  local search_m = args.search_landmarks or args.n_landmarks

  local fastpath_cal = use_oof and (search_m == args.n_landmarks)
  local lm_slot
  local xtx_slot, xty_slot, factor_slot
  local enc_slot
  local search_fb
  local fold_bufs_release
  local function release_enc_scratch ()
    spectral_args.proj_buf = nil
    spectral_args.sims_buf = nil; spectral_args.row_buf = nil
    spectral_args.factor_buf = nil
    spectral_args.encoder = nil
    if enc_slot then enc_slot:destroy(); enc_slot = nil end
    if proj_shared then proj_shared:destroy() end
    if sims_shared then sims_shared:destroy() end
    if row_shared then row_shared:destroy() end
    if lm_slot then lm_slot:destroy() end
    if xtx_slot then xtx_slot:destroy() end
    if xty_slot then xty_slot:destroy() end
    if factor_slot then factor_slot:destroy() end
    proj_shared = nil; sims_shared = nil; row_shared = nil -- luacheck: ignore
    lm_slot = nil
    xtx_slot = nil; xty_slot = nil; factor_slot = nil
    if search_fb then fold_bufs_release(search_fb); search_fb = nil end
    spectral_args.landmarks = nil
    ensure_shared_bufs()
  end
  local function release_cv ()
    if xtx_shared then xtx_shared:destroy() end
    if xty_shared then xty_shared:destroy() end
  end
  local function build_kd (spec, at_search)
    if args.rebuild and spec.params ~= nil then
      local rb = args.rebuild(spec.params, at_search ~= nil)
      spectral_args.x = rb.x
      spectral_args.blocks = rb.blocks
      spectral_args.colscale = rb.colscale
    end
    spectral_args.y = args.y
    spectral_args.targets = args.targets
    spectral_args.kernel = spec.kernel
    spectral_args.gamma = spec.gamma
    spectral_args.nu = spec.nu
    spectral_args.strata = args.pool_strata

    if at_search == "search" then
      if lm_slot == nil then
        lm_slot = spectral.uniform_landmarks(spectral_args, search_m, seed + 1000, args.stratum_rows)
        xtx_slot = fvec.create(search_m * search_m)
        xty_slot = fvec.create(search_m * nl_cap)

        factor_slot = fvec.create(search_m * search_m)
      end
      spectral_args.landmarks = lm_slot
      spectral_args.xtx_buf = xtx_slot
      spectral_args.xty_buf = xty_slot
      spectral_args.factor_buf = factor_slot

      spectral_args.encoder = enc_slot
    elseif at_search == "cal" then
      spectral_args.landmarks = spectral.uniform_landmarks(spectral_args, args.n_landmarks, seed + 1000, args.stratum_rows)
      spectral_args.xtx_buf = xtx_shared
      spectral_args.xty_buf = xty_shared
      spectral_args.factor_buf = nil
      spectral_args.encoder = nil
    else
      spectral_args.landmarks = spectral.uniform_landmarks(spectral_args, args.n_landmarks, seed + 1000)
      spectral_args.xtx_buf = xtx_shared
      spectral_args.xty_buf = xty_shared
      spectral_args.factor_buf = nil
      spectral_args.encoder = nil
    end
    local _, sp_enc, gram = spectral.encode(spectral_args)
    if at_search == "search" then enc_slot = sp_enc end
    return { sp_enc = sp_enc, gram = gram }
  end

  local function build_folds (spec, fb, at_search)
    spectral_args.fold_assign = fb.assign
    spectral_args.fold_xtx = fb.xtx
    spectral_args.fold_xty = fb.xty
    spectral_args.fold_sv = fb.sv
    spectral_args.fold_tv = fb.tv
    spectral_args.fold_codes = fb.codes
    local kd = build_kd(spec, at_search)
    spectral_args.fold_assign = nil
    spectral_args.fold_xtx = nil; spectral_args.fold_xty = nil
    spectral_args.fold_sv = nil; spectral_args.fold_tv = nil
    spectral_args.fold_codes = nil
    local kds = {}
    for f = 1, fb.n do
      kds[f] = {
        gram = kd.gram:fold(fb.xtx[f], fb.xty[f], fb.sv[f], fb.tv[f], fb.counts[f]),
        val_codes = fb.codes[f],
      }
      kds[f].gram:attach(fb.factor)
    end
    return kd, kds
  end
  local function fold_bufs (nf, m, split)
    local mtx = require("santoku.mtx")
    local dvec = require("santoku.dvec")
    local store = require("santoku.store")
    local counts = split.val_n
    local fb = { n = nf, counts = counts, assign = split.assign,
      xtx = {}, xty = {}, sv = {}, tv = {}, codes = {}, store = store.create({ disk = true }) }
    local views = {}
    for f = 1, nf do
      fb.xtx[f] = fvec.create(m * m)
      fb.xty[f] = fvec.create(m * nl_cap)
      fb.sv[f] = fvec.create(m)
      fb.tv[f] = dvec.create(nl_cap)
      views[f] = fb.store:fvec(counts[f] * m)
    end
    fb.store:open()
    for f = 1, nf do
      fb.codes[f] = mtx.create({ data = views[f], n_rows = counts[f], n_cols = m })
    end
    fb.factor = fvec.create(m * m)
    return fb
  end
  fold_bufs_release = function (fb)
    for f = 1, fb.n do
      fb.xtx[f]:destroy(); fb.xty[f]:destroy()
    end
    fb.store:close()
    fb.factor:destroy()
  end

  local function oof_decider (kds, split)
    local ivec = require("santoku.ivec")
    local nf = #kds
    local fvc = split.val_cand
    local fvg = split.val_gold
    local fvy = split.val_y
    local fvn = split.val_n
    if mode == "span" then
      local spans = require("santoku.spans")
      local pooled_s = fvec.create()
      local fold_s = {}
      local pool = split._oof_pool
      if not pool then
        local function clone_spans (S)
          local o, s, e, t = S:offsets(), S:col("s"), S:col("e"), S:col("ty")
          local no, ns, ne, nt = ivec.create(), ivec.create(), ivec.create(), ivec.create()
          no:copy(o); ns:copy(s); ne:copy(e); nt:copy(t)
          return spans.create({ offsets = no, s = ns, e = ne, ty = nt })
        end
        for f = 1, nf do
          if not pool then pool = { cand = clone_spans(fvc[f]), gold = clone_spans(fvg[f]) }
          else pool.cand:append(clone_spans(fvc[f])); pool.gold:append(clone_spans(fvg[f])) end
        end
        split._oof_pool = pool
      end
      for f = 1, nf do
        local r = kds[f].ridge or ridge.create({ gram = kds[f].gram })
        kds[f].ridge = r
        local s = r:regress(kds[f].val_codes)
        fold_s[f] = s
        pooled_s:copy(s)
      end
      local decider, m = M.decide({ n_labels = args.n_labels, reject = args.reject, val_scores = pooled_s,
        val_n_samples = pool.cand:offsets():size() - 1, val_cand = pool.cand, val_gold = pool.gold })
      local scs, fms = {}, {}
      for f = 1, nf do
        local f1, fm = decider:score({ scores = fold_s[f],
          n_samples = fvc[f]:offsets():size() - 1,
          cand = fvc[f], gold = fvg[f] })
        scs[f] = f1; fms[f] = fm
      end
      return decider, m, scs, fms
    end

    local ml = split._oof_ml
    if not ml then
      ml = { foldP = {} }
      local Y
      for f = 1, nf do
        if f == 1 then Y = fvy[f]:clone() else Y:append(fvy[f]) end
      end
      ml.Y = Y
      ml.Yn = select(1, Y:shape())
      split._oof_ml = ml
    end
    local fold_P = ml.foldP
    for f = 1, nf do
      local r = kds[f].ridge or ridge.create({ gram = kds[f].gram })
      kds[f].ridge = r
      fold_P[f] = r:label(kds[f].val_codes, k, fold_P[f])
      r:shrink()
    end
    local P = ml.P
    if not P then
      P = fold_P[1]:clone()
      for f = 2, nf do P:append(fold_P[f]) end
      ml.P = P
    else
      P:clear()
      for f = 1, nf do P:append(fold_P[f]) end
    end
    local decider, m = M.decide({ n_labels = args.n_labels, val_pred = P,
      val_n_samples = ml.Yn, val_expected = ml.Y })
    local scs, fms = {}, {}
    for f = 1, nf do
      local f1, fm = decider:score({ pred = fold_P[f], n_samples = fvn[f],
        expected = fvy[f] })
      scs[f] = f1; fms[f] = fm
    end
    return decider, m, scs, fms
  end

  local function eval_folds (kds, lam, fns, split, race, best_hint)
    local nf = #kds
    local mean, agg, decider, pooled_m = 0.0, {}, nil, nil
    if not fns then
      for f = 1, nf do kds[f].gram:solve(lam) end
      local dec, m, foldsc, foldms = oof_decider(kds, split)
      decider = dec
      pooled_m = m
      for f = 1, nf do mean = mean + foldsc[f] end
      mean = mean / nf
      if foldms then
        for f = 1, nf do
          local fm = foldms[f]
          if fm then for kk, vv in pairs(fm) do if type(vv) == "number" then agg[kk] = (agg[kk] or 0) + vv end end end
        end
        for kk, vv in pairs(agg) do agg[kk] = vv / nf end
      else
        agg = m
      end
      agg.offset = dec:offset()
      agg.fold_std = fold_std(foldsc, mean, nf)
    else
      local pfx = {}
      local folds_sc = {}
      for f = 1, nf do
        kds[f].gram:solve(lam)
        local sc, m = fns[f](kds[f])
        mean = mean + sc
        folds_sc[f] = sc
        pfx[f] = mean / f
        if m then for kk, vv in pairs(m) do if type(vv) == "number" then agg[kk] = (agg[kk] or 0) + vv end end end
        if race and best_hint and f >= 2 and f < nf and race.count >= 8 then
          local st = race.stats[f]
          if st and st.n >= 8 then
            local sd = num.sqrt(st.m2 / (st.n - 1))
            if sd < 1e-4 then sd = 1e-4 end
            local predicted = pfx[f] + st.mean
            if predicted + 3 * sd < best_hint then
              for kk, vv in pairs(agg) do agg[kk] = vv / f end
              return predicted, agg, nil, nil, true
            end
          end
        end
      end
      mean = mean / nf
      for kk, vv in pairs(agg) do agg[kk] = vv / nf end
      agg.fold_std = fold_std(folds_sc, mean, nf)
      if race then
        for f = 2, nf - 1 do
          local st = race.stats[f]
          if not st then st = { n = 0, mean = 0.0, m2 = 0.0 }; race.stats[f] = st end
          local r = mean - pfx[f]
          st.n = st.n + 1
          local d = r - st.mean
          st.mean = st.mean + d / st.n
          st.m2 = st.m2 + d * (r - st.mean)
        end
        race.count = race.count + 1
      end
    end
    return mean, agg, decider, pooled_m
  end

  local function calibrate_and_deploy (spec, params)
    local fin = nil
    if want_decode and (mode == "span" or mode == "multilabel") then
      local fb = fold_bufs(args.folds, args.n_landmarks, args.fold_split)
      local cal_kd, kds = build_folds(spec, fb, "cal")
      cal_kd.gram:release()
      cal_kd.sp_enc:destroy()
      cal_kd = nil -- luacheck: ignore
      for f = 1, args.folds do kds[f].gram:solve(params.lambda) end
      local decider, metrics = oof_decider(kds, args.fold_split)
      for f = 1, args.folds do kds[f].gram:release() end
      kds = nil -- luacheck: ignore
      fold_bufs_release(fb)
      fin = { decider = decider, metrics = metrics }
    end
    local kd = build_kd(spec)
    kd.gram:solve(params.lambda, w_shared)
    return kd, fin
  end
  local function deploy_of (enc)
    if spectral_args.blocks then
      return function (ext, out, start, count)
        return enc:encode({ blocks = ext, start = start, count = count }, out)
      end
    end
    return function (x, out) return enc:encode(x, out) end
  end

  local function finish (kd, params, solve, fin, fold_sd)
    local r = ridge.create({ gram = kd.gram })
    kd.gram:release()
    if args.each then args.each({ event = "done", params = params, emb_d = kd.sp_enc:dims(),
      solve = solve, fold_std = fold_sd }) end
    local decider, decider_metrics
    if want_decode then
      if decode_offset ~= nil and mode ~= "single" then
        decider = require("santoku.learn.decide").create({ n_labels = args.n_labels,
          span = mode == "span", reject = args.reject, offset = decode_offset })
      elseif fin and fin.decider then
        decider, decider_metrics = fin.decider, fin.metrics
      elseif mode == "single" then
        decider = require("santoku.learn.decide").create({ n_labels = args.n_labels, single = true })
      end
    end
    return kd.sp_enc, r, deploy_of(kd.sp_enc), params, decider, decider_metrics
  end

  local function center_spec ()
    local kname = kernels[1]
    local base = { kernel = kname }
    if kname == "matern" then
      base.nu = kernel_samplers.nu.center
      base.gamma = kernel_samplers.gamma.center
    end
    return kname, base
  end
  local function spec_with (base, p)
    local spec = { params = p }
    for kk, vv in pairs(base) do spec[kk] = vv end
    return spec
  end
  local rebuild_knobs = {}
  for _, kdef in ipairs(REBUILD_KNOBS) do
    local spec = args[kdef.key]
    if spec ~= nil then
      local knob = { key = kdef.key, gauge = kdef.gauge, names = {}, samplers = {} }
      if type(spec) == "table" and spec[1] == nil and type(spec.def) == "table" then
        local vec = {}
        for i = 1, veclen(spec.def) do
          local d = spec.def[i]
          if d == nil or d == false then
            vec[i] = false
          else
            vec[i] = { min = spec.min, max = spec.max, log = spec.log, def = d }
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
        local c = num.log(kdef.defaults.max or 100) / num.sqrt(1 - 1 / num.max(M, 2))
        local centers
        if all_def and M > 1 then
          local y = {}
          for j = 1, M do y[j] = num.log(defs[j]) end
          centers = ilr_forward(y)
        end
        for k = 1, M - 1 do
          local nm = kdef.key .. "_z" .. k
          args[nm] = { min = -c, max = c, def = centers and centers[k] }
          knob.names[#knob.names + 1] = nm
          knob.samplers[nm] = build_samplers(args, { nm }, seed)[nm]
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
            knob.samplers[nm] = build_samplers(args, { nm }, seed)[nm]
          end
        end
      else
        knob.kind = "scalar"
        args[kdef.key] = spec_defaults(spec, kdef.defaults)
        knob.names[1] = kdef.key
        knob.samplers[kdef.key] = build_samplers(args, { kdef.key }, seed)[kdef.key]
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
          v[knob.active[j]] = num.exp(y[j])
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
          if type(v[i]) == "number" and v[i] > 0 then logsum = logsum + num.log(v[i]); cnt = cnt + 1 end
        end
        if cnt > 0 then
          local gm = num.exp(logsum / cnt)
          for i = 1, #v do if type(v[i]) == "number" then v[i] = v[i] / gm end end
        end
      end
    end
    return p
  end
  if not do_search then
    local _, base = center_spec()

    local rk = {}
    for _, knob in ipairs(rebuild_knobs) do
      for _, nm in ipairs(knob.names) do local sm = knob.samplers[nm]; rk[nm] = sm and sm.center end
    end
    local rparams = has_knobs and params_of(rk) or nil
    local spec = spec_with(base, rparams)
    local lp = center_params(label_samplers, label_names)
    release_enc_scratch()
    local params = {}
    for kk, vv in pairs(base) do params[kk] = vv end
    if rparams then for kk, vv in pairs(rparams) do params[kk] = vv end end
    for _, n in ipairs(label_names) do params[n] = lp[n] end
    local kd, sstr, fin
    if frozen then
      err.assert(not (want_decode and decode_offset == nil and (mode == "span" or mode == "multilabel")),
        "krr: frozen (search_trials=0) span/multilabel decode requires a pinned decode_offset; use search_trials=1 to calibrate")
      kd = build_kd(spec)
      kd.gram:solve(params.lambda, w_shared)
      sstr = "cholesky"
    else

      kd, fin = calibrate_and_deploy(spec, params)
      sstr = "calibrate"
    end
    local sp_enc, r, vcodes, params_out, decider, dmetrics = finish(kd, params, sstr, fin)
    release_cv()
    if args.release then args.release() end
    return sp_enc, r, vcodes, params_out, decider, dmetrics
  end
  local nfolds = args.folds or 1
  err.assert(not do_search or nfolds > 1, "krr: search requires folds > 1 (all-CV; no external dev set)")
  local fold_trial_fns
  if nfolds > 1 then
    local sp = args.fold_split
    fold_trial_fns = {}
    for f = 1, nfolds do
      fold_trial_fns[f] = default_trial_fn({
        val_y = sp.val_y and sp.val_y[f] or nil,
        val_cand = sp.val_cand and sp.val_cand[f] or nil,
        val_gold = sp.val_gold and sp.val_gold[f] or nil,
        reject = args.reject,
        val_n_samples = sp.val_n[f],
        val_targets = sp.val_targets and sp.val_targets[f] or nil,
        n_labels = args.n_labels,
      }, dense, mode == "multilabel" and "fmeasure" or mode, k)
    end
  end

  local race_state = (not use_oof) and nfolds > 1 and { stats = {}, count = 0 } or nil
  local function run_folds (spec, lam, best_hint)
    if not search_fb then
      search_fb = fold_bufs(nfolds, search_m, args.fold_split)
    end
    local kd, kds = build_folds(spec, search_fb, "search")
    local meta = { dims = kd.sp_enc:dims() }
    kd.gram:release()
    local mean, agg, dec, pooled_m, raced = eval_folds(kds, lam, (not use_oof) and fold_trial_fns or nil,
      args.fold_split, race_state, best_hint)

    for f = 1, #kds do kds[f].ridge = nil; kds[f].gram:destroy() end
    kd.gram:destroy()
    return mean, agg, meta, dec, pooled_m, raced
  end
  local btot = args.search_trials or 0
  local function base_with (base, p)
    base.params = p
    for kk, vv in pairs(p) do base[kk] = vv end
    return base
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
  local best_params, best_score = nil, -num.huge
  local best_fin = nil
  local best_fold_std = nil
  local worst_score = nil
  local bi = 0
  local function eval_kd (spec, base, gp)
    local ib = {}
    for kk, vv in pairs(base) do ib[kk] = vv end
    for _, n in ipairs(label_names) do ib[n] = gp[n] end
    local ok, sc, sm, meta, dec, pooled_m, raced = pcall(run_folds, spec, gp.lambda, best_score)
    local failed = not ok or sc == nil
    if failed then
      sc, sm, meta = (worst_score or -1e18), { failed = true }, meta or {}
    elseif raced then

      sm.failed = true
      sm.raced = true
    else
      worst_score = worst_score and num.min(worst_score, sc) or sc
    end

    local sel = (failed or raced) and -num.huge or sc
    bi = bi + 1
    local improved = sel > best_score
    if improved then
      best_score = sel; best_params = ib
      best_fold_std = sm and sm.fold_std
      if fastpath_cal and dec then best_fin = { decider = dec, metrics = pooled_m } end
    end
    if args.each then
      args.each({ event = "trial", phase = "kernel", trial = bi, trials = btot,
        params = ib, score = sc, metrics = sm, emb_d = meta.dims,
        best = best_score, is_new_best = improved })
    end
    return sc, sm
  end
  local run
  if families.matern then
    run = {
      names = { "nu", "gamma" },
      samplers = { nu = kernel_samplers.nu, gamma = kernel_samplers.gamma },
      base_of = function (gp) return { kernel = "matern", nu = gp.nu, gamma = gp.gamma } end,
    }
  else
    run = {
      names = {},
      samplers = {},
      base_of = function () return { kernel = "cosine" } end,
    }
  end
  cmaes_search({
    param_names = with_knobs(run.names), samplers = merge_knob_samplers(run.samplers),
    trials = args.search_trials or 0,
    trial_fn = function (gp)
      local p = params_of(gp)
      local base = run.base_of(gp)
      return eval_kd(spec_with(base, p), base_with(base, p), gp)
    end,
  })
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
  release_enc_scratch()

  local best_kd, fin, solve_tag
  if best_fin then
    best_kd = build_kd(best_params)
    best_kd.gram:solve(best_params.lambda, w_shared)
    fin = best_fin
    solve_tag = "calibrate"
  else
    best_kd, fin = calibrate_and_deploy(best_params, best_params)
    solve_tag = "calibrate"
  end
  local sp_enc, r, vcodes, params_out, decider, dmetrics =
    finish(best_kd, best_params, solve_tag, fin, best_fold_std)
  release_cv()
  if args.release then args.release() end
  return sp_enc, r, vcodes, params_out, decider, dmetrics
end

M.retrieval = function (args)
  local retrieval = require("santoku.learn.retrieval")
  local sets = err.assert(args.datasets, "datasets required")
  local downside = args.downside or 0
  local names = { "alpha" }
  args.alpha = spec_defaults(args.alpha, { min = 0, max = 1, def = 0.96 })
  local samplers = build_samplers(args, names, 1)
  local base = {}
  for _, s in ipairs(sets) do
    local _, b = s.candidates:ndcg(s.qrels, 10)
    s.base = b
    base[s.name] = b
  end
  local function evaluate (p, per)
    local obj = 0
    for _, s in ipairs(sets) do
      local R = retrieval.rerank({ candidates = s.candidates, query_codes = s.query_codes,
        doc_codes = s.doc_codes, alpha = p.alpha })
      local _, nd = R:ndcg(s.qrels, 10)
      local r = nd / s.base
      obj = obj + r - downside * (r < 1 and 1 - r or 0)
      per[s.name] = nd
    end
    return obj / #sets
  end
  local best = cmaes_search({
    param_names = names, samplers = samplers, trials = args.search_trials or 0,
    trial_fn = function (p, info)
      local per = {}
      local sc = evaluate(p, per)
      if args.each then args.each({ event = "trial", params = p, score = sc, per = per, info = info }) end
      return sc, per
    end,
  })
  return best, base
end

M.lexical = function (args)
  local retrieval = require("santoku.learn.retrieval")
  local sets = err.assert(args.datasets, "datasets required")
  local downside = args.downside or 0
  local depth = args.depth or 100
  local names = { "k1", "b", "ngram" }
  args.k1 = spec_defaults(args.k1, { min = 0.1, max = 4, log = true, def = 1.2 })
  args.b = spec_defaults(args.b, { min = 0, max = 1, def = 0.75 })
  args.ngram = args.ngram or { 1, 2, def = 1 }
  local samplers = build_samplers(args, names, 1)
  local function ranker (s, ng)
    local r = s.rankers[ng]
    if not r then
      local X, Q = retrieval.lexical({ corpus_texts = s.corpus_texts, query_texts = s.query_texts, ngram_max = ng })
      r = retrieval.bm25_ranker(X, Q)
      s.rankers[ng] = r
    end
    return r
  end
  local base = {}
  for _, s in ipairs(sets) do
    s.rankers = {}
    local _, nd = ranker(s, 1)(1.2, 0.75, depth):ndcg(s.qrels, 10)
    s.base = nd
    base[s.name] = nd
  end
  local function evaluate (p, per)
    local obj = 0
    for _, s in ipairs(sets) do
      local _, nd = ranker(s, p.ngram)(p.k1, p.b, depth):ndcg(s.qrels, 10)
      local r = nd / s.base
      obj = obj + r - downside * (r < 1 and 1 - r or 0)
      per[s.name] = nd
    end
    return obj / #sets
  end
  local best = cmaes_search({
    param_names = names, samplers = samplers, trials = args.search_trials or 0,
    trial_fn = function (p, info)
      local per = {}
      local sc = evaluate(p, per)
      if args.each then args.each({ event = "trial", params = p, score = sc, per = per, info = info }) end
      return sc, per
    end,
  })
  return best, base
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

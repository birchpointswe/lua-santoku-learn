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
    local is_log = not not spec.log
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
          if is_int then x = num.floor(x + 0.5) end
        elseif is_int then
          x = num.floor(u * (maxv - minv + 1) + minv)
        else
          x = u * lin_span + minv
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
        if is_int then x = num.floor(x + 0.5) end
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

-- Cyclic Jacobi eigensolver for a symmetric n x n matrix (n small, <= ~15).
-- Input C is a flat row-major array of length n*n (assumed already symmetrized by caller).
-- Returns ev (eigenvalues, 1-based array of length n) and B (eigenvectors as columns:
-- B[(i-1)*n + j] is row i, column j; column j is the eigenvector for ev[j]).
local function jacobi_eigen (C, n)
  local a = {}
  for i = 1, n * n do a[i] = C[i] end
  local V = {}
  for i = 1, n do
    for j = 1, n do
      V[(i - 1) * n + j] = (i == j) and 1.0 or 0.0
    end
  end
  local function off_norm ()
    local s = 0.0
    for p = 1, n - 1 do
      for q = p + 1, n do
        local v = a[(p - 1) * n + q]
        s = s + v * v
      end
    end
    return num.sqrt(2.0 * s)
  end
  local function frob ()
    local s = 0.0
    for i = 1, n * n do s = s + a[i] * a[i] end
    return num.sqrt(s)
  end
  local tol = 1e-12 * frob()
  if tol <= 0 then tol = 1e-300 end
  for _ = 1, 100 do
    if off_norm() < tol then break end
    for p = 1, n - 1 do
      for q = p + 1, n do
        local apq = a[(p - 1) * n + q]
        if apq ~= 0.0 then
          local app = a[(p - 1) * n + p]
          local aqq = a[(q - 1) * n + q]
          local theta = (aqq - app) / (2.0 * apq)
          local t
          if theta >= 0 then
            t = 1.0 / (theta + num.sqrt(theta * theta + 1.0))
          else
            t = -1.0 / (-theta + num.sqrt(theta * theta + 1.0))
          end
          local c = 1.0 / num.sqrt(t * t + 1.0)
          local s = t * c
          -- rotate rows/cols p,q of a
          for k = 1, n do
            local akp = a[(k - 1) * n + p]
            local akq = a[(k - 1) * n + q]
            a[(k - 1) * n + p] = c * akp - s * akq
            a[(k - 1) * n + q] = s * akp + c * akq
          end
          for k = 1, n do
            local apk = a[(p - 1) * n + k]
            local aqk = a[(q - 1) * n + k]
            a[(p - 1) * n + k] = c * apk - s * aqk
            a[(q - 1) * n + k] = s * apk + c * aqk
          end
          -- accumulate eigenvectors
          for k = 1, n do
            local vkp = V[(k - 1) * n + p]
            local vkq = V[(k - 1) * n + q]
            V[(k - 1) * n + p] = c * vkp - s * vkq
            V[(k - 1) * n + q] = s * vkp + c * vkq
          end
        end
      end
    end
  end
  local ev = {}
  for j = 1, n do ev[j] = a[(j - 1) * n + j] end
  return ev, V
end

local cmaes_search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local trials = args.trials or 120
  local each_cb = args.each
  local skip_final = args.skip_final
  local constrain_fn = args.constrain
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
  local n = #search_dims

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

  local rmax = rand.fast_max + 1
  local function uniform () return rand.fast_random() / rmax end

  -- Trivial case: no searchable dims -> single center eval (mirrors search's t=1 path).
  if n == 0 then
    local params = {}
    fill_rest(params)
    if constrain_fn then constrain_fn(params) end
    local score, metrics, result = trial_fn(params, {
      trial = 1, trials = trials, is_final = false,
      global_best_score = best_score, phase = "cmaes",
    })
    local new_best = not (metrics and metrics.failed)
    if new_best then
      best_score = score; best_params = params
      best_result = result; best_metrics = metrics
    end
    if each_cb then
      each_cb({ event = "trial", trial = 1, trials = trials, params = params,
        score = score, metrics = metrics, global_best_score = best_score,
        is_new_best = new_best, phase = "cmaes" })
    end
    if not skip_final and best_params then
      local _, final_metrics, final_result = trial_fn(best_params, { is_final = true })
      best_result = final_result; best_metrics = final_metrics
    end
    return best_result, best_params, best_metrics
  end

  -- Center point in normalized [0,1]^n (warm seed if pinned, else box center).
  local def_pt = {}
  for i, name in ipairs(search_dims) do
    def_pt[i] = samplers[name].normalize(samplers[name].center)
  end

  -- Evaluate a normalized point u (n-vec). Returns f (=-score), viol, feasible.
  -- Updates global best/each_cb; never lets a failed trial become best.
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
    if constrain_fn then constrain_fn(params) end
    eval_idx = eval_idx + 1
    local score, metrics, result = trial_fn(params, {
      trial = eval_idx, trials = trials, is_final = false,
      global_best_score = best_score, phase = "cmaes",
    })
    local failed = metrics and metrics.failed
    local feasible = (viol == 0.0) and not failed
    -- CMA optimizes raw CV score, the same objective the deploy selects by.
    local new_best = (not failed) and (score > best_score)
    if new_best then
      best_score = score
      best_params = params
      best_result = result
      best_metrics = metrics
    end
    if each_cb then
      each_cb({
        event = "trial", trial = eval_idx, trials = trials, params = params,
        score = score, metrics = metrics, global_best_score = best_score,
        is_new_best = new_best, phase = "cmaes",
      })
    end
    return -score, viol, feasible
  end

  -- One CMA-ES run. m0: n-vec start mean; lambda; sigma0; run_cap: this run's eval cap.
  -- Returns evals consumed by this run.
  local function cma_run (m0, lambda, sigma0, run_cap, eval_mean)
    local mu = num.floor(lambda / 2)
    if mu < 1 then mu = 1 end
    -- recombination weights (positive, sum to 1)
    local w = {}
    local wsum = 0.0
    local lhalf = num.log(lambda / 2 + 0.5)
    for i = 1, mu do
      local wi = lhalf - num.log(i)
      w[i] = wi
      wsum = wsum + wi
    end
    local w2 = 0.0
    for i = 1, mu do w[i] = w[i] / wsum; w2 = w2 + w[i] * w[i] end
    local mu_eff = 1.0 / w2
    local c_sig = (mu_eff + 2) / (n + mu_eff + 5)
    local d_sig = 1 + 2 * num.max(0, num.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sig
    local c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    local c_1 = 2 / ((n + 1.3) * (n + 1.3) + mu_eff)
    local c_mu = num.min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) * (n + 2) + mu_eff))
    local chiN = num.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    local m = {}
    for i = 1, n do m[i] = m0[i] end
    local sigma = sigma0
    -- C flat row-major (length n*n), init identity
    local C = {}
    for i = 1, n do
      for j = 1, n do C[(i - 1) * n + j] = (i == j) and 1.0 or 0.0 end
    end
    local p_sig = {}
    local p_c = {}
    for i = 1, n do p_sig[i] = 0.0; p_c[i] = 0.0 end
    local B, D
    local g = 0
    local run_evals = 0

    -- stopping-history of per-gen best f
    local histlen = num.max(10, 10 + num.floor(30 * n / lambda))
    local best_hist = {}

    local safety_cap = 100 + num.floor(50 * n * n / lambda)

    -- Evaluate the exact mean (the incumbent/pin) as a real trial before sampling, so a
    -- warm/refine restart holds >= its seed by genuine evaluation (not a clamp). Standard
    -- CMA never evaluates its mean; we do, only when asked.
    if eval_mean and eval_idx < trials then
      local u0 = {}
      for i = 1, n do u0[i] = m[i] end
      evaluate(u0)
      run_evals = run_evals + 1
    end

    while true do
      -- Budget guard: never start a generation we cannot fully fund (keeps every
      -- CMA update over a full lambda-population and never overshoots `trials`).
      if eval_idx + lambda > trials then break end
      if run_evals + lambda > run_cap then break end
      -- 1. eigendecompose C (every generation)
      -- symmetrize
      for i = 1, n do
        for j = i + 1, n do
          local avg = 0.5 * (C[(i - 1) * n + j] + C[(j - 1) * n + i])
          C[(i - 1) * n + j] = avg
          C[(j - 1) * n + i] = avg
        end
      end
      local ev, V = jacobi_eigen(C, n)
      B = V
      D = {}
      for j = 1, n do
        local e = ev[j]
        if e < 1e-30 then e = 1e-30 end
        D[j] = num.sqrt(e)
      end

      -- 2-3. sample and evaluate lambda offspring
      local ys = {}   -- ys[k][i]
      local fs = {}   -- ys f-values
      local viols = {}
      local feas = {}
      for k = 1, lambda do
        local z = {}
        for j = 1, n do z[j] = rand.fast_normal(0, 1) end
        -- y = B * (D .* z)
        local dz = {}
        for j = 1, n do dz[j] = D[j] * z[j] end
        local y = {}
        for i = 1, n do
          local acc = 0.0
          for j = 1, n do acc = acc + B[(i - 1) * n + j] * dz[j] end
          y[i] = acc
        end
        local x = {}
        for i = 1, n do x[i] = m[i] + sigma * y[i] end
        local f, viol, feasible = evaluate(x)
        ys[k] = y
        fs[k] = f
        viols[k] = viol
        feas[k] = feasible
        run_evals = run_evals + 1
      end

      -- 4. rank-based sort: feasible first (by f asc), then infeasible (by viol asc, then f asc)
      local idx = {}
      for k = 1, lambda do idx[k] = k end
      table.sort(idx, function (a2, b2)
        local fa, fb = feas[a2], feas[b2]
        if fa ~= fb then return fa end
        if fa then
          return fs[a2] < fs[b2]
        else
          if viols[a2] ~= viols[b2] then return viols[a2] < viols[b2] end
          return fs[a2] < fs[b2]
        end
      end)

      -- track per-gen best f
      local gen_best_f = fs[idx[1]]
      best_hist[#best_hist + 1] = gen_best_f

      -- 5. y_w = sum w_i * y_{idx[i]}
      local y_w = {}
      for i = 1, n do y_w[i] = 0.0 end
      for r = 1, mu do
        local yk = ys[idx[r]]
        local wr = w[r]
        for i = 1, n do y_w[i] = y_w[i] + wr * yk[i] end
      end

      -- 6. p_sig = (1-c_sig) p_sig + sqrt(c_sig (2-c_sig) mu_eff) * Cinvhalf(y_w)
      -- Cinvhalf(v) = B * ((1./D) .* (B^T v))
      local bt = {}
      for j = 1, n do
        local acc = 0.0
        for i = 1, n do acc = acc + B[(i - 1) * n + j] * y_w[i] end
        bt[j] = acc / D[j]
      end
      local cinv = {}
      for i = 1, n do
        local acc = 0.0
        for j = 1, n do acc = acc + B[(i - 1) * n + j] * bt[j] end
        cinv[i] = acc
      end
      local ps_coef = num.sqrt(c_sig * (2 - c_sig) * mu_eff)
      for i = 1, n do
        p_sig[i] = (1 - c_sig) * p_sig[i] + ps_coef * cinv[i]
      end

      -- ||p_sig||
      local ps_norm = 0.0
      for i = 1, n do ps_norm = ps_norm + p_sig[i] * p_sig[i] end
      ps_norm = num.sqrt(ps_norm)

      -- 7. h_sig
      local denom = num.sqrt(1 - (1 - c_sig) ^ (2 * (g + 1)))
      local h_sig = 0
      if (ps_norm / denom) < (1.4 + 2 / (n + 1)) * chiN then h_sig = 1 end

      -- 8. p_c
      local pc_coef = h_sig * num.sqrt(c_c * (2 - c_c) * mu_eff)
      for i = 1, n do
        p_c[i] = (1 - c_c) * p_c[i] + pc_coef * y_w[i]
      end

      -- 9. C update
      local delta = (1 - h_sig) * c_c * (2 - c_c)
      local c_scale = 1 - c_1 - c_mu
      for i = 1, n do
        for j = 1, n do
          local rank1 = p_c[i] * p_c[j] + delta * C[(i - 1) * n + j]
          local rankmu = 0.0
          for r = 1, mu do
            local yk = ys[idx[r]]
            rankmu = rankmu + w[r] * yk[i] * yk[j]
          end
          C[(i - 1) * n + j] = c_scale * C[(i - 1) * n + j] + c_1 * rank1 + c_mu * rankmu
        end
      end

      -- 10. sigma update (capture pre-update sigma for the step-5 mean move)
      local sigma_pre = sigma
      sigma = sigma * num.exp((c_sig / d_sig) * (ps_norm / chiN - 1))

      -- 11. m = m_new; g += 1  (m_new = m + sigma_pre * y_w, c_m = 1)
      for i = 1, n do m[i] = m[i] + sigma_pre * y_w[i] end
      g = g + 1

      -- stopping criteria
      if run_evals >= run_cap then break end
      if eval_idx >= trials then break end
      -- TolX
      local maxD = D[1]
      for j = 2, n do if D[j] > maxD then maxD = D[j] end end
      if sigma * maxD < 1e-11 then break end
      -- Condition
      local maxD2, minD2 = D[1] * D[1], D[1] * D[1]
      for j = 2, n do
        local d2 = D[j] * D[j]
        if d2 > maxD2 then maxD2 = d2 end
        if d2 < minD2 then minD2 = d2 end
      end
      if minD2 > 0 and (maxD2 / minD2) > 1e14 then break end
      -- Safety cap
      if g >= safety_cap then break end
      -- TolFun stagnation
      if #best_hist >= histlen then
        local lo, hi = num.huge, -num.huge
        for t2 = #best_hist - histlen + 1, #best_hist do
          local v = best_hist[t2]
          if v < lo then lo = v end
          if v > hi then hi = v end
        end
        if (hi - lo) < 1e-12 then break end
      end
    end

    return run_evals
  end

  -- BIPOP restart wrapper
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

  -- Phase 1 (exploration, ~half the budget): locate the basin. Run 0 starts from the
  -- seed/center and evaluates the exact incumbent first (warm-start holds >= its pin, by
  -- real eval). Then BIPOP random-mean restarts for coverage.
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
      -- SMALL regime
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
      -- LARGE regime
      i_large = i_large + 1
      local lam = lambda0 * 2 ^ i_large
      if lam > remaining then lam = remaining end
      used = cma_run(rand_mean(), lam, sigma0, remaining, false)
      evals_large = evals_large + used
    end
    if used == 0 then break end
  end

  -- Phase 2 (refinement): restart CMA from the best-so-far with a small, shrinking sigma,
  -- evaluating the incumbent each time so `best` never decreases. This is the automatic
  -- "iterated restart-from-best" that polishes the located basin to its optimum.
  local sigma_ref = 0.1
  while eval_idx < trials do
    if not best_params then break end
    local seed = {}
    for i, name in ipairs(search_dims) do
      seed[i] = samplers[name].normalize(best_params[name])
    end
    local used = cma_run(seed, lambda0, sigma_ref, trials, true)
    sigma_ref = sigma_ref * 0.5
    if used == 0 or sigma_ref < 1e-3 then break end
  end

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
    kd.ridge = kd.ridge or ridge.create({ gram = kd.gram })
    return kd.ridge
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
  tt = args.verbose and utc.ticktock() or nil
  local function prof_emit ()
    if tt and args.each then
      local stats, total = tt()
      args.each({ event = "profile", stats = stats, total = total })
    end
    tt = nil
  end
  args.folds = args.folds or 5
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
  -- raw defs, used only for the initial data-prep rebuild (shapes/fingerprints); model
  -- params always resolve through the knob machinery (params_of), search and frozen alike
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
  local seed = args.seed or 4
  local kernel_samplers = build_samplers(args, { "nu", "gamma" }, nil, seed)
  -- search_trials is the sole regime knob: 0 = frozen (pinned config+lambda+offset);
  -- 1 = calibrate (no search, use defs/pins, derive offset at deploy rank); >1 = search
  -- (kernel/gauge/lambda) then calibrate. The finalize (deploy-rank OOF offset calibration)
  -- runs whenever trials>=1; lambda is always the search/def lambda.
  local strials = args.search_trials or 0
  local do_search = strials > 1
  local frozen = strials == 0
  -- offset pin binds ONLY when frozen (trials=0); any trials>=1 re-derives it at deploy rank
  local decode_offset = args.decode_offset
  if decode_offset == SEARCH then decode_offset = nil
  elseif type(decode_offset) == "table" then
    decode_offset = frozen and decode_offset.def or nil
  elseif not frozen then decode_offset = nil end
  args.lambda = spec_defaults(args.lambda, { min = 1e-7, max = 8, log = true })
  -- def is the deploy-rank pin (binds at trials=0); search, when present, is the
  -- search-rank warm-start center (optimal lambda is rank-coupled, ~decades lower at
  -- search_landmarks than at deploy) -- without it the search centers on the deploy pin
  if do_search and type(args.lambda) == "table" and args.lambda.search ~= nil then
    args.lambda.def = args.lambda.search  -- luacheck: ignore
  end
  local label_names = { "lambda" }
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
  local nl_cap = args.n_labels or args.n_targets or 1
  -- deploy-rank shared XtX/xty are lazy: search folds use their own slots, so allocating
  -- these up front would idle m^2 floats for the whole search
  local xtx_shared, xty_shared
  local function ensure_shared_bufs ()
    xtx_shared = xtx_shared or fvec.create(args.n_landmarks * args.n_landmarks)
    xty_shared = xty_shared or fvec.create(args.n_landmarks * nl_cap)
    spectral_args.xtx_buf = xtx_shared
    spectral_args.xty_buf = xty_shared
  end
  local w_shared = fvec.create(args.n_landmarks * nl_cap)
  if tiled then
    spectral_args.tile_labels = tile_labels
    w_auto = args.w_buf -- only copy W out of the gram when the caller wants a serving buffer
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
  local lm_slot
  local xtx_slot, xty_slot, factor_slot
  local enc_slot
  local search_fb
  local function release_enc_scratch ()
    spectral_args.proj_buf = args.enc_chol_buf
    spectral_args.sims_buf = nil; spectral_args.row_buf = nil
    spectral_args.factor_buf = nil
    spectral_args.encoder = nil
    proj_shared = nil; sims_shared = nil; row_shared = nil -- luacheck: ignore
    lm_slot = nil
    xtx_slot = nil; xty_slot = nil; factor_slot = nil
    -- the pooled search encoder is dead once the search loop ends; free its C buffers now
    -- (it is the only encoder that survives across trials -- deploy/calibrate encoders are
    -- one-shot and GC'd normally)
    if enc_slot then enc_slot:destroy(); enc_slot = nil end
    search_fb = nil
    spectral_args.landmarks = nil
    ensure_shared_bufs()
    collectgarbage("collect")
  end
  local function release_cv ()
    if xtx_shared then xtx_shared:destroy() end
    if xty_shared then xty_shared:destroy() end
  end
  local function build_kd (spec, at_search)
    -- A/B perf test: per-trial GC disabled -- local tgc = tick("gc"); collectgarbage("collect"); tock(tgc)
    if args.rebuild and spec.params ~= nil then
      local trb = tick("rebuild"); local rb = args.rebuild(spec.params, at_search ~= nil); tock(trb)
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
    -- CV bases (search + calibration folds) draw landmarks ONLY from the never-val
    -- stratum: a fold basis containing its own val rows gives those rows near-free
    -- coefficients that poison val predictions at small lambda (val-landmark leak).
    -- The deployed fit draws from the full pool (basis choice after selection is fine).
    if at_search == "search" then
      if lm_slot == nil then
        lm_slot = spectral.uniform_landmarks(spectral_args, search_m, seed + 1000, args.stratum_rows)
        xtx_slot = fvec.create(search_m * search_m)
        xty_slot = fvec.create(search_m * nl_cap)
        -- pooled m*m kss/factor scratch: fully overwritten by landmark_kss each encode, so
        -- reuse is bit-identical. The per-trial gram borrows it as factor_ext (its solve
        -- scratch); gram:destroy at trial end drops the borrow, leaving this slot intact.
        factor_slot = fvec.create(search_m * search_m)
      end
      spectral_args.landmarks = lm_slot
      spectral_args.xtx_buf = xtx_slot
      spectral_args.xty_buf = xty_slot
      spectral_args.factor_buf = factor_slot
      -- pooled encoder: hand the previous trial's encoder back so encode overwrites its
      -- value/chol/csc buffers in place (trial-invariant shape: same lm_slot + block
      -- structure) instead of allocating a fresh userdata each trial. First trial (nil) builds
      -- it; thereafter it is reused and released once at end-of-search.
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
    local tse = tick("spectral.encode"); local _, sp_enc, gram = spectral.encode(spectral_args); tock(tse)
    if at_search == "search" then enc_slot = sp_enc end
    if tt and spectral_args.enc_phases then
      for name, dt in pairs(spectral_args.enc_phases) do prof_add_raw("~spectral/" .. name, dt) end
      spectral_args.enc_phases = nil
    end
    return { sp_enc = sp_enc, gram = gram }
  end
  -- CV downdate: ONE full-pool encode per trial/finalize also accumulates each fold's
  -- uncentered val moments and scatters its val codes (spectral fold_* args); fold grams
  -- are then assembled by subtraction (gram:fold) -- no per-fold encodes, no X slicing.
  -- All fold grams share one factor scratch (solves are sequential within a trial).
  -- build all splits' fold sets from ONE encode: assignments and slot tables are passed
  -- flattened set-major, matching the C accumulator layout
  -- one encode + fold-moment accumulation, then the K fold grams by subtraction (gram:fold)
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
  local function fold_bufs (nf, m, split, scratch)
    local mtx = require("santoku.mtx")
    local dvec = require("santoku.dvec")
    local counts = split.val_n
    local fb = { n = nf, counts = counts, assign = split.assign,
      xtx = {}, xty = {}, sv = {}, tv = {}, codes = {}, paths = {} }
    for f = 1, nf do
      fb.xtx[f] = fvec.create(m * m)
      fb.xty[f] = fvec.create(m * nl_cap)
      fb.sv[f] = fvec.create(m)
      fb.tv[f] = dvec.create(nl_cap)
      if scratch then
        fb.paths[f] = scratch .. ".fval" .. f .. ".bin"
        fb.codes[f] = mtx.create({ data = fvec.mmap_create(fb.paths[f], counts[f] * m),
          n_rows = counts[f], n_cols = m })
      else
        fb.codes[f] = mtx.create({ n_rows = counts[f], n_cols = m, type = "f32" })
      end
    end
    if scratch then
      fb.factor_path = scratch .. ".ffac.bin"
      fb.factor = fvec.mmap_create(fb.factor_path, m * m)
    else
      fb.factor = fvec.create(m * m)
    end
    return fb
  end
  local function fold_bufs_release (fb)
    for f = 1, fb.n do
      fb.xtx[f]:destroy(); fb.xty[f]:destroy()
      if fb.paths[f] then
        fb.codes[f]:data():destroy()
        os.remove(fb.paths[f])
      end
    end
    fb.factor:destroy()
    if fb.factor_path then os.remove(fb.factor_path) end
  end
  -- per-trial OOF decider over the fold split's kds (never rebuilds). The pooled fold
  -- cand/gold spans are trial-invariant (the split is fixed for the phase); clone + pool
  -- once and cache on the split. Serves both search scoring and deploy-rank calibration.
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
        local tr = tick("regress")
        local s = r:regress(kds[f].val_codes)
        tock(tr)
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
    -- Pooled per-trial buffers (shapes are trial-invariant): reuse across all trials so a long
    -- search doesn't leak a vocab*k CSR per fold each iteration (eurlex: 4270 labels, k=256 ->
    -- hundreds of MB/trial). Y (gold concat) never changes -> built once. fold_P reuse the
    -- r:label out-buffer; P is cleared + re-appended. Mirrors the span branch's _oof_pool.
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
      local tr = tick("regress")
      fold_P[f] = r:label(kds[f].val_codes, k, fold_P[f])
      tock(tr)
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
  -- shared fold evaluation: solve every fold gram at lam, then score -- pooled-OOF
  -- decider when the offset is being derived (span/multilabel, fns == nil), per-fold trial
  -- fns otherwise -- and aggregate mean/metrics. Used by both the search trials
  -- and the final-explore lambda trials.
  local function eval_folds (kds, lam, fns, split)
    local nf = #kds
    local mean, agg, decider = 0.0, {}, nil
    if not fns then
      for f = 1, nf do kds[f].gram:solve(lam) end
      local dec, m, foldsc, foldms = oof_decider(kds, split)
      decider = dec
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
    else
      for f = 1, nf do
        kds[f].gram:solve(lam)
        local sc, m = fns[f](kds[f])
        mean = mean + sc
        if m then for kk, vv in pairs(m) do if type(vv) == "number" then agg[kk] = (agg[kk] or 0) + vv end end end
      end
      mean = mean / nf
      for kk, vv in pairs(agg) do agg[kk] = vv / nf end
    end
    return mean, agg, decider
  end
  -- Finalize = deploy-rank OOF offset calibration (span/multilabel only) + the deployed
  -- fit. The calibration folds use the STRATUM basis (never-val landmarks -> no val-landmark
  -- poison of the offset); the shipped model uses the FULL-pool basis. lambda is the search/
  -- def lambda (no deploy-rank lambda re-optimization -- the rank gap measured ~0).
  local function calibrate_and_deploy (spec, params)
    local fin = nil
    if want_decode and (mode == "span" or mode == "multilabel") then
      local fb = fold_bufs(args.folds, args.n_landmarks, args.fold_split, args.scratch_path)
      local cal_kd, kds = build_folds(spec, fb, "cal")
      cal_kd.gram:release()
      cal_kd = nil -- luacheck: ignore
      collectgarbage("collect")
      local tc = tick("~calibrate")
      for f = 1, args.folds do kds[f].gram:solve(params.lambda) end
      local decider, metrics = oof_decider(kds, args.fold_split)
      tock(tc)
      kds = nil -- luacheck: ignore
      collectgarbage("collect")
      fold_bufs_release(fb)
      fin = { decider = decider, metrics = metrics }
    end
    local kd = build_kd(spec)
    local tsv = tick("~final_solve")
    kd.gram:solve(params.lambda, w_shared)
    tock(tsv)
    return kd, fin
  end
  local function finish (kd, params, solve, fin)
    local r = ridge.create({
      gram = kd.gram,
      w_buf = tiled and w_auto or nil,
    })
    kd.gram:release()
    if args.each then args.each({ event = "done", params = params, emb_d = kd.sp_enc:dims(),
      solve = solve }) end
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
    end
    return kname, base
  end
  local function spec_with (base, p)
    local spec = { params = p }
    for kk, vv in pairs(base) do spec[kk] = vv end
    return spec
  end
  local n_blocks = args.gauge_dims or (args.pool_blocks and #args.pool_blocks) or 1
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
            vec[i] = { min = spec.min, max = spec.max, log = spec.log }
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
        -- No box re-clamp on the scales: the ilr z-coords are already bounded by [-c, c]
        -- (c derived from box_max), which is the real constraint. A post-normalize clamp is
        -- both ineffective (geomean normalization reintroduces sub-box values) AND breaks
        -- idempotence -- a pinned/emitted scale below box_min would be re-clamped on ingest,
        -- shifting the whole gauge and making frozen 0/0 diverge from the search deploy.
        for j = 1, M do
          v[knob.active[j]] = math.exp(y[j])
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
        end
      end
    end
    return p
  end
  if not do_search then
    local _, base = center_spec()
    -- trials=0 resolves gauge/vector knobs through the SAME knob machinery as the search
    -- (ilr centers -> knob_value -> params_of gauge normalization). With the scale re-clamp
    -- removed this is now a true fixpoint: params_of(ilr_forward(log(D))) == D for any
    -- geomean-1 D, so a pinned def reproduces the search deploy exactly.
    local rk = {}
    for _, knob in ipairs(rebuild_knobs) do
      for _, nm in ipairs(knob.names) do local sm = knob.samplers[nm]; rk[nm] = sm and sm.center end
    end
    local rparams = has_knobs and params_of(rk) or nil
    local spec = spec_with(base, rparams)
    local lp = sample_params(label_samplers, label_names, nil, true)
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
      -- search_trials=1: no search; deploy the def config, calibrate offset at deploy rank
      kd, fin = calibrate_and_deploy(spec, params)
      sstr = "calibrate"
    end
    local sp_enc, r, vcodes, params_out, decider, dmetrics, bake = finish(kd, params, sstr, fin)
    release_cv()
    prof_emit()
    return sp_enc, r, vcodes, params_out, decider, dmetrics, bake
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
  local function run_folds (spec, lam)
    if not search_fb then
      search_fb = fold_bufs(nfolds, search_m, args.fold_split, nil)
    end
    local kd, kds = build_folds(spec, search_fb, "search")
    local meta = { dims = kd.sp_enc:dims() }
    kd.gram:release()
    local ts = tick("~fold_solve")
    local use_oof = decode_offset == nil and (mode == "span" or mode == "multilabel")
    local mean, agg = eval_folds(kds, lam, (not use_oof) and fold_trial_fns or nil,
      args.fold_split)
    tock(ts)
    -- search pooling: the per-trial gram shells are dead here. eval_folds has finished all
    -- fold regress/label reads (the fold ridges view the fold grams' W, so this must come
    -- AFTER eval_folds returns), so free the full gram shell + the K fold shells O(1) instead
    -- of letting them pile up for a GC heap walk. The pooled fvec slots (search_fb
    -- xtx/xty/factor, xtx_slot, xty_slot, factor_buf) are external to these shells and
    -- untouched; only per-shell scratch (Bcm, fold cm_t/y_mean, intercept, W) is freed.
    -- The encoder (kd.sp_enc == enc_slot) is POOLED across trials -- it is NOT destroyed here;
    -- next trial's build_kd hands it back to encode for in-place overwrite, and it is released
    -- once at end-of-search in release_enc_scratch.
    for f = 1, #kds do kds[f].ridge = nil; kds[f].gram:destroy() end
    kd.gram:destroy()
    return mean, agg, meta
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
  local worst_score = nil
  local bi = 0
  local function eval_kd (spec, base, gp)
    local ib = {}
    for kk, vv in pairs(base) do ib[kk] = vv end
    for _, n in ipairs(label_names) do ib[n] = gp[n] end
    local ok, sc, sm, meta = pcall(run_folds, spec, gp.lambda)
    local failed = not ok or sc == nil
    if failed then
      sc, sm, meta = (worst_score or -1e18), { failed = true }, meta or {}
    else
      worst_score = worst_score and num.min(worst_score, sc) or sc
    end
    -- failed trials (e.g. degenerate gauges with singular landmark kernels) keep a finite
    -- score but must NEVER become the incumbent
    local sel = failed and -num.huge or sc
    bi = bi + 1
    local improved = sel > best_score
    if improved then best_score = sel; best_params = ib end
    if args.each then
      args.each({ event = "trial", phase = "kernel", trial = bi, trials = btot,
        params = ib, score = sc, metrics = sm, emb_d = meta.dims,
        best = best_score, is_new_best = improved })
    end
    -- A/B perf test: per-trial GC disabled -- collectgarbage("collect")
    return sc, sm
  end
  local t_ks = tick("~kernel_search")
  local fam_runs = {}
  if families.matern and families.cosine then
    local kfam_vals, kseen = {}, {}
    for _, kn in ipairs(kernels) do
      local fam
      if kn == "cosine" then fam = "cosine"
      else fam = "matern" end
      if fam and not kseen[fam] then kseen[fam] = true; kfam_vals[#kfam_vals + 1] = fam end
    end
    args.kfam = kfam_vals
    fam_runs[#fam_runs + 1] = {
      names = { "kfam", "nu", "gamma" },
      samplers = { kfam = build_samplers(args, { "kfam" }, nil, seed).kfam,
        nu = kernel_samplers.nu, gamma = kernel_samplers.gamma },
      trials = args.search_trials or 0, tag = "kernel",
      base_of = function (gp)
        if gp.kfam == "cosine" then return { kernel = "cosine" } end
        return { kernel = "matern", nu = gp.nu, gamma = gp.gamma }
      end,
    }
  elseif families.matern then
    fam_runs[#fam_runs + 1] = {
      names = { "nu", "gamma" },
      samplers = { nu = kernel_samplers.nu, gamma = kernel_samplers.gamma },
      trials = args.search_trials or 0, tag = "kernel",
      base_of = function (gp) return { kernel = "matern", nu = gp.nu, gamma = gp.gamma } end,
    }
  elseif families.cosine then
    fam_runs[#fam_runs + 1] = {
      names = {},
      samplers = {},
      trials = args.search_trials or 0, tag = "kernel",
      base_of = function () return { kernel = "cosine" } end,
    }
  end
  for _, run in ipairs(fam_runs) do
    cmaes_search({
      param_names = with_knobs(run.names), samplers = merge_knob_samplers(run.samplers),
      trials = run.trials, skip_final = true, prof_tag = run.tag,
      trial_fn = function (gp)
        local p = params_of(gp)
        local base = run.base_of(gp)
        return eval_kd(spec_with(base, p), base_with(base, p), gp)
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
  -- deploy the search winner at its search lambda, offset calibrated at deploy rank
  local best_kd, fin = calibrate_and_deploy(best_params, best_params)
  local sp_enc, r, vcodes, params_out, decider, dmetrics, bake = finish(best_kd, best_params, "calibrate", fin)
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

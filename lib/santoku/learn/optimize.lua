local gp = require("santoku.learn.gp")

local dvec = require("santoku.dvec")
local num = require("santoku.num")
local err = require("santoku.error")
local rand = require("santoku.random")

local M = {}

-- plateaus(curve, n, klo, khi): Jenks natural breaks (Fisher-Jenks). Partition the curve values over
-- [klo, khi] into n contiguous classes minimizing total within-class variance (exact O(n*m^2) DP), and
-- return the COARSEST k of each class (its start = smallest k = fewest clusters = most general). The curve
-- is monotone in k, so value-classes are k-contiguous. 1-indexed; default range [2, #curve] (k=1 is the
-- degenerate whole-doc point). Returns the n sorted cut indices.
M.plateaus = function (curve, n, klo, khi)
  klo = klo or 2
  khi = khi or #curve
  local m = khi - klo + 1
  if n >= m then local out = {}; for k = klo, khi do out[#out + 1] = k end; return out end
  local s1, s2 = { [0] = 0 }, { [0] = 0 }       -- prefix sums for O(1) within-range SSD
  for i = 1, m do local v = curve[klo + i - 1]; s1[i] = s1[i - 1] + v; s2[i] = s2[i - 1] + v * v end
  local function ssd (a, b)                      -- sum of squared deviations from the mean over d[a..b]
    local c = b - a + 1; local sm = s1[b] - s1[a - 1]
    return (s2[b] - s2[a - 1]) - sm * sm / c
  end
  -- cost[c][i] = min total SSD partitioning d[1..i] into c classes; back[c][i] = start of the c-th class
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
    cuts[#cuts + 1] = klo + start - 1            -- coarsest k of the class (class start)
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

-- Merge a (possibly partial) user spec over a defaults table. nil => the full defaults (the param is
-- omitted entirely); a scalar => passthrough (a fixed, non-searched value); a table => fill any missing
-- min/max/log/def from defs. This is what lets callers omit min/max -- or the whole param -- and still
-- get a sensible searched range, since every search dim here is dimensionless/portable.
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

  -- A nested search (e.g. an inner lambda BO under an outer bandwidth BO) must NOT reseed the global
  -- RNG: the outer seeds once, and the inner reseeding to a fixed value each call would make the outer's
  -- candidate sampling identical every iteration. Pass reseed=false from the inner.
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
      end
      ei_buf = gp.suggest(X_obs, Y_obs, n_dims, cand_flat, n_candidates, n_hyper_restarts, ei_buf)
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

-- Default validation metric for a (gram, params) trial. The metric matches the decode that will be
-- applied, so model selection optimizes the real objective rather than a cheap proxy:
--   dense        -> MAE.
--   "single"     -> in-trial macro-F1 of argmax(score - per-label offset): solve a ridge at the trial's
--                   params, regress dense val scores, run the decide (single) calibration.
--   "fmeasure"   -> the multilabel micro-F1 global-threshold sweep.
--   nil          -> plain best-k label-F1 (the M.ridge default).
local function default_trial_fn (args, dense, metric, k)
  if dense then
    return function (g, params)
      local mae, nmae = g:regress_accuracy(params.lambda, nil, nil, args.val_targets)
      return -mae, { mae = mae, nmae = nmae }
    end
  end
  if metric == "single" or metric == "span" then
    local ridge = require("santoku.learn.ridge")
    local decide = require("santoku.learn.decide")
    local fvec = require("santoku.fvec")
    local nl, vn = args.n_labels, args.val_n_samples
    -- Reused across trials: one dense val-score buffer + one scratch decider, recalibrated in place each
    -- trial. The per-trial ridge solves W (carrying the intercept, so in-trial scores match the bundle).
    local scores = fvec.create(vn * nl)
    local sp = args.val_spans
    local probe = (metric == "span")
      and decide.create({ n_labels = nl, span = true, reject = args.reject })
      or decide.create({ n_labels = nl, single = true })
    local n_docs = sp and (sp.cand_offsets:size() - 1) or nil
    return function (g, params, kd)
      local rt = ridge.create({ gram = g, lambda = params.lambda,
        propensity_a = params.propensity_a, propensity_b = params.propensity_b })
      rt:regress((kd and kd.val_codes) or args.val_codes, vn, scores)
      if metric == "span" then
        local f1 = probe:calibrate({ scores = scores, n_samples = n_docs,
          cand_offsets = sp.cand_offsets, cand_starts = sp.cand_starts, cand_ends = sp.cand_ends,
          expected_offsets = sp.gold_offsets, expected_starts = sp.gold_starts,
          expected_ends = sp.gold_ends, expected_types = sp.gold_types })
        return f1, { span_f1 = f1 }
      end
      local macro = probe:calibrate({ scores = scores, n_samples = vn,
        expected_offsets = args.val_expected_offsets,
        expected_neighbors = args.val_expected_neighbors })
      return macro, { macro_f1 = macro }
    end
  end
  return function (g, params)
    local f1, p, r = g:label_accuracy(params.lambda, k,
      params.propensity_a, params.propensity_b,
      args.val_expected_offsets, args.val_expected_neighbors, metric)
    return f1, { f1 = f1, precision = p, recall = r }
  end
end

-- Append the label-head search dims to param_names (in order) and fill their default specs in args:
-- lambda always; propensity only for true multi-label (n_labels > 1), where it's a tail-label
-- reweighting -- it's a rank-invariant no-op for a single label, so we don't waste search dims on it.
local function add_label_params (param_names, args, dense)
  -- lambda is mean-eigenvalue-relative (mu = lambda * mean_eig), hence dimensionless/portable -- a single
  -- log range serves every dataset, so callers can omit it (or just override def).
  args.lambda = spec_defaults(args.lambda, { min = 1e-4, max = 16, log = true, def = 1.0 })
  param_names[#param_names + 1] = "lambda"
  if not dense and (args.n_labels or 0) > 1 then
    args.propensity_a = spec_defaults(args.propensity_a, { min = 0, max = 8.0, def = 0.5 })
    args.propensity_b = spec_defaults(args.propensity_b, { min = 0, max = 16.0, def = 1.5 })
    param_names[#param_names + 1] = "propensity_a"
    param_names[#param_names + 1] = "propensity_b"
  end
end

-- Decide ownership + shape of the decode, inferred (no objective knob). We own the decode only when the
-- caller gave no custom trial_fn (a custom trial_fn means a bespoke decode). Returns want_decode + a mode:
--   "span"       -- caller passed val_spans (candidate geometry + gold spans): nms_dp + REJECT offset.
--   "single"     -- >1 labels and every val gold set has exactly one: argmax + per-label offsets.
--   "multilabel" -- otherwise: global score threshold.
local function decode_mode (args, dense)
  if dense or args.trial_fn ~= nil then return false, nil end
  if args.val_spans then return true, "span" end
  if (args.n_labels or 0) > 1 and args.val_expected_offsets then
    local eo = args.val_expected_offsets
    for i = 0, args.val_n_samples - 1 do
      if eo:get(i + 1) - eo:get(i) ~= 1 then return true, "multilabel" end
    end
    return true, "single"
  end
  return true, "multilabel"
end

-- Calibrate the bundled decider on the final model's val predictions (one pass, not per-trial).
local function bundle_decider (r, val_codes, args, mode, k)
  if mode == "span" then
    local sp = args.val_spans
    local scores = r:regress(val_codes, args.val_n_samples)
    return M.decide({ n_labels = args.n_labels, reject = args.reject, val_scores = scores,
      val_n_samples = sp.cand_offsets:size() - 1,
      val_cand_offsets = sp.cand_offsets, val_cand_starts = sp.cand_starts, val_cand_ends = sp.cand_ends,
      val_expected_offsets = sp.gold_offsets, val_expected_starts = sp.gold_starts,
      val_expected_ends = sp.gold_ends, val_expected_types = sp.gold_types })
  end
  if mode == "single" then
    local scores = r:regress(val_codes, args.val_n_samples)
    return M.decide({ n_labels = args.n_labels, val_scores = scores,
      val_n_samples = args.val_n_samples,
      val_expected_offsets = args.val_expected_offsets,
      val_expected_neighbors = args.val_expected_neighbors })
  end
  local voff, vnbr, vsco = r:label(val_codes, args.val_n_samples, k)
  return M.decide({ n_labels = args.n_labels, val_offsets = voff, val_neighbors = vnbr,
    val_scores = vsco, val_n_samples = args.val_n_samples,
    val_expected_offsets = args.val_expected_offsets,
    val_expected_neighbors = args.val_expected_neighbors })
end

M.ridge = function (args)
  local ridge = require("santoku.learn.ridge")
  local dense = args.val_targets ~= nil
  -- Linear ridge over caller-supplied codes (linear probing of frozen, possibly nonlinear embeddings).
  -- No kernel / spectral embedding; otherwise parity with M.krr (auto-detected decode + bundled decider).
  local param_names = {}
  add_label_params(param_names, args, dense)
  local samplers = build_samplers(args, param_names)
  local k = not dense and (args.k or 32) or nil
  local want_decode, mode = decode_mode(args, dense)
  local locked = all_fixed(samplers) or not args.search_trials or args.search_trials <= 0
  local locked_params = locked and sample_params(samplers, param_names, nil, true) or nil
  local gram = args.gram
  if not gram then
    local train_codes = err.assert(args.train_codes, "train_codes required")
    local n_samples = err.assert(args.n_samples, "n_samples required")
    local n_dims = err.assert(args.n_dims, "n_dims required")
    local gram_args = {
      codes = train_codes, n_samples = n_samples, n_dims = n_dims,
      label_offsets = args.label_offsets, label_neighbors = args.label_neighbors,
      n_labels = args.n_labels,
      targets = args.targets, n_targets = args.n_targets,
    }
    -- Fixed lambda+propensity and we own the gram: bake W via Cholesky, skip eigendecomp.
    if locked then
      gram_args.solve_lambda = locked_params.lambda
      if not dense then
        gram_args.solve_propensity_a = locked_params.propensity_a
        gram_args.solve_propensity_b = locked_params.propensity_b
      end
    end
    gram = ridge.gram(gram_args)
  end
  -- We bake (Cholesky) only when locked AND we built the gram ourselves; a caller-supplied gram is eigen.
  local baked = locked and args.gram == nil
  -- Build the final head, bundle the calibrated decider (parity with M.krr), and return both. The bundle
  -- runs r:label/r:regress once on the final ridge -- no gram:prepare needed for the locked path.
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
  local dense = args.val_targets ~= nil
  local kernel_spec = args.kernel or "cosine"
  local kernels = type(kernel_spec) == "table" and kernel_spec or { kernel_spec }
  args.kernel = kernels
  -- inner label-head params (lambda [+ propensity] [+ extra]); searched per kernel on a fixed gram.
  -- args.extra = ordered pass-through search dims { { name=, min=, max=, log=, def= }, ... }: GP-searched
  -- and handed to trial_fn in params, but ignored by the gram/kernel logic (e.g. a Viterbi weight).
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
  local do_search = args.search_trials and args.search_trials > 0
  local k = not dense and (args.k or 32) or nil
  -- Classification always runs through the tiled gram/ridge (one tile when n_labels is small); tiling is
  -- purely a memory lever (tile_labels), decoupled from the decode. Dense regression is the only
  -- non-tiled path. The tiled ridge still materializes a full W, so single-label dense regress works.
  local tiled = not dense
  local tile_labels = tiled and (args.tile_labels or 1024) or nil
  local want_decode, mode = decode_mode(args, dense)
  local spectral_args = {
    offsets = args.offsets, tokens = args.tokens, values = args.values,
    n_tokens = args.n_tokens, n_samples = args.n_samples,
    codes = args.codes, d_input = args.d_input,
    n_landmarks = args.n_landmarks, trace_tol = args.trace_tol,
    label_offsets = args.label_offsets, label_neighbors = args.label_neighbors,
    n_labels = args.n_labels,
    targets = args.targets, n_targets = args.n_targets,
  }
  if tiled then
    spectral_args.tile_labels = tile_labels
    spectral_args.tile_samples = args.tile_samples
    spectral_args.chol_buf = args.chol_buf
  end
  -- Deterministic landmark selection: reseed the global RNG to a fixed value before every spectral.encode
  -- so a kernel's landmarks depend only on its inputs, not on build order / prior RNG state.
  local seed = args.seed or 1
  local function val_encode (sp_enc)
    if args.val_encode then return args.val_encode(sp_enc) end
    return sp_enc:encode({
      offsets = args.val_offsets, tokens = args.val_tokens,
      values = args.val_values, n_samples = args.val_n_samples,
    })
  end
  -- Build one kernel's eigendecomposed gram (no solve_lambda, no cache -- the outer loop visits each
  -- kernel exactly once). collectgarbage first so only ~one d x d gram (plus the running best) is resident.
  local function build_kd (kname)
    collectgarbage("collect")
    spectral_args.kernel = kname
    spectral_args.solve_lambda = nil
    spectral_args.solve_propensity_a = nil
    spectral_args.solve_propensity_b = nil
    if tiled and args.pqty_buf then
      spectral_args.pqty_buf = type(args.pqty_buf) == "function"
        and args.pqty_buf(kname) or args.pqty_buf
    end
    rand.fast_seed(seed)
    local _, sp_enc, gram = spectral.encode(spectral_args)
    local val_codes = val_encode(sp_enc)
    gram:prepare(val_codes, args.val_n_samples)
    return { sp_enc = sp_enc, gram = gram, val_codes = val_codes }
  end
  local function finish (kd, params, solve)
    local r = ridge.create({
      gram = kd.gram, lambda = params.lambda,
      propensity_a = not dense and params.propensity_a or nil,
      propensity_b = not dense and params.propensity_b or nil,
      w_buf = tiled and args.w_buf or nil,
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
  -- Locked path: first kernel + fixed lambda/propensity; bake W via Cholesky in spectral.encode (skip the
  -- eigendecomposition -- the gram is "baked" and ridge.create just copies W).
  if not do_search then
    local kname = kernels.def or kernels[1]
    local lp = sample_params(inner_samplers, inner_names, nil, true)
    collectgarbage("collect")
    spectral_args.kernel = kname
    spectral_args.solve_lambda = lp.lambda
    if not dense then
      spectral_args.solve_propensity_a = lp.propensity_a
      spectral_args.solve_propensity_b = lp.propensity_b
    end
    if tiled and args.pqty_buf then
      spectral_args.pqty_buf = type(args.pqty_buf) == "function"
        and args.pqty_buf(kname) or args.pqty_buf
    end
    rand.fast_seed(seed)
    local _, sp_enc, gram = spectral.encode(spectral_args)
    local val_codes = val_encode(sp_enc)
    local params = { kernel = kname }
    for _, n in ipairs(inner_names) do params[n] = lp[n] end
    return finish({ sp_enc = sp_enc, gram = gram, val_codes = val_codes }, params, "cholesky")
  end
  local trial_fn = args.trial_fn or default_trial_fn(args, dense, mode == "multilabel" and "fmeasure" or mode, k)
  -- trial_fn(gram, params, kd): kd = { sp_enc, gram, val_codes } so a custom trial can solve a ridge at
  -- the trial's (lambda, prop) and regress val_codes itself (e.g. to decode + score span-F1).
  local inner_trials = args.inner_trials or args.search_trials or 30
  -- Outer: a simple loop over the discrete kernel list, tracking the running best ACROSS kernels. Inner:
  -- a SILENT GP/BO over lambda/propensity/extra on that kernel's fixed eigenbasis (cheap closed-form
  -- tests). One log line per kernel reports its best with the global-best marker (the inner trials are
  -- hidden). The running-best kd is kept so the final reuses its gram without a rebuild.
  local best_kd, best_params, best_score = nil, nil, -num.huge
  for ki, kname in ipairs(kernels) do
    local kd = build_kd(kname)
    local _, ib = search({
      param_names = inner_names, samplers = inner_samplers,
      trials = inner_trials,
      trial_fn = function (p) return trial_fn(kd.gram, p, kd) end,
      skip_final = true,
    })
    ib.kernel = kname
    local sc, sm = trial_fn(kd.gram, ib, kd)
    if args.each then
      args.each({ event = "trial", phase = "kernel", trial = ki, trials = #kernels,
        params = ib, score = sc, metrics = sm,
        global_best_score = best_score, is_new_best = sc > best_score })
    end
    if sc > best_score then
      best_score, best_params, best_kd = sc, ib, kd
    end
  end
  return finish(best_kd, best_params, "eigen")
end

-- Decide what to do with ridge outputs. The decode is inferred from the input shape (no objective knob):
--   span: pass dense val_scores [n_cand*nl] + candidate geometry (val_cand_offsets/starts/ends) + gold
--     spans (val_expected_offsets/starts/ends/types) -> a REJECT decision offset fit by golden-section on
--     span-F1; decode is per-candidate argmax(score; REJECT-offset) resolved by nms_dp into typed spans.
--   multilabel: pass (val_offsets, val_neighbors, val_scores) from ridge:label -> one global score
--     threshold calibrated to maximize micro-F1; predict/score consume the same sparse ranked form.
--   single-label: pass dense val_scores [n*nl] from ridge:regress (no offsets) -> per-label additive
--     offsets fit by coordinate-ascent on macro-F1; decode is argmax(score_l - offset_l).
M.decide = function (args)
  local decide = require("santoku.learn.decide")
  if args.val_cand_offsets ~= nil then
    local g = decide.create({ n_labels = args.n_labels, span = true, reject = args.reject })
    local f1, precision, recall = g:calibrate({
      scores = args.val_scores, n_samples = args.val_n_samples,
      cand_offsets = args.val_cand_offsets, cand_starts = args.val_cand_starts, cand_ends = args.val_cand_ends,
      expected_offsets = args.val_expected_offsets, expected_starts = args.val_expected_starts,
      expected_ends = args.val_expected_ends, expected_types = args.val_expected_types,
    })
    return g, { span_f1 = f1, precision = precision, recall = recall, f1 = f1 }
  end
  local single = args.val_offsets == nil
  local g = decide.create({ n_labels = args.n_labels, single = single })
  if single then
    local macro_f1, accuracy = g:calibrate({
      scores = args.val_scores,
      n_samples = args.val_n_samples,
      expected_offsets = args.val_expected_offsets,
      expected_neighbors = args.val_expected_neighbors,
    })
    return g, { macro_f1 = macro_f1, accuracy = accuracy }
  end
  local best_f1, precision, recall = g:calibrate({
    offsets = args.val_offsets,
    neighbors = args.val_neighbors,
    scores = args.val_scores,
    n_samples = args.val_n_samples,
    expected_offsets = args.val_expected_offsets,
    expected_neighbors = args.val_expected_neighbors,
  })
  return g, { f1 = best_f1, precision = precision, recall = recall }
end

return M

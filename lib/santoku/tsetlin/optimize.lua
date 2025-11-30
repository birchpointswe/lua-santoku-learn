local tm = require("santoku.tsetlin.capi")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local itq = require("santoku.tsetlin.itq")
local ann = require("santoku.tsetlin.ann")
local evaluator = require("santoku.tsetlin.evaluator")
local num = require("santoku.num")
local str = require("santoku.string")
local err = require("santoku.error")
local rand = require("santoku.random")
local cvec = require("santoku.cvec")

local M = {}

M.elbow_methods = {
  max_gap = {},  -- uses absolute gap values
  otsu = {},
  lmethod = {},
  max_curvature = {},
  max_acceleration = {},
  first_gap = { alpha = { def = 5, min = 1, max = 50 } },
  plateau = { alpha = { def = 1, min = 0, max = 10 } },
  first_gap_ratio = { alpha = { def = 3, min = 1, max = 10 } },
  kneedle = { alpha = { def = 1, min = 0.1, max = 10, log = true } },
}

M.select_metrics = {
  variance = {},
  skewness = {},
  entropy = { n_bins = { def = 0, min = 0, max = 100, int = true } },  -- 0 = auto
  bimodality = {},
  dip = {},
}

M.sample_elbow = function (elbow_list, elbow_alpha_override)
  if type(elbow_list) ~= "table" or #elbow_list == 0 then
    local method = elbow_list
    if method == "none" then
      return "none", nil
    end
    local method_info = M.elbow_methods[method]
    local alpha = nil
    if method_info and method_info.alpha then
      local spec = (elbow_alpha_override and elbow_alpha_override[method]) or method_info.alpha
      alpha = spec.def
    end
    return method, alpha
  end
  local method = elbow_list[num.random(#elbow_list)]
  if method == "none" then
    return "none", nil
  end
  local method_info = M.elbow_methods[method]
  if not method_info then
    err.error("Unknown elbow method: " .. tostring(method))
  end
  local alpha = nil
  if method_info.alpha then
    local spec = (elbow_alpha_override and elbow_alpha_override[method]) or method_info.alpha
    if spec.log then
      local log_val = num.random() * (num.log(spec.max) - num.log(spec.min)) + num.log(spec.min)
      alpha = num.exp(log_val)
    else
      alpha = num.random() * (spec.max - spec.min) + spec.min
    end
  end
  return method, alpha
end

M.sample_metric = function (metric_list, metric_alpha_override)
  local metric
  if type(metric_list) ~= "table" or #metric_list == 0 then
    metric = metric_list
  else
    metric = metric_list[num.random(#metric_list)]
  end
  local metric_info = M.select_metrics[metric]
  if not metric_info then
    return metric, nil
  end
  local alpha = {}
  for param_name, param_spec in pairs(metric_info) do
    local spec = (metric_alpha_override and metric_alpha_override[metric] and metric_alpha_override[metric][param_name]) or param_spec
    local val
    if spec.log then
      local log_val = num.random() * (num.log(spec.max) - num.log(spec.min)) + num.log(spec.min)
      val = num.exp(log_val)
    else
      val = num.random() * (spec.max - spec.min) + spec.min
    end
    if spec.int then
      val = num.floor(val + 0.5)
    end
    alpha[param_name] = val
  end
  return metric, alpha
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
  if type(spec) == "table" and spec.def ~= nil then
    local def, minv, maxv = spec.def, spec.min, spec.max
    err.assert(def and minv and maxv, "range spec missing def|min|max")
    local is_log = not not spec.log
    local is_int = not not spec.int
    local is_pow2 = not not spec.pow2
    local span, base_center
    if is_log then
      span = num.log(maxv) - num.log(minv)
      base_center = num.log(def)
    else
      span = maxv - minv
      base_center = def
    end
    local init_jitter = (spec.dev or global_dev or 1.0) * span
    local jitter = init_jitter
    local round = 0
    return {
      type = "range",
      center = def,
      sample = function (center)
        local c = center and (is_log and num.log(center) or center) or base_center
        local x = rand.fast_normal(c, jitter)
        if is_log then
          x = num.exp(x)
        end
        if x < minv then
          x = minv
        elseif x > maxv then
          x = maxv
        end
        if is_pow2 then
          x = round_to_pow2(x)
          if x < minv then
            x = minv
          elseif x > maxv then
            x = maxv
          end
        elseif is_int then
          x = num.floor(x + 0.5)
        end
        return x
      end,
      shrink = function ()
        round = round + 1
        jitter = init_jitter / num.sqrt(round + 1)
      end,
      reset = function ()
        round = 0
        jitter = init_jitter
      end,
    }
  end
  if type(spec) == "table" and #spec > 0 then
    return {
      type = "list",
      center = spec[1],
      sample = function ()
        return spec[num.random(#spec)]
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

M.sample_params = function (samplers, param_names)
  local p = {}
  for _, name in ipairs(param_names) do
    local s = samplers[name]
    if s then
      if s.type == "range" then
        p[name] = s.sample(s.center)
      else
        p[name] = s.sample()
      end
    end
  end
  return p
end

M.sample_tier = function (tier_cfg)
  local result = {}
  for k, v in pairs(tier_cfg) do
    if type(v) == "table" and v.def ~= nil then
      local minv, maxv = v.min, v.max
      local is_log = v.log
      local is_int = v.int
      local is_pow2 = v.pow2
      local x
      if is_log then
        local log_val = num.random() * (num.log(maxv) - num.log(minv)) + num.log(minv)
        x = num.exp(log_val)
      else
        x = num.random() * (maxv - minv) + minv
      end
      if is_pow2 then
        x = round_to_pow2(x)
        if x < minv then x = minv elseif x > maxv then x = maxv end
      elseif is_int then
        x = num.floor(x + 0.5)
      end
      result[k] = x
    elseif type(v) == "table" and #v > 0 then
      result[k] = v[num.random(#v)]
    else
      result[k] = v
    end
  end
  return result
end

M.tier_all_fixed = function (tier_cfg)
  for _, v in pairs(tier_cfg) do
    if type(v) == "table" and v.def ~= nil then
      return false
    end
    if type(v) == "table" and #v > 1 then
      return false
    end
  end
  return true
end

M.all_fixed = function (samplers)
  for _, s in pairs(samplers) do
    if s and s.type ~= "fixed" then
      return false
    end
  end
  return true
end

M.recenter_and_shrink = function (samplers, param_names, best_params)
  for _, pname in ipairs(param_names) do
    local s = samplers[pname]
    if s and s.type == "range" and best_params[pname] then
      s.center = best_params[pname]
      if s.shrink then
        s.shrink()
      end
    end
  end
end

M.search = function (args)

  local param_names = err.assert(args.param_names, "param_names required")
  local samplers = err.assert(args.samplers, "samplers required")
  local trial_fn = err.assert(args.trial_fn, "trial_fn required")
  local rounds = args.rounds or 3
  local trials = args.trials or 10
  local tolerance = args.tolerance or 1e-8
  local make_key = args.make_key
  local each_cb = args.each
  local cleanup_fn = args.cleanup
  local skip_final = args.skip_final
  local best_score = -num.huge
  local best_params = nil
  local best_result = nil
  local best_metrics = nil

  if M.all_fixed(samplers) or rounds <= 0 or trials <= 0 then
    best_params = M.sample_params(samplers, param_names)
    if skip_final then
      return nil, best_params, nil
    else
      local _, metrics, result = trial_fn(best_params, { is_final = true })
      return result, best_params, metrics
    end
  end

  for r = 1, rounds do
    local seen = {}
    local round_best_score = -num.huge
    local round_best_params = nil
    for t = 1, trials do
      local params = M.sample_params(samplers, param_names)
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
        if score > best_score + tolerance then
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
    end
    if round_best_params then
      M.recenter_and_shrink(samplers, param_names, round_best_params)
    end
    if each_cb then
      each_cb({
        event = "round",
        round = r,
        round_best_score = round_best_score,
        round_best_params = round_best_params,
        global_best_score = best_score,
        global_best_params = best_params,
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
      clauses = 8,
      clause_tolerance = 8,
      clause_maximum = 8,
      target = 4,
      specificity = 1000,
      reusable = true,
    })
  elseif typ == "classifier" then
    return tm.create("classifier", {
      features = args.features,
      classes = args.classes,
      negative = args.negative,
      clauses = 8,
      clause_tolerance = 8,
      clause_maximum = 8,
      target = 4,
      specificity = 1000,
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
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
    })
  elseif typ == "classifier" then
    return tm.create("classifier", {
      features = args.features,
      classes = args.classes,
      negative = args.negative,
      clauses = params.clauses,
      clause_tolerance = params.clause_tolerance,
      clause_maximum = params.clause_maximum,
      target = params.target,
      specificity = params.specificity,
    })
  else
    err.error("unexpected type", typ)
  end
end

local function train_tm (typ, tmobj, args, params, iterations, early_patience, tolerance, metric_fn, each_cb, info)
  local best_epoch_score = -num.huge
  local last_epoch_score = -num.huge
  local last_metrics = nil
  local epochs_since_improve = 0
  local checkpoint = (early_patience and early_patience > 0) and cvec.create(0) or nil
  local has_checkpoint = false

  local function on_epoch (epoch)
    local score, metrics = metric_fn(tmobj)
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
    if score > best_epoch_score + tolerance then
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
      iterations = iterations,
      each = on_epoch,
    })
  elseif typ == "classifier" then
    tmobj:train({
      samples = args.samples,
      problems = args.problems,
      solutions = args.solutions,
      iterations = iterations,
      each = on_epoch,
    })
  end

  if checkpoint and has_checkpoint then
    tmobj:restore(checkpoint)
    last_epoch_score, last_metrics = metric_fn(tmobj)
  end

  return last_epoch_score, last_metrics
end

local function optimize_tm (args, typ)

  local patience = args.search_patience or 10
  local use_early_stop = patience > 0
  local final_patience = args.final_patience or 40
  local use_final_early_stop = final_patience > 0
  local tolerance = args.search_tolerance or 1e-8
  local iters_search = args.search_iterations or 10
  local final_iters = args.final_iterations or (iters_search * 10)
  local global_dev = args.search_dev or 1.0
  local metric_fn = err.assert(args.search_metric, "search_metric required")
  local each_cb = args.each

  local param_names = { "clauses", "clause_tolerance", "clause_maximum", "target", "specificity" }
  local samplers = M.build_samplers(args, param_names, global_dev)

  local search_tm
  if not M.all_fixed(samplers) then
    search_tm = create_tm(typ, args)
  end

  local function search_trial_fn (params, info)
    if params.clause_tolerance and params.clause_maximum and params.clause_tolerance > params.clause_maximum then
      params.clause_tolerance, params.clause_maximum = params.clause_maximum, params.clause_tolerance
    end
    search_tm:reconfigure(params)
    local score, metrics = train_tm(typ, search_tm, args, params, iters_search,
      use_early_stop and patience or nil, tolerance, metric_fn, each_cb, info)
    return score, metrics, nil
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    rounds = args.search_rounds or 3,
    trials = args.search_trials or 10,
    tolerance = tolerance,
    trial_fn = search_trial_fn,
    skip_final = true,
    make_key = function (p)
      return str.format("%d|%d|%d|%d|%d",
        p.clauses,
        p.clause_tolerance,
        p.clause_maximum,
        num.floor(p.target * 1e6 + 0.5),
        num.floor(p.specificity * 1e6 + 0.5))
    end,
  })

  if search_tm then
    search_tm:destroy()
  end

  local final_tm = create_final_tm(typ, args, best_params)
  local _, final_metrics = train_tm(typ, final_tm, args, best_params, final_iters,
    use_final_early_stop and final_patience or nil, tolerance, metric_fn, each_cb, { is_final = true })
  collectgarbage("collect")
  return final_tm, final_metrics
end

M.classifier = function (args)
  return optimize_tm(args, "classifier")
end

M.encoder = function (args)
  return optimize_tm(args, "encoder")
end

M.destroy_spectral = function (model)
  if not model then return end
  if model.index then model.index:destroy() end
  -- Note: model.ids is NOT destroyed here because spectral.encode may return
  -- the same vector that was passed as input (adj_ids). Destroying it would
  -- corrupt the adjacency vectors used by subsequent spectral builds.
  -- The adj_ids is destroyed separately when model.adj_ids is set.
  if model.codes then model.codes:destroy() end
  if model.eigs then model.eigs:destroy() end
  if model.adj_ids then model.adj_ids:destroy() end
  if model.adj_offsets then model.adj_offsets:destroy() end
  if model.adj_neighbors then model.adj_neighbors:destroy() end
  if model.adj_weights then model.adj_weights:destroy() end
end

M.build_adjacency = function (args)
  local index = args.index
  local knn_index = args.knn_index or index
  local p = args.params
  local each = args.each

  return graph.adjacency({
    weight_index = index,
    weight_cmp = p.cmp,
    weight_decay = p.weight_decay,
    knn_index = knn_index,
    knn = p.knn,
    knn_mode = p.knn_mode,
    knn_alpha = p.knn_alpha,
    knn_mutual = p.knn_mutual,
    knn_cache = p.knn_cache or 32,
    bridge = p.bridge,
    each = each,
  })
end

M.build_spectral_from_adjacency = function (args)
  local adj_ids, adj_offsets, adj_neighbors, adj_weights = args.adj_ids, args.adj_offsets, args.adj_neighbors, args.adj_weights
  local p = args.params
  local bucket_size = args.bucket_size
  local each = args.each

  local ids, codes, _, eigs = spectral.encode({
    type = p.laplacian or "unnormalized",
    n_hidden = p.n_dims,
    eps = p.eps or 1e-10,
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    weights = adj_weights,
    each = each,
  })

  local dims = p.n_dims

  if p.select_elbow and p.select_elbow ~= "none" then
    local metric = p.select_metric or "entropy"
    local malpha = p.select_metric_alpha or {}
    local escores
    if metric == "entropy" then
      escores = codes:mtx_top_entropy(dims, malpha.n_bins)
    elseif metric == "variance" then
      escores = codes:mtx_top_variance(dims)
    elseif metric == "skewness" then
      escores = codes:mtx_top_skewness(dims)
    elseif metric == "bimodality" then
      escores = codes:mtx_top_bimodality(dims)
    elseif metric == "dip" then
      escores = codes:mtx_top_dip(dims)
    else
      err.error("Unknown select_metric: " .. tostring(metric))
    end
    local kept = escores:scores_elbow(p.select_elbow, p.select_elbow_alpha)
    codes:mtx_select(kept, nil, dims)
    dims = kept:size()
    escores:destroy()
  end

  local threshold_fn = p.threshold or function (c, d)
    return itq.median({ codes = c, n_dims = d })
  end
  codes = threshold_fn(codes, dims)

  local ann_index = ann.create({
    expected_size = ids:size(),
    bucket_size = bucket_size,
    features = dims,
  })
  ann_index:add(codes, ids)

  return {
    ids = ids,
    codes = codes,
    eigs = eigs,
    dims = dims,
    index = ann_index,
  }
end

M.score_spectral_eval = function (args)
  local model = args.model
  local eval_params = args.eval_params
  local expected = args.expected
  local knn_target_ids = args.knn_target_ids
  local bucket_size = args.bucket_size

  local knn_index = model.index
  if knn_target_ids then
    knn_index = ann.create({
      expected_size = knn_target_ids:size(),
      bucket_size = bucket_size,
      features = model.dims,
    })
    knn_index:add(model.index:get(knn_target_ids), knn_target_ids)
  end

  local adj_retrieved_ids, adj_retrieved_offsets, adj_retrieved_neighbors, adj_retrieved_weights =
    graph.adjacency({
      weight_index = model.index,
      knn_index = knn_index,
      knn_query_ids = expected.ids,
      knn_query_codes = model.index:get(expected.ids),
      knn = eval_params.knn,
      bridge = "none",
    })

  local stats = evaluator.score_retrieval({
    retrieved_ids = adj_retrieved_ids,
    retrieved_offsets = adj_retrieved_offsets,
    retrieved_neighbors = adj_retrieved_neighbors,
    retrieved_weights = adj_retrieved_weights,
    expected_ids = expected.ids,
    expected_offsets = expected.offsets,
    expected_neighbors = expected.neighbors,
    expected_weights = expected.weights,
    ranking = eval_params.ranking,
    metric = eval_params.metric,
    elbow = eval_params.elbow,
    elbow_alpha = eval_params.elbow_alpha,
    n_dims = model.dims,
  })

  if adj_retrieved_ids then adj_retrieved_ids:destroy() end
  if adj_retrieved_offsets then adj_retrieved_offsets:destroy() end
  if adj_retrieved_neighbors then adj_retrieved_neighbors:destroy() end
  if adj_retrieved_weights then adj_retrieved_weights:destroy() end
  if knn_target_ids and knn_index then knn_index:destroy() end

  local target = eval_params.target or "f1"
  local target_score = stats[target]

  return target_score, {
    score = stats.score,
    quality = stats.quality,
    recall = stats.recall,
    f1 = stats.f1,
  }
end

M.spectral = function (args)
  local index = args.index
  local knn_index = args.knn_index
  local bucket_size = args.bucket_size
  local each_cb = args.each
  local adjacency_each = args.adjacency_each
  local spectral_each = args.spectral_each
  local knn_target_ids = args.knn_target_ids

  local adjacency_cfg = err.assert(args.adjacency, "adjacency config required")
  local spectral_cfg = err.assert(args.spectral, "spectral config required")
  local eval_cfg = err.assert(args.eval, "eval config required")

  local adjacency_samples = args.adjacency_samples or 1
  local spectral_samples = args.spectral_samples or 1
  local eval_samples = args.eval_samples or 1

  local expected = {
    ids = args.expected_ids,
    offsets = args.expected_offsets,
    neighbors = args.expected_neighbors,
    weights = args.expected_weights,
  }

  local seen = {}
  local best_score = -num.huge
  local best_params = nil
  local best_model = nil
  local best_metrics = nil
  local sample_count = 0

  local adj_fixed = M.tier_all_fixed(adjacency_cfg)
  local spec_fixed = M.tier_all_fixed(spectral_cfg)
    and (type(spectral_cfg.select_elbow) ~= "table" or #spectral_cfg.select_elbow <= 1)
    and (type(spectral_cfg.select_metric) ~= "table" or #spectral_cfg.select_metric <= 1)
  local eval_fixed = M.tier_all_fixed(eval_cfg) and (type(eval_cfg.elbow) ~= "table" or #eval_cfg.elbow <= 1)

  local function make_adj_key (p)
    return str.format("%s|%s|%s|%s",
      tostring(p.knn), tostring(p.knn_alpha), tostring(p.weight_decay), tostring(p.knn_mutual))
  end

  local function make_spec_key (adj_key, p)
    local select_str
    if p.select_elbow then
      local metric_alpha_str = ""
      if p.select_metric_alpha then
        for k, v in pairs(p.select_metric_alpha) do
          metric_alpha_str = metric_alpha_str .. str.format("|%s=%.2f", k, v)
        end
      end
      select_str = str.format("%s%s|%s|%.2f", p.select_metric or "entropy", metric_alpha_str, p.select_elbow, p.select_elbow_alpha or 0)
    else
      select_str = "none"
    end
    return str.format("%s|%s|%s|%s", adj_key, tostring(p.n_dims), tostring(p.laplacian), select_str)
  end

  local function make_eval_key (spec_key, p)
    local alpha_str = p.elbow_alpha and str.format("%.2f", p.elbow_alpha) or "nil"
    return str.format("%s|%s|%s|%s", spec_key, p.elbow, alpha_str, p.ranking)
  end

  for _ = 1, (adj_fixed and 1 or adjacency_samples) do
    local adj_params = M.sample_tier(adjacency_cfg)
    local adj_key = make_adj_key(adj_params)

    if not seen[adj_key] then
      seen[adj_key] = true
      sample_count = sample_count + 1

      if each_cb then
        each_cb({
          event = "stage",
          stage = "adjacency",
          sample = sample_count,
          is_final = false,
          params = { adjacency = adj_params },
        })
      end

      local adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
        index = index,
        knn_index = knn_index,
        params = adj_params,
        each = adjacency_each,
      })

      for _ = 1, (spec_fixed and 1 or spectral_samples) do
        local spec_params = M.sample_tier(spectral_cfg)
        if spectral_cfg.select_elbow then
          local select_elbow, select_elbow_alpha = M.sample_elbow(spectral_cfg.select_elbow, spectral_cfg.select_elbow_alpha)
          spec_params.select_elbow = select_elbow
          spec_params.select_elbow_alpha = select_elbow_alpha
          local select_metric, select_metric_alpha = M.sample_metric(spectral_cfg.select_metric, spectral_cfg.select_metric_alpha)
          spec_params.select_metric = select_metric or "entropy"
          spec_params.select_metric_alpha = select_metric_alpha
        end
        local spec_key = make_spec_key(adj_key, spec_params)

        if not seen[spec_key] then
          seen[spec_key] = true

          if each_cb then
            each_cb({
              event = "stage",
              stage = "spectral",
              sample = sample_count,
              is_final = false,
              params = { adjacency = adj_params, spectral = spec_params },
            })
          end

          local model = M.build_spectral_from_adjacency({
            adj_ids = adj_ids,
            adj_offsets = adj_offsets,
            adj_neighbors = adj_neighbors,
            adj_weights = adj_weights,
            params = spec_params,
            bucket_size = bucket_size,
            each = spectral_each,
          })

          local model_is_best = false

          for _ = 1, (eval_fixed and 1 or eval_samples) do
            local eval_params = M.sample_tier(eval_cfg)
            local elbow, elbow_alpha = M.sample_elbow(eval_cfg.elbow, eval_cfg.elbow_alpha)
            eval_params.elbow = elbow
            eval_params.elbow_alpha = elbow_alpha

            local eval_key = make_eval_key(spec_key, eval_params)

            if not seen[eval_key] then
              seen[eval_key] = true

              local score, metrics = M.score_spectral_eval({
                model = model,
                eval_params = eval_params,
                expected = expected,
                knn_target_ids = knn_target_ids,
                bucket_size = bucket_size,
              })

              if each_cb then
                each_cb({
                  event = "eval",
                  sample = sample_count,
                  is_final = false,
                  params = { adjacency = adj_params, spectral = spec_params, eval = eval_params },
                  score = score,
                  metrics = metrics,
                })
              end

              if score > best_score then
                if best_model then
                  M.destroy_spectral(best_model)
                end
                best_score = score
                best_params = { adjacency = adj_params, spectral = spec_params, eval = eval_params }
                best_model = model
                best_metrics = metrics
                model_is_best = true
              end
            end
          end

          if not model_is_best then
            M.destroy_spectral(model)
          end
        end
      end

      if adj_ids then adj_ids:destroy() end
      if adj_offsets then adj_offsets:destroy() end
      if adj_neighbors then adj_neighbors:destroy() end
      if adj_weights then adj_weights:destroy() end
      collectgarbage("collect")
    end
  end

  if best_model then
    if each_cb then
      each_cb({
        event = "stage",
        stage = "adjacency",
        is_final = true,
        params = best_params,
      })
    end

    local adj_ids, adj_offsets, adj_neighbors, adj_weights = M.build_adjacency({
      index = index,
      knn_index = knn_index,
      params = best_params.adjacency,
      each = adjacency_each,
    })

    M.destroy_spectral(best_model)

    if each_cb then
      each_cb({
        event = "stage",
        stage = "spectral",
        is_final = true,
        params = best_params,
      })
    end

    best_model = M.build_spectral_from_adjacency({
      adj_ids = adj_ids,
      adj_offsets = adj_offsets,
      adj_neighbors = adj_neighbors,
      adj_weights = adj_weights,
      params = best_params.spectral,
      bucket_size = bucket_size,
      each = spectral_each,
    })

    best_model.adj_ids = adj_ids
    best_model.adj_offsets = adj_offsets
    best_model.adj_neighbors = adj_neighbors
    best_model.adj_weights = adj_weights
  end

  collectgarbage("collect")
  return best_model, best_params, best_metrics
end

return M

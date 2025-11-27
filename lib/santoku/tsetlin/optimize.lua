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
      local score, metrics, result = trial_fn(best_params, { is_final = true })
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
      best_result = nil
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

M.spectral_param_names = {
  "knn",
  "knn_alpha",
  "knn_mutual",
  "category_knn",
  "category_knn_decay",
  "weight_cmp",
  "weight_alpha",
  "weight_beta",
  "weight_decay",
  "knn_cmp",
  "knn_beta",
  "knn_decay",
  "category_cmp",
  "category_alpha",
  "category_beta",
  "category_decay",
}

M.build_spectral = function (args)

  local index = args.index
  local knn_index = args.knn_index or index
  local category_index = args.category_index
  local cfg = args.cfg
  local each = args.each
  if each then
    each("adjacency")
  end

  local adj_ids, adj_offsets, adj_neighbors, adj_weights = graph.adjacency({
    weight_index = index,
    weight_cmp = cfg.weight_cmp,
    weight_alpha = cfg.weight_alpha,
    weight_beta = cfg.weight_beta,
    weight_decay = cfg.weight_decay,
    knn_index = knn_index,
    knn = cfg.knn,
    knn_mode = cfg.knn_mode,
    knn_alpha = cfg.knn_alpha,
    knn_mutual = cfg.knn_mutual,
    knn_cmp = cfg.knn_cmp,
    knn_beta = cfg.knn_beta,
    knn_decay = cfg.knn_decay,
    knn_cache = cfg.knn_cache or 32,
    knn_min = cfg.knn_min,
    knn_rank = cfg.knn_rank,
    knn_eps = cfg.knn_eps,
    knn_query_ids = cfg.knn_query_ids,
    knn_query_codes = cfg.knn_query_codes,
    category_index = category_index,
    category_knn = cfg.category_knn,
    category_knn_decay = cfg.category_knn_decay,
    category_cmp = cfg.category_cmp,
    category_alpha = cfg.category_alpha,
    category_beta = cfg.category_beta,
    category_decay = cfg.category_decay,
    category_anchors = cfg.category_anchors,
    bipartite_ids = cfg.bipartite_ids,
    bipartite_features = cfg.bipartite_features,
    bipartite_nodes = cfg.bipartite_nodes,
    bipartite_dims = cfg.bipartite_dims,
    seed_ids = cfg.seed_ids,
    seed_offsets = cfg.seed_offsets,
    seed_neighbors = cfg.seed_neighbors,
    random_pairs = cfg.random_pairs,
    bridge = cfg.bridge,
    each = cfg.adjacency_each,
  })

  if each then
    each("spectral")
  end

  local ids, codes, _, eigs = spectral.encode({
    type = cfg.laplacian or "unnormalized",
    n_hidden = cfg.n_dims,
    eps = cfg.eps or 1e-10,
    ids = adj_ids,
    offsets = adj_offsets,
    neighbors = adj_neighbors,
    weights = adj_weights,
    each = cfg.spectral_each,
  })

  local dims = cfg.n_dims

  if cfg.select then
    if each then each("select") end
    local kept = cfg.select(codes, eigs, ids:size(), dims)
    codes:mtx_select(kept, nil, dims)
    dims = kept:size()
  end

  if each then
    each("threshold")
  end

  local threshold_fn = cfg.threshold or function (c, d)
    return itq.median({ codes = c, n_dims = d })
  end
  codes = threshold_fn(codes, dims)

  if each then
    each("index")
  end

  local ann_index = ann.create({
    expected_size = ids:size(),
    bucket_size = cfg.bucket_size,
    features = dims,
  })
  ann_index:add(codes, ids)

  return {
    ids = ids,
    codes = codes,
    eigs = eigs,
    dims = dims,
    index = ann_index,
    adj_ids = adj_ids,
    adj_offsets = adj_offsets,
    adj_neighbors = adj_neighbors,
    adj_weights = adj_weights,
  }
end

M.evaluate_spectral_retrieval = function (args)

  local model = args.model
  local query_ids = args.query_ids
  local target_ids = args.target_ids
  local ground_truth = args.ground_truth
  local cfg = args.cfg or {}

  local index_targets = ann.create({
    features = model.dims,
    expected_size = target_ids:size(),
    bucket_size = cfg.bucket_size,
  })
  index_targets:add(model.index:get(target_ids), target_ids)

  local adj_retrieved_ids,
        adj_retrieved_offsets,
        adj_retrieved_neighbors,
        adj_retrieved_weights = graph.adjacency({
    weight_index = model.index,
    knn_index = index_targets,
    knn_query_ids = query_ids,
    knn_query_codes = model.index:get(query_ids),
    knn = cfg.eval_knn or 32,
    bridge = "none",
  })

  local stats = evaluator.score_retrieval({
    retrieved_ids = adj_retrieved_ids,
    retrieved_offsets = adj_retrieved_offsets,
    retrieved_neighbors = adj_retrieved_neighbors,
    retrieved_weights = adj_retrieved_weights,
    expected_ids = ground_truth.ids,
    expected_offsets = ground_truth.offsets,
    expected_neighbors = ground_truth.neighbors,
    expected_weights = ground_truth.weights,
    metric = cfg.retrieval_metric or "min",
    n_dims = model.dims,
  })

  local margin_fn = cfg.retrieval_margin or function (quality)
    local _, idx = quality:scores_plateau(cfg.tolerance or 1e-4)
    return idx - 1
  end
  local best_margin = margin_fn(stats.quality)
  local quality = stats.quality:get(best_margin)
  local recall = stats.recall:get(best_margin)

  index_targets:destroy()
  adj_retrieved_ids:destroy()
  adj_retrieved_offsets:destroy()
  adj_retrieved_neighbors:destroy()
  adj_retrieved_weights:destroy()
  stats.quality:destroy()
  stats.recall:destroy()
  stats.f1:destroy()

  return {
    best_margin = best_margin,
    quality = quality,
    recall = recall,
  }
end

M.evaluate_spectral_entropy = function (args)
  local model = args.model
  return evaluator.entropy_stats(model.codes, model.ids:size(), model.dims)
end

M.destroy_spectral = function (model)
  if not model then return end
  if model.index then model.index:destroy() end
  if model.ids then model.ids:destroy() end
  if model.codes then model.codes:destroy() end
  if model.eigs then model.eigs:destroy() end
  if model.adj_ids then model.adj_ids:destroy() end
  if model.adj_offsets then model.adj_offsets:destroy() end
  if model.adj_neighbors then model.adj_neighbors:destroy() end
  if model.adj_weights then model.adj_weights:destroy() end
end

M.spectral = function (args)
  local index = args.index
  local knn_index = args.knn_index
  local category_index = args.category_index
  local query_ids = args.query_ids
  local target_ids = args.target_ids
  local ground_truth = args.ground_truth
  local param_names = args.param_names or M.spectral_param_names
  local samplers = M.build_samplers(args, param_names, args.search_dev or 0.2)
  local each_cb = args.each

  local base_cfg = {
    n_dims = args.n_dims or 64,
    laplacian = args.laplacian or "unnormalized",
    eps = args.eps or 1e-10,
    bucket_size = args.bucket_size,
    knn_cache = args.knn_cache or 32,
    knn_mode = args.knn_mode,
    bridge = args.bridge,
    select = args.select,
    threshold = args.threshold,
    tolerance = args.tolerance,
    retrieval_metric = args.retrieval_metric,
    retrieval_margin = args.retrieval_margin,
    eval_knn = args.eval_knn,
    knn_min = args.knn_min,
    knn_rank = args.knn_rank,
    knn_eps = args.knn_eps,
    knn_query_ids = args.knn_query_ids,
    knn_query_codes = args.knn_query_codes,
    category_anchors = args.category_anchors,
    bipartite_ids = args.bipartite_ids,
    bipartite_features = args.bipartite_features,
    bipartite_nodes = args.bipartite_nodes,
    bipartite_dims = args.bipartite_dims,
    seed_ids = args.seed_ids,
    seed_offsets = args.seed_offsets,
    seed_neighbors = args.seed_neighbors,
    random_pairs = args.random_pairs,
    adjacency_each = args.adjacency_each,
    spectral_each = args.spectral_each,
  }

  local metric_fn = args.search_metric
  if not metric_fn then
    if ground_truth and query_ids and target_ids then
      metric_fn = function (model, cfg)
        local ret = M.evaluate_spectral_retrieval({
          model = model,
          query_ids = query_ids,
          target_ids = target_ids,
          ground_truth = ground_truth,
          cfg = cfg,
        })
        return ret.quality, {
          quality = ret.quality,
          recall = ret.recall,
          best_margin = ret.best_margin,
        }
      end
    else
      metric_fn = function (model)
        local ent = M.evaluate_spectral_entropy({ model = model })
        return ent.mean, { entropy = ent }
      end
    end
  end

  local function make_cfg (params)
    local cfg = {}
    for k, v in pairs(base_cfg) do
      cfg[k] = v
    end
    for k, v in pairs(params) do
      cfg[k] = v
    end
    return cfg
  end

  local function build_and_eval (params, info)
    local cfg = make_cfg(params)

    local model = M.build_spectral({
      index = index,
      knn_index = knn_index,
      category_index = category_index,
      cfg = cfg,
      each = each_cb and function (stage)
        each_cb({
          event = "stage",
          stage = stage,
          round = info.round,
          trial = info.trial,
          is_final = info.is_final,
          params = params,
        })
      end or nil,
    })

    local score, metrics = metric_fn(model, cfg)

    if each_cb then
      each_cb({
        event = "eval",
        round = info.round,
        trial = info.trial,
        is_final = info.is_final,
        params = params,
        score = score,
        metrics = metrics,
      })
    end

    return model, score, metrics
  end

  local _, best_params, _ = M.search({
    param_names = param_names,
    samplers = samplers,
    rounds = args.search_rounds or 3,
    trials = args.search_trials or 10,
    tolerance = args.search_tolerance or 1e-6,
    skip_final = true,

    trial_fn = function (params, info)
      local model, score, metrics = build_and_eval(params, info)
      M.destroy_spectral(model)
      collectgarbage("collect")
      return score, metrics, nil
    end,

    make_key = function (p)
      return str.format("%d|%s|%s|%s|%s|%s|%s",
        p.knn or 0,
        tostring(p.knn_alpha),
        tostring(p.knn_mutual),
        tostring(p.weight_cmp),
        tostring(p.knn_cmp),
        tostring(p.category_knn),
        tostring(p.category_cmp))
    end,

    each = each_cb and function (info)
      if info.event == "round" or info.event == "final_start" or info.event == "final_end" then
        each_cb(info)
      end
    end or nil,
  })

  local final_model, final_score, final_metrics = build_and_eval(best_params, { is_final = true })

  collectgarbage("collect")
  return final_model, best_params, final_metrics
end

return M

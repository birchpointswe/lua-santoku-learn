local tbl = require("santoku.table")
local arr = require("santoku.array")
local ivec = require("santoku.ivec")
local cluster = require("santoku.tsetlin.cluster")
local eval = require("santoku.tsetlin.evaluator.capi")

local M = {}

local function score_clustering (opts, margin)
  local ids, assignments, n_clusters = cluster.dbscan(opts.index, opts.min, margin)
  local score = eval.clustering_accuracy(assignments, ids, opts.pos, opts.neg, opts.threads)
  return score, ids, assignments, n_clusters
end

M.optimize_clustering = function (opts)
  local margins = ivec.create(opts.index:features() + 1)
  margins:fill_indices()
  local best_score = nil
  local best_ids = nil
  local best_assignments = nil
  local best_n_clusters
  local best_margin = nil
  local ncs = {}
  for m in margins:each() do
    local score, ids, assignments, n_clusters = score_clustering(opts, m)
    opts.each(score.f1, score.precision, score.recall, m, n_clusters)
    ncs[#ncs + 1] = n_clusters
    if not best_score or score.f1 > best_score.f1 then
      best_score = score
      best_ids = ids
      best_assignments = assignments
      best_n_clusters = n_clusters
      best_margin = m
    end
    collectgarbage("collect")
    if #ncs >= 4 and n_clusters == 1 and arr.mean(ncs, #ncs - 3, #ncs - 2) > arr.mean(ncs, #ncs - 1, #ncs) then
      break
    end
  end
  best_score.margin = best_margin
  return best_score, best_ids, best_assignments, best_n_clusters
end

return tbl.merge({}, M, eval)

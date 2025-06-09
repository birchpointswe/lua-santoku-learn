local tbl = require("santoku.table")
local arr = require("santoku.array")
local ivec = require("santoku.ivec")
local cluster = require("santoku.tsetlin.cluster")
local eval = require("santoku.tsetlin.evaluator.capi")

local M = {}

local function score_clustering (opts, margin, cache, fast)
  if fast then
    local c = cache[margin]
    if c then
      return c.score, c.n_clusters
    else
      local ids, clusters, assignments = cluster.dbscan(opts.index, opts.min, margin)
      local score = eval.clustering_accuracy(assignments, ids, opts.pos, opts.neg, opts.threads)
      cache[margin] = { score = score.f1, n_clusters = #clusters }
      if opts.each then
        opts.each(score.f1, score.precision, score.recall, margin, #clusters)
      end
      return score.f1, #clusters
    end
  else
    local ids, clusters, assignments = cluster.dbscan(opts.index, opts.min, margin)
    local score = eval.clustering_accuracy(assignments, ids, opts.pos, opts.neg, opts.threads)
    return score.f1, score, ids, clusters
  end
end

M.optimize_clustering = function (opts)

  local margins = ivec.create(opts.index:features())
  margins:fill_indices()

  local cache = {}
  local best_score = nil
  local best_margin = nil

  if opts.binary then

    local left = 0
    local right = margins:size() - 1
    while left <= right do
      local mid = math.floor((left + right) / 2)
      local score_mid =
        score_clustering(opts, margins:get(mid), cache, true)
      local score_left, score_right
      if mid > 0 then
        score_left = score_clustering(opts, margins:get(mid - 1), cache, true)
      end
      if mid < margins:size() - 1 then
        score_right = score_clustering(opts, margins:get(mid + 1), cache, true)
      end
      if not best_score or score_mid > best_score then
        best_score = score_mid
        best_margin = margins:get(mid)
      end
      if score_left and score_left > score_mid then
        right = mid - 1
      elseif score_right and score_right > score_mid then
        left = mid + 1
      else
        break
      end
    end

  else

    local ncs = {}
    for m in margins:each() do
      local score, nc = score_clustering(opts, m, cache, true)
      ncs[#ncs + 1] = nc
      if not best_score or score > best_score then
        best_score = score
        best_margin = m
      end
      if #ncs >= 4 and nc == 1 and arr.mean(ncs, #ncs - 3, #ncs - 2) > arr.mean(ncs, #ncs - 1, #ncs) then
        break
      end
    end

  end

  local _, score, ids, clusters = score_clustering(opts, best_margin, cache, false)
  score.margin = best_margin
  return score, ids, clusters

end

return tbl.merge({}, M, eval)

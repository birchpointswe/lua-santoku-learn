local ivec = require("santoku.ivec")
local tbl = require("santoku.table")

local ann = require("santoku.tsetlin.ann")

local function medoids (sample_index, k, max_iters)

  -- Initialize
  local n = sample_index:size()
  local medoid_index = ann.create({
    hash_bits = 0, -- Exhaustive medoid comparison
    features = sample_index:features(),
    threads = sample_index:threads()
  })
  local medoid_ids = ivec.create(n)
  local candidates = ivec.create()
  local assignment = ivec.create(n)
  local centroids = ivec.create()
  local seen = {}
  local clusters = {}
  local changed
  k = k > n and n or k
  max_iters = max_iters or math.huge

  -- Pick random k for initial medoids
  medoid_ids:fill_indices()
  medoid_ids:shuffle()
  medoid_ids:resize(k)
  medoid_index:import(sample_index, medoid_ids)

  for iter = 1, max_iters do

    -- Reset cluster assignments
    tbl.clear(clusters)
    changed = false

    -- Assign nodes to clusters
    for i = 1, n do
      candidates:clear()
      medoid_index:neighbors(sample_index:get(i), candidates)
      local cluster = candidates:get(0)
      if assignment:get(i) ~= cluster then
        assignment:set(i, cluster)
        changed = true
      end
      clusters[cluster] = clusters[cluster] or ivec.create()
      clusters[cluster]:push(i)
    end

    if not changed then
      print("Converged after", iter, "iterations")
      break
    end

    -- Re-assign medoids
    medoid_index:clear()
    medoid_ids:clear()
    tbl.clear(seen)
    for _, members in pairs(clusters) do
      centroids:clear()
      sample_index:centroids(members, centroids)
      for next_id in centroids:each() do
        if not seen[next_id] then
          seen[next_id] = true
          medoid_ids:push(next_id)
          medoid_index:add(next_id, sample_index:get(next_id))
          break
        end
      end
    end

  end

  -- Renumber
  local medoid_clusters = {}
  for i, a in medoid_ids:ieach() do
    medoid_clusters[a] = i
  end
  for i, a in assignment:ieach() do
    assignment:set(i, medoid_clusters[a])
  end

  return clusters, assignment
end

local function dbscan (index, min_pts, eps)

  local n = index:size()
  local neighborhoods = index:neighborhoods(eps)
  local assignment = ivec.create(n)
  assignment:fill(-2)
  local next_id = 0

  for i, ia in assignment:ieach() do
    if ia == -2 then
      local neighbors = neighborhoods[i]
      if neighbors:size() < min_pts then
        ia = -1
        assignment:set(i, ia)
      else
        ia = next_id
        assignment:set(i, ia)
        next_id = next_id + 1
        for neighbor in neighbors:each() do
          local na = assignment:get(neighbor)
          if na == -2 or na == -1 then
            assignment:set(neighbor, ia)
            local neighbor_neighbors = neighborhoods[neighbor]
            if neighbor_neighbors:size() >= min_pts then
              neighbors:copy(neighbor_neighbors)
            end
          end
        end
      end
    end
  end

  local clusters = {}

  for i, a in assignment:ieach() do
    clusters[a] = clusters[a] or ivec.create()
    clusters[a]:push(i)
  end

  return clusters, assignment
end

return {
  medoids = medoids,
  dbscan = dbscan,
}

-- TODO: required to populate pvec metatable. Can we avoid this? Perhaps load
-- from ivec?
require("santoku.pvec")

local ivec = require("santoku.ivec")

local function dbscan (index, min_pts, eps)

  local n = index:size()
  local ids, neighborhoods = index:neighborhoods(index:size(), eps)
  local assignments = ivec.create(n)
  local queue = ivec.create()

  assignments:fill(-2)
  local next_id = 0

  for i = 0, n - 1 do
    if assignments:get(i) ~= -2 then -- luacheck: ignore
      -- Already visitied
    else
      local neighbors = neighborhoods:get(i)
      if neighbors:size() < min_pts then
        assignments:set(i, -1) -- mark as noise
      else
        local cluster_id = next_id
        next_id = next_id + 1
        assignments:set(i, cluster_id)
        queue:clear()
        queue:copy(neighbors:keys())
        local qidx = 0
        while qidx < queue:size() do
          local neighbor = queue:get(qidx)
          qidx = qidx + 1
          local na = assignments:get(neighbor)
          if na == -1 then
            assignments:set(neighbor, cluster_id)
          end
          if na ~= -2 then -- luacheck: ignore
            -- Already processed
          else
            assignments:set(neighbor, cluster_id)
            local nn = neighborhoods:get(neighbor)
            if nn:size() >= min_pts then
              for k = 0, nn:size() - 1 do
                queue:push(nn:get(k))
              end
            end
          end
        end
      end
    end
  end

  -- Collect clusters
  local clusters = {}
  for i = 0, n - 1 do
    local a = assignments:get(i) + 1
    if a > 0 then
      if not clusters[a] then
        clusters[a] = ivec.create()
      end
      clusters[a]:push(ids:get(i))
    end
  end

  return ids, clusters, assignments

end

return {
  dbscan = dbscan,
}

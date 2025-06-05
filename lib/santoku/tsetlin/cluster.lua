local ivec = require("santoku.ivec")

local function dbscan (index, min_pts, eps)
  local n = index:size()
  local neighborhoods = index:neighborhoods(eps)
  local assignment = ivec.create(n)
  local queue = ivec.create()
  assignment:fill(-2)
  local next_id = 0
  for i = 0, n - 1 do
    if assignment:get(i) ~= -2 then -- luacheck: ignore
      -- Already visitied
    else
      local neighbors = neighborhoods:get(i)
      if neighbors:size() < min_pts then
        assignment:set(i, -1) -- mark as noise
      else
        local cluster_id = next_id
        next_id = next_id + 1
        assignment:set(i, cluster_id)
        queue:clear()
        queue:copy(neighbors)
        local qidx = 0
        while qidx < queue:size() do
          local neighbor = queue:get(qidx)
          qidx = qidx + 1
          local na = assignment:get(neighbor)
          if na == -1 then
            assignment:set(neighbor, cluster_id)
          end
          if na ~= -2 then -- luacheck: ignore
            -- Already processed
          else
            assignment:set(neighbor, cluster_id)
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
    local a = assignment:get(i) + 1
    if a > 0 then
      if not clusters[a] then
        clusters[a] = ivec.create()
      end
      clusters[a]:push(i)
    end
  end
  return clusters, assignment
end

return {
  dbscan = dbscan,
}

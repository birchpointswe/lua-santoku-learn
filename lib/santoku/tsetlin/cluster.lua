local pvec = require("santoku.pvec")
local ivec = require("santoku.ivec")

local function dbscan (index, min, eps)
  local ids = index:ids()
  local uididx = {}
  for i, uid in ids:ieach() do
    uididx[uid] = i
  end
  local assignments = ivec.create(ids:size())
  local queue = ivec.create()
  local nbrs = pvec.create()
  assignments:fill(-2)
  local next_id = 0
  for i, uid in ids:ieach() do
    if assignments:get(i) ~= -2 then -- luacheck: ignore
      -- Already visitied
    else
      nbrs:clear()
      index:neighbors(uid, eps, nbrs)
      if nbrs:size() < min then
        assignments:set(i, -1) -- mark as noise
      else
        local cluster_id = next_id
        next_id = next_id + 1
        assignments:set(i, cluster_id)
        queue:clear()
        queue:copy_pkeys(nbrs)
        local qidx = 0
        while qidx < queue:size() do
          local vid = queue:get(qidx)
          local neighbor = uididx[vid]
          qidx = qidx + 1
          local na = assignments:get(neighbor)
          if na == -1 then
            assignments:set(neighbor, cluster_id)
          end
          if na ~= -2 then -- luacheck: ignore
            -- Already processed
          else
            assignments:set(neighbor, cluster_id)
            nbrs:clear()
            index:neighbors(vid, eps, nbrs)
            if nbrs:size() >= min then
              for k = 0, nbrs:size() - 1 do
                queue:push((nbrs:get(k)))
              end
            end
          end
        end
      end
    end
  end
  return ids, assignments, next_id
end

return {
  dbscan = dbscan,
}

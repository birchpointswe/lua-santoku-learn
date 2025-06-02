local ds = require("santoku.tsetlin.dataset")
local booleanizer = require("santoku.tsetlin.booleanizer")
local cluster = require("santoku.tsetlin.cluster")
local test = require("santoku.test")

local MAX = nil
local N_THRESHOLDS = 4



local DBSCAN_MARGIN = 0.5
local DBSCAN_MIN = 2

test("clusters", function ()

  local dataset = ds.read_glove("test/res/glove.10k.txt", MAX)


  local booleanizer = booleanizer.create({
    n_thresholds = N_THRESHOLDS,
    categorical = nil,
    continuous = nil,
  })




















  local f_id_dim0, f_id_dimN = booleanizer:observe(dataset.embeddings, dataset.n_dims)
















  booleanizer:finalize()


  dataset.n_features = booleanizer:features()
















  dataset.bits = booleanizer:encode(dataset.embeddings, f_id_dim0)


  local medoid_clusters = cluster.medoids(dataset.bits, dataset.n_features, K_MEDOIDS)


  local dbscan_clusters = cluster.dbscan(dataset.bits, dataset.n_features, DBSCAN_MIN, DBSCAN_MARGIN)



end)

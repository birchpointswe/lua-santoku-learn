-- skip for now
os.exit(0)

local test = require("santoku.test")
local serialize = require("santoku.serialize")

local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local booleanizer = require("santoku.tsetlin.booleanizer")
local cluster = require("santoku.tsetlin.cluster")

local MAX = nil
local THREADS = nil
local N_THRESHOLDS = 4
local K_MEDOIDS = 128
local DBSCAN_MARGIN = 0.5
local DBSCAN_MIN = 2
local ANN_BUCKET_TARGET = 50

test("clusters", function ()

  local dataset = ds.read_glove("test/res/glove.10k.txt", MAX)

  local bzr = booleanizer.create({ n_thresholds = N_THRESHOLDS, threads = THREADS })
  bzr:observe(dataset.embeddings, dataset.n_dims)
  bzr:finalize()
  dataset.n_bits = bzr:bits()
  dataset.n_features = bzr:features()
  dataset.bits = bzr:encode(dataset.embeddings, dataset.n_dims)

  local index = ann.create({
    features = dataset.n_features,
    bucket_target = ANN_BUCKET_TARGET,
    guidance = dataset.bits,
    threads = THREADS,
  })
  index:add(dataset.bits)

  print("Input dimensions", dataset.n_dims)
  print("Input features", dataset.n_features)
  print("Booleanized features", dataset.n_bits)

  local medoid_clusters = cluster.medoids(index, K_MEDOIDS)
  local dbscan_clusters = cluster.dbscan(index, DBSCAN_MIN, DBSCAN_MARGIN)
  print(serialize(medoid_clusters))
  print(serialize(dbscan_clusters))

end)

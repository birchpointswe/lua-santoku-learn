local ds = require("santoku.tsetlin.dataset")
local booleanizer = require("santoku.tsetlin.booleanizer")
local cluster = require("santoku.tsetlin.cluster")
local test = require("santoku.test")

local MAX = nil
local N_THRESHOLDS = 4

-- Note: this would typically be decided via some best-margin search for the
-- given problem domain.
local DBSCAN_MARGIN = 0.5
local DBSCAN_MIN = 2

test("clusters", function ()

  local dataset = ds.read_glove("test/res/glove.10k.txt", MAX)

  -- Create a new booleanizer, waiting to receive observations of features
  local booleanizer = booleanizer.create({
    n_thresholds = N_THRESHOLDS,
    categorical = nil, -- ivec of feature ids to explicitly be considered categorical
    continuous = nil, -- ivec of feature ids to explicitly be considered continuous
  })

  -- Observe feature values. This API supports two modes:
  --
  --   local f_id_dim0, f_id_dimN = observe(<input-dvec>, <n-dims>, [ <dim0-feature-id> ])
  --
  --      Given a matrix with <n-dims> columns, add feature observations for
  --      each of the column values, treating column numbers as feature numbers
  --      starting at <dim0-feature-id> (default 0).
  --
  --      Returns the (first_id, last_id) of the assigned feature range
  --
  --   local f_id = observe(<feature-label>, <feature-value>)
  --
  --      Given a single feature label (integer or string) and feature value
  --      (boolean, number, or string), add a feature observation for the given
  --      feature/value pair.
  --
  --      Internally, a mapping from feature label to feature id is maintained.
  --
  --      Return the id of the feature observed.
  local f_id_dim0, f_id_dimN = booleanizer:observe(dataset.embeddings, dataset.n_dims)

  -- Finalize feature schema, mapping input features to
  -- booleanized output features using one/multi-hot encodings for those deemed
  -- categorical and thresholding for those deemed continuous.
  --
  -- The default type selection heuristic is as follows:
  --
  --   If all observations of a feature are numeric, consider it continuous.
  --
  --   If observations of a feature are a mixture of types, consider it
  --   categorical.
  --
  --   If a categorical feature has only two observations, consider it binary.
  --
  -- Optionally alow specific overrides for categorical, continuous, or binary
  -- features.
  booleanizer:finalize()

  -- After finalization, we know the fixed number of booleanized features
  dataset.n_features = booleanizer:features()

  -- Booleanize the embeddings. This API supports two modes:
  --
  --    local bits = encode(<input-dvec>, <dim0-feature-id>, [ <output-ivec> ])
  --
  --      Given a matrix, a feature id for the first dimension, and optionally
  --      an output ivec (defaulted to a new ivec), populate the output ivec
  --      with the corresponding booleanized per-sample bits. Returns the output
  --      ivec.
  --
  --    local bits = encode(<sample-id>, <feature-id>, <feature-value>, [ <output-ivec> ])
  --
  --      Given a sample id, a feature id, a value, and optionally an output
  --      ivec (defaulted to a new ivec), populate the corresponding booleanized
  --      bits for that sample/feature/value combo to the output ivec. Returns
  --      the output ivec.
  dataset.bits = booleanizer:encode(dataset.embeddings, f_id_dim0)

  -- Cluster via k-medoids
  local medoid_clusters = cluster.medoids(dataset.bits, dataset.n_features, K_MEDOIDS)

  -- Cluster via dbscan
  local dbscan_clusters = cluster.dbscan(dataset.bits, dataset.n_features, DBSCAN_MIN, DBSCAN_MARGIN)

  -- code continues...

end)

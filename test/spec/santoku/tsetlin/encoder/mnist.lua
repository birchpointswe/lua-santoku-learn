local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local pvec = require("santoku.pvec")
local ivec = require("santoku.ivec")
local cvec = require("santoku.cvec")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil --4000
local MAX_CLASS = nil
local THREADS = nil

local VISIBLE = 784
local HIDDEN = 9
local LANDMARKS = 4
local FEATURE_MODE = "landmarks" -- "landmarks", "landmarks+raw", or "raw"

local BINARIZE = "itq" -- "itq", "median", or "sign"
local TCH = true
local EPS_SPECTRAL = 1e-5
local NORMALIZED = true
local KNN = 256
local KNN_EPS = nil
local KNN_MIN = nil
local KNN_MUTUAL = true
local SEED_ANCHORS = 0
local SEED_PAIRS = 0
local SAMPLED_ANCHORS = 4
local TEST_ANCHORS = 4
local RANK_WINDOW = 0
local RANK_FLOOR = 0

local CLAUSES = { def = 8, min = 8, max = 32, int = true, log = true, pow2 = true }
local CLAUSE_TOLERANCE = { def = 8, min = 8, max = 64, int = true, log = true, pow2 = true }
local CLAUSE_MAXIMUM = { def = 8, min = 8, max = 64, int = true, log = true, pow2 = true }
local TARGET = { def = 4, min = 2, max = 64, int = true, log = true, pow2 = true }
local SPECIFICITY = { def = 1000, min = 2, max = 4000, int = true, log = true }

local SEARCH_PATIENCE = 10
local SEARCH_ROUNDS = 10
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 20
local ITERATIONS = 100

test("tsetlin", function ()

  print("Reading data")
  -- TODO: this should read the data directly into a packed cvec instead of ivec
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", VISIBLE, MAX, MAX_CLASS)
  dataset.n_visible = VISIBLE
  dataset.n_hidden = HIDDEN
  dataset.n_landmarks = LANDMARKS
  dataset.n_latent = HIDDEN * LANDMARKS

  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)
  if KNN then
    local idx = ann.create({ features = dataset.n_visible, expected_size = train.n })
    idx:add(train.problems:bits_to_cvec(train.n, dataset.n_visible), train.ids)
    local ids, hoods = idx:neighborhoods(KNN, KNN_EPS, KNN_MIN, KNN_MUTUAL)
    train.seed = ds.star_hoods(ids, hoods)
  else
    train.seed = pvec.create()
  end
  if SEED_PAIRS then
    ds.random_pairs(train.ids, SEED_PAIRS, train.seed)
  end
  if SEED_ANCHORS then
    ds.anchor_pairs(train.ids, SEED_ANCHORS, train.seed)
  end

  print("Building the graph index")
  local graph_ranks = ivec.create(dataset.n_visible)
  graph_ranks:fill(0)
  for _ = 1, 10 do
    graph_ranks:push(1) -- add ranks for class features
  end
  local graph_supervision = ivec.create(train.n)
  for i = 0, train.n - 1 do
    graph_supervision:set(i, i * 10 + train.solutions:get(i))
  end
  local graph_problems = ivec.create()
  graph_problems:copy(train.problems)
  graph_problems:bits_extend(graph_supervision, dataset.n_visible, 10)
  local graph_features = dataset.n_visible + 10
  local graph_index = inv.create({
    features = graph_features,
    ranks = graph_ranks,
    n_ranks = 2,
    rank_decay_window = RANK_WINDOW,
    rank_decay_floor = RANK_FLOOR,
  })
  graph_index:add(graph_problems, train.ids)

  print("Creating graph")
  local stopwatch = utc.stopwatch()
  train.graph = graph.create({
    edges = train.seed,
    index = graph_index,
    threads = THREADS,
    each = function (s, b, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Components: %-6d  Edges: %-6d\n", d, dd, dt, s, b) -- luacheck: ignore
    end
  })
  train.adj_ids,
  train.adj_offsets,
  train.adj_neighbors,
  train.adj_weights =
    train.graph:adjacency()

  print("Spectral eigendecomposition")
  train.ids_spectral, train.codes_spectral = spectral.encode({
    ids = train.adj_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    n_hidden = dataset.n_hidden,
    normalized = NORMALIZED,
    eps_primme = EPS_SPECTRAL,
    threads = THREADS,
    each = function (t, s, v, k)
      local d, dd = stopwatch()
      if t == "done" then
        str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
      elseif t == "eig" then
        local gap = train.eig_last and v - train.eig_last or 0
        train.eig_last = v
        str.printf("  Time: %6.2f %6.2f  Eig: %3d = %.8f   Gap = %.8f   %s\n", d, dd, s, v, gap, k and "" or "drop")
      end
    end
  })
  train.solutions_spectral = ivec.create()
  train.solutions_spectral:copy(train.solutions, train.ids_spectral)

  train.codes_spectral_cont = train.codes_spectral
  if BINARIZE == "itq" then
    print("Iterative Quantization")
    train.codes_spectral = itq.encode({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      threads = THREADS,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("Median thresholding")
    train.codes_spectral = itq.median({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  elseif BINARIZE == "sign" then
    print("Sign thresholding")
    train.codes_spectral = itq.sign({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  end

  if TCH then
    print("Flipping bits")
    tch.refine({
      ids = train.adj_ids,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  print("Creating spectral codes index for landmarks")
  local spectral_index = ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
  spectral_index:add(train.codes_spectral, train.ids_spectral)

  print("Codebook stats")
  train.pos_sampled, train.neg_sampled = ds.multiclass_pairs(train.ids_spectral, train.solutions_spectral, SAMPLED_ANCHORS, SAMPLED_ANCHORS) -- luacheck: ignore
  train.entropy = eval.entropy_stats(train.codes_spectral, train.n, dataset.n_hidden, THREADS)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  train.auc_binary = eval.auc(train.ids_spectral, train.codes_spectral, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS) -- luacheck: ignore
  train.auc_continuous = eval.auc(train.ids_spectral, train.codes_spectral_cont, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS) -- luacheck: ignore
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", train)

  print("Retrieval stats")
  train.similarity = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = dataset.n_hidden,
    ids = train.ids_spectral,
    pos = train.pos_sampled,
    neg = train.neg_sampled,
    threads = THREADS,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("Best\n  BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    train.similarity)

  if true then
    -- DEBUG: early exit
    return
  end

  print("Creating landmark index")
  -- TODO: Instead of using all spectral points as landmarks, implement coreset
  -- sampling to maximally cover the spectral space with the fewest number of
  -- landmarks (using ANN for better performance on dense features)
  train.ids_landmark = train.ids_spectral
  train.problems_landmark_index = ivec.create()
  train.problems_landmark_index:bits_copy(train.problems, nil, train.ids_landmark, dataset.n_visible) -- luacheck: ignore
  train.problems_landmark_index = train.problems_landmark_index:bits_to_cvec(train.ids_landmark:size(), dataset.n_visible) -- luacheck: ignore
  -- train.problems_landmark_index = train.problems_landmark_index:bits_to_cvec(train.ids_landmark:size(), dataset.n_visible) -- luacheck: ignore
  local landmark_index = ann.create({ features = dataset.n_visible, expected_size = train.ids_landmark:size() }) -- luacheck: ignore
  landmark_index:add(train.problems_landmark_index, train.ids_landmark)
  train.ids_landmark, train.hoods_landmark = landmark_index:neighborhoods_by_ids(train.ids_landmark, dataset.n_landmarks) -- luacheck: ignore

  print("Prepping for encoder (train) - Mode: " .. FEATURE_MODE)
  -- Set up in-sample training points
  local bytes_per_code = math.ceil(dataset.n_hidden / 8)
  local bits_per_code_padded = bytes_per_code * 8
  train.solutions_landmark = spectral_index:get(train.ids_landmark)
  train.n = train.ids_landmark:size()
  local tmp_hood_ids = ivec.create()

  if FEATURE_MODE == "landmarks" then
    train.problems_landmark = cvec.create(train.ids_landmark:size() * dataset.n_landmarks * bytes_per_code)
    train.problems_landmark:zero()
    train.problems_landmark:setn(0)
    for hood in train.hoods_landmark:each() do
      hood:keys(tmp_hood_ids)
      tmp_hood_ids:lookup(train.ids_landmark)
      spectral_index:get(tmp_hood_ids, train.problems_landmark, true)
      local pad = dataset.n_landmarks - tmp_hood_ids:size()
      train.problems_landmark:setn(train.problems_landmark:size() + pad * bytes_per_code)
    end
    train.problems_landmark:bits_flip_interleave(train.n, dataset.n_landmarks * bits_per_code_padded)
    dataset.encoder_input_dims = dataset.n_landmarks * bits_per_code_padded
  elseif FEATURE_MODE == "landmarks+raw" then
    train.problems_landmark = cvec.create(train.ids_landmark:size() * dataset.n_landmarks * bytes_per_code)
    train.problems_landmark:zero()
    train.problems_landmark:setn(0)
    for hood in train.hoods_landmark:each() do
      hood:keys(tmp_hood_ids)
      tmp_hood_ids:lookup(train.ids_landmark)
      spectral_index:get(tmp_hood_ids, train.problems_landmark, true)
      local pad = dataset.n_landmarks - tmp_hood_ids:size()
      train.problems_landmark:setn(train.problems_landmark:size() + pad * bytes_per_code)
    end
    local train_raw = ivec.create()
    train_raw:bits_copy(train.problems, nil, train.ids_landmark, dataset.n_visible)
    train.problems_landmark:bits_extend(train.problems, train.n, dataset.n_landmarks * bits_per_code_padded, dataset.n_visible)
    train.problems_landmark:bits_flip_interleave(train.n, dataset.n_landmarks * bits_per_code_padded + dataset.n_visible)
    dataset.encoder_input_dims = dataset.n_landmarks * bits_per_code_padded + dataset.n_visible
  elseif FEATURE_MODE == "raw" then
    train.problems_landmark = ivec.create()
    train.problems_landmark:bits_copy(train.problems, nil, train.ids_landmark, dataset.n_visible)
    train.problems_landmark = train.problems_landmark:bits_to_cvec(train.n, dataset.n_visible)
    train.problems_landmark:bits_flip_interleave(train.n, dataset.n_visible)
    dataset.encoder_input_dims = dataset.n_visible
  else
    error("Invalid FEATURE_MODE: " .. FEATURE_MODE)
  end

  print("Prepping for encoder (test)")
  test.pos_sampled, test.neg_sampled = ds.multiclass_pairs(test.ids, test.solutions, TEST_ANCHORS, TEST_ANCHORS)
  -- Set up test problems (find in-sample landmarks for out-of-sample vectors)
  if FEATURE_MODE == "landmarks" then
    local test_vecs = test.problems:bits_to_cvec(test.n, dataset.n_visible)
    test.ids_landmark, test.hoods_landmark = landmark_index:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks)
    test.problems_landmark = cvec.create(test.n * dataset.n_landmarks * bytes_per_code)
    test.problems_landmark:zero()
    test.problems_landmark:setn(0)
    for hood in test.hoods_landmark:each() do
      hood:keys(tmp_hood_ids)
      tmp_hood_ids:lookup(test.ids_landmark)
      spectral_index:get(tmp_hood_ids, test.problems_landmark, true)
      local pad = dataset.n_landmarks - tmp_hood_ids:size()
      test.problems_landmark:setn(test.problems_landmark:size() + pad * bytes_per_code)
    end
    test.problems_landmark:bits_flip_interleave(test.n, dataset.n_landmarks * bits_per_code_padded)
  elseif FEATURE_MODE == "landmarks+raw" then
    local test_vecs = test.problems:bits_to_cvec(test.n, dataset.n_visible)
    test.ids_landmark, test.hoods_landmark = landmark_index:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks)
    test.problems_landmark = cvec.create(test.n * dataset.n_landmarks * bytes_per_code)
    test.problems_landmark:zero()
    test.problems_landmark:setn(0)
    for hood in test.hoods_landmark:each() do
      hood:keys(tmp_hood_ids)
      tmp_hood_ids:lookup(test.ids_landmark)
      spectral_index:get(tmp_hood_ids, test.problems_landmark, true)
      local pad = dataset.n_landmarks - tmp_hood_ids:size()
      test.problems_landmark:setn(test.problems_landmark:size() + pad * bytes_per_code)
    end
    test.problems_landmark:bits_extend(test.problems, test.n, dataset.n_landmarks * bits_per_code_padded, dataset.n_visible)
    test.problems_landmark:bits_flip_interleave(test.n, dataset.n_landmarks * bits_per_code_padded + dataset.n_visible)
  elseif FEATURE_MODE == "raw" then
    test.problems_landmark = test.problems:bits_to_cvec(test.n, dataset.n_visible, true)
  else
    error("Invalid FEATURE_MODE: " .. FEATURE_MODE)
  end

  print()
  str.printf("Feature mode      %s\n", FEATURE_MODE)
  str.printf("Input Features    %d\n", dataset.encoder_input_dims)
  str.printf("Output Features   %d (padded: %d)\n", dataset.n_hidden, bits_per_code_padded)
  str.printf("Train problems    %d\n", train.n)
  str.printf("Test problems     %d\n", test.n)

  print("Creating encoder")
  stopwatch = utc.stopwatch()
  --[[local t = ]] tm.optimize_encoder({

    visible = dataset.encoder_input_dims,
    hidden = dataset.n_hidden,
    sentences = train.problems_landmark,
    codes = train.solutions_landmark,
    samples = train.n,

    clauses = CLAUSES,
    clause_tolerance = CLAUSE_TOLERANCE,
    clause_maximum = CLAUSE_MAXIMUM,
    target = TARGET,
    specificity = SPECIFICITY,

    search_patience = SEARCH_PATIENCE,
    search_rounds = SEARCH_ROUNDS,
    search_trials = SEARCH_TRIALS,
    search_iterations = SEARCH_ITERATIONS,
    final_iterations = ITERATIONS,
    threads = THREADS,

    search_metric = function (t)
      local predicted = t.predict(train.problems_landmark, train.n)
      local accuracy = eval.encoding_accuracy(predicted, train.solutions_landmark, train.n, dataset.n_hidden)
      return accuracy.mean_hamming, accuracy
    end,

    each = function (t, is_final, train_accuracy, params, epoch, round, trial)
      -- luacheck: push ignore
      local d, dd = stopwatch()
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%d S=%.2f  Epoch  %d\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, epoch)
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%d S=%.2f  R=%d T=%d  Epoch  %d\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, epoch)
      end
      train.accuracy_predicted = train_accuracy
      str.printi("    Train (acc) | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.accuracy_predicted)
      if is_final then
        train.codes_predicted = t.predict(train.problems_landmark, train.n)
        train.auc_predicted = eval.auc(train.ids_spectral, train.codes_predicted, train.pos_sampled, train.neg_sampled, dataset.n_hidden)
        train.similarity_predicted = eval.optimize_retrieval({
          codes = train.codes_predicted,
          n_dims = dataset.n_hidden,
          ids = train.ids_spectral,
          pos = train.pos_sampled,
          neg = train.neg_sampled,
        })
        test.codes_predicted = t.predict(test.problems_landmark, test.n)
        test.auc_predicted = eval.auc(test.ids, test.codes_predicted, test.pos_sampled, test.neg_sampled, dataset.n_hidden)
        test.similarity_predicted = eval.optimize_retrieval({
          codes = test.codes_predicted,
          n_dims = dataset.n_hidden,
          ids = test.ids,
          pos = test.pos_sampled,
          neg = test.neg_sampled,
        })
        train.similarity.auc = train.auc_binary
        train.similarity_predicted.auc = train.auc_predicted
        test.similarity_predicted.auc = test.auc_predicted
        str.printi("    Codes (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity)
        str.printi("    Train (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity_predicted)
        str.printi("    Test (sim)  | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", test.similarity_predicted)
        print()
      end
      -- luacheck: pop
    end,
  })

end)

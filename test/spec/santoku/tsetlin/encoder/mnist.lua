local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
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
local HIDDEN = 8
local LANDMARKS = 4

local BINARIZE = "itq"
local TCH = true
local EPS_SPECTRAL = 1e-5
local NORMALIZED = true
local KNN = 4
local KNN_CACHE = 8
local SEED_ANCHORS = 0
local SAMPLED_ANCHORS = 4
local TEST_ANCHORS = 4

local CLAUSES = 16 --{ def = 8, min = 8, max = 32, log = true, int = true }
local CLAUSE_TOLERANCE = 16 --{ def = 8, min = 8, max = 64, int = true }
local CLAUSE_MAXIMUM = 16 --{ def = 8, min = 8, max = 64, int = true }
local TARGET = 8 --{ def = 4, min = 8, max = 64 }
local SPECIFICITY = 1000 --{ def = 1000, min = 100, max = 2000 }

local SEARCH_PATIENCE = 100
local SEARCH_ROUNDS = 10
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 100
local ITERATIONS = 100

test("tsetlin", function ()

  print("Reading data")
  -- TODO: this should read the data directly into a packed cvec instead of ivec
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", VISIBLE, MAX, MAX_CLASS)
  dataset.n_visible = VISIBLE
  dataset.n_hidden = HIDDEN
  dataset.n_landmarks = LANDMARKS
  dataset.n_latent = HIDDEN * LANDMARKS
  dataset.latent_chunks = num.floor((dataset.n_latent + 7) / 8)

  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  train.pos_seed, train.neg_seed = ds.multiclass_pairs(train.ids, train.solutions, SEED_ANCHORS, SEED_ANCHORS)
  train.pos_sampled, train.neg_sampled = ds.multiclass_pairs(train.ids, train.solutions, SAMPLED_ANCHORS, SAMPLED_ANCHORS) -- luacheck: ignore
  train.seed = pvec.create()
  train.seed:copy(train.pos_seed)
  train.seed:copy(train.neg_seed)
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.pos_sampled, test.neg_sampled = ds.multiclass_pairs(test.ids, test.solutions, TEST_ANCHORS, TEST_ANCHORS)

  -- print("\nInjecting Supervision")
  print("\nIndexing visible features")
  local graph_index = ann.create({ features = dataset.n_visible, expected_size = train.n })
  graph_index:add(train.problems:bits_to_cvec(train.n, dataset.n_visible), train.ids)
  -- local graph_ranks = ivec.create(dataset.n_visible)
  -- graph_ranks:fill(0)
  -- for _ = 1, 10 do
  --   graph_ranks:push(1) -- add ranks for class features
  -- end
  -- local graph_supervision = ivec.create(train.n)
  -- for i = 0, train.n - 1 do
  --   graph_supervision:set(i, i * 10 + train.solutions:get(i))
  -- end
  -- local graph_problems = ivec.create()
  -- graph_problems:copy(train.problems)
  -- graph_problems:bits_extend(graph_supervision, dataset.n_visible, 10)
  -- local graph_features = dataset.n_visible + 10
  -- local graph_index = ann.create({
  --   features = graph_features,
  --   expected_size = train.n,
  --   ranks = graph_ranks,
  --   n_ranks = 2,
  --   rank_decay_window = 2,
  --   rank_decay_sigma = 100,
  --   rank_decay_floor = 0.5
  -- })
  -- graph_index:add(graph_problems, train.ids)

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  train.graph = graph.create({
    edges = train.seed,
    index = graph_index,
    knn = KNN,
    knn_cache = KNN_CACHE,
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

  print("\nSpectral eigendecomposition")
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
        str.printf("  Time: %6.2f %6.2f  Eig: %3d = %.5f  %s\n", d, dd, s, v, k and "" or "drop")
      end
    end
  })

  train.codes_spectral_cont = train.codes_spectral
  if BINARIZE == "itq" then
    print("\nIterative Quantization")
    train.codes_spectral = itq.encode({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      threads = THREADS,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("\nMedian thresholding")
    train.codes_spectral = itq.median({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  elseif BINARIZE == "sign" then
    print("\nSign thresholding")
    train.codes_spectral = itq.sign({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  end

  if TCH then
    print("\nFlipping bits")
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

  train.codes_spectral = train.codes_spectral:bits_to_cvec(train.ids_spectral:size(), dataset.n_hidden)

  print("\nCreating spectral codes index for landmarks")
  local spectral_index = ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
  spectral_index:add(train.codes_spectral, train.ids_spectral)

  print("\nCodebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.n, dataset.n_hidden, THREADS)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  train.auc_binary = eval.auc(train.ids_spectral, train.codes_spectral, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS) -- luacheck: ignore
  train.auc_continuous = eval.auc(train.ids_spectral, train.codes_spectral_cont, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS) -- luacheck: ignore
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", train)

  print("\nRetrieval stats")
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
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    train.similarity)

  print("\nPrepping for encoder (train)")
  -- Set up in-sample training points
  --
  -- TODO: Instead of using all spectral points as landmarks, implement coreset
  -- sampling to maximally cover the spectral space with the fewest number of
  -- landmarks
  train.ids_landmark = train.ids_spectral
  train.ids_landmark, train.hoods_landmark = spectral_index:neighborhoods(dataset.n_landmarks)
  train.solutions_landmark = spectral_index:get(train.ids_landmark)
  train.problems_landmark = cvec.create(train.ids_landmark:size() * dataset.n_landmarks * dataset.n_hidden / 8)
  train.problems_landmark:zero()
  train.problems_landmark:setn(0)
  train.n = train.ids_landmark:size()
  local tmp_hood_ids = ivec.create()
  for hood in train.hoods_landmark:each() do
    hood:keys(tmp_hood_ids)
    tmp_hood_ids:lookup(train.ids_landmark)
    spectral_index:get(tmp_hood_ids, train.problems_landmark, true)
    local pad = dataset.n_landmarks - tmp_hood_ids:size()
    train.problems_landmark:setn(train.problems_landmark:size() + pad * dataset.n_hidden / 8)
  end
  train.problems_landmark:bits_flip_interleave(train.n, dataset.n_latent)

  print("Prepping for encoder (test)")
  -- Set up test problems (find in-sample landmarks for out-of-sample vectors)
  local test_vecs = test.problems:bits_to_cvec(test.n, dataset.n_visible)
  test.ids_landmark, test.hoods_landmark = graph_index:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks)
  test.problems_landmark = cvec.create(test.n * dataset.n_landmarks * dataset.n_hidden / 8)
  test.problems_landmark:zero()
  test.problems_landmark:setn(0)
  for hood in test.hoods_landmark:each() do
    hood:keys(tmp_hood_ids)
    tmp_hood_ids:lookup(test.ids_landmark)
    spectral_index:get(tmp_hood_ids, test.problems_landmark, true)
    local pad = dataset.n_landmarks - tmp_hood_ids:size()
    test.problems_landmark:setn(test.problems_landmark:size() + pad * dataset.n_hidden / 8)
  end
  test.problems_landmark:bits_flip_interleave(test.n, dataset.n_latent)

  print()
  str.printf("Input Features    %d\n", dataset.n_latent)
  str.printf("Output Features   %d\n", dataset.n_hidden)
  str.printf("Train problems    %d\n", train.n)
  str.printf("Test problems     %d\n", test.n)

  print("\nCreating encoder\n")
  stopwatch = utc.stopwatch()
  --[[local t = ]] tm.optimize_encoder({

    visible = dataset.n_latent,
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
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%.2f S=%.2f  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, epoch)
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%.2f S=%.2f  R=%d T=%d  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, epoch)
      end
      train.accuracy_predicted = train_accuracy
      str.printi("    Train (acc) | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)\n", train.accuracy_predicted)
      if is_final then
        train.codes_predicted = t.predict(train.problems_landmark, train.n)
        train.auc_predicted = eval.auc(train.ids, train.codes_predicted, train.pos_sampled, train.neg_sampled, dataset.n_hidden)
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

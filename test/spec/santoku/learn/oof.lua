local optimize = require("santoku.learn.optimize")
local ivec = require("santoku.ivec")
local fvec = require("santoku.fvec")
local test = require("santoku.test")

test("oof", function ()

  test("coverage, no leakage, scatter alignment", function ()
    local n, k = 7, 3
    local fold = ivec.create({ 0, 1, 2, 0, 1, 2, 0 })
    local fit_calls = 0
    local out = optimize.oof({
      n = n, k = k, fold = fold,
      fit = function (train_idx)
        fit_calls = fit_calls + 1
        local seen = {}
        for j = 0, train_idx:size() - 1 do seen[train_idx:get(j)] = true end
        return seen
      end,
      predict = function (seen, eval_idx)
        local m = eval_idx:size()
        local scores = fvec.create(m)
        for j = 0, m - 1 do
          local i = eval_idx:get(j)
          scores:set(j, seen[i] and 0 or (i + 1))
        end
        return scores
      end,
    })
    assert(fit_calls == k)
    assert(out:size() == n)
    for i = 0, n - 1 do
      assert(out:get(i) == i + 1,
        "oof[" .. i .. "] = " .. out:get(i) .. " (expected " .. (i + 1) .. ")")
    end
  end)

  test("train is exact complement of eval per fold", function ()
    local n, k = 6, 2
    local fold = ivec.create({ 0, 0, 0, 1, 1, 1 })
    optimize.oof({
      n = n, k = k, fold = fold,
      fit = function (train_idx) return train_idx end,
      predict = function (train_idx, eval_idx)
        assert(train_idx:size() + eval_idx:size() == n)
        local seen = {}
        for j = 0, train_idx:size() - 1 do seen[train_idx:get(j)] = true end
        local m = eval_idx:size()
        local scores = fvec.create(m)
        for j = 0, m - 1 do
          assert(not seen[eval_idx:get(j)], "leak: eval row present in train")
          scores:set(j, 0)
        end
        return scores
      end,
    })
  end)

end)

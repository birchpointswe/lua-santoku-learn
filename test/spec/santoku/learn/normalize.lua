require("santoku.error")
local csr = require("santoku.learn.csr")
local ivec = require("santoku.ivec")
local test = require("santoku.test")

-- The streaming normalizer must (a) produce byte-identical output whether a doc is
-- normalized whole or in arbitrary literal runs (prev_space carries across runs;
-- trim is row-level), and (b) fold C0+0x7F to space.
test("normalize stream", function ()

  local function bounds (...)
    local b = ivec.create()
    for _, x in ipairs({ ... }) do b:push(x) end
    return b
  end

  test("run-splitting invariance", function ()
    local s = "The Quick Brown Fox jumps"
    local whole = csr.normalize_runs(s)
    local split = csr.normalize_runs(s, bounds(0, 4, 9, 15, #s))
    assert(whole == split)
    assert(whole == "the quick brown fox jumps")
  end)

  test("boundary space carries across runs", function ()
    -- a double space split mid-gap must still collapse to one space, like the whole doc
    local s = "a  b"
    assert(csr.normalize_runs(s) == "a b")
    assert(csr.normalize_runs(s, bounds(0, 2, #s)) == "a b")   -- "a " | " b"
  end)

  test("fold + collapse on accented/clean text", function ()
    assert(csr.normalize_runs("Caf\195\169 du Monde -- 2024  ") == "cafe du monde -- 2024")
  end)

  test("control-fold scrub (C0 + DEL -> space)", function ()
    assert(csr.normalize_runs("a\008b") == "a b")
    assert(csr.normalize_runs("x\127y") == "x y")
  end)

  test("lowercase, ws-collapse, trim", function ()
    assert(csr.normalize_runs("  Hello   World  ") == "hello world")
  end)

end)

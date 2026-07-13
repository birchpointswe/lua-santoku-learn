local spectral = require("santoku.learn.spectral")
local util = require("santoku.learn.util")
local csr = require("santoku.csr")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fvec = require("santoku.fvec")
local str = require("santoku.string")
local test = require("santoku.test")

io.stdout:setvbuf("line")

local N, C = 256, 8

local function make_vals ()
  math.randomseed(42)
  local vals = {}
  for i = 1, N * C do vals[i] = math.random() * 2 - 1 end
  return vals
end

local function col_csr (vals, c0)
  local off, nbr, v = ivec.create(), ivec.create(), fvec.create()
  off:push(0)
  for r = 0, N - 1 do
    nbr:push(0); v:push(vals[r * C + c0 + 1])
    off:push(nbr:size())
  end
  return csr.create({ offsets = off, neighbors = nbr, values = v, n_cols = 1 }):i32()
end

local function range_csr (vals, c0, c1)
  local nc = c1 - c0
  local off, nbr, v = ivec.create(), ivec.create(), fvec.create()
  off:push(0)
  for r = 0, N - 1 do
    for c = c0, c1 - 1 do nbr:push(c - c0); v:push(vals[r * C + c + 1]) end
    off:push(nbr:size())
  end
  return csr.create({ offsets = off, neighbors = nbr, values = v, n_cols = nc }):i32()
end

local function offsets (...)
  local iv = ivec.create()
  for _, x in ipairs({ ... }) do iv:push(x) end
  return iv
end

local function codes_for (bl, lms)
  local _, enc = spectral.encode({ blocks = bl, landmarks = lms, n_landmarks = lms:size(), kernel = "cosine" })
  return enc:encode({ blocks = bl })
end

local function landmarks (m)
  local lms = ivec.create()
  for i = 0, m - 1 do lms:push(i * 7) end
  return lms
end

local function assert_codes_close (cA, cB, tag)
  local level = "none"
  for _, tol in ipairs({ 1e-4, 1e-5, 1e-6 }) do
    if cA:eq(cB, tol) then level = tostring(tol) end
  end
  if cA:eq(cB) then level = "exact" end
  str.printf("[ard] %s codes agree to: %s\n", tag, level)
  assert(cA:eq(cB, 1e-3), tag .. ": grouped codes deviate beyond fp-noise tolerance")
end

test("group_gauge matches colscale on a single group", function ()
  math.randomseed(7)
  local pcs, w = dvec.create(C), fvec.create(C)
  for c = 0, C - 1 do pcs:set(c, math.random() * 5 + 0.1); w:set(c, math.random()) end
  local e, s = 2.5, 1.3
  local cs, wssq = w:colscale(pcs, e, 1e-6)
  local baked = pcs:group_gauge(offsets(0, C), { s }, N, { w = w, exps = { e } })
  local mult = s * math.sqrt(N / wssq)
  for c = 0, C - 1 do
    local a, b = baked:get(c), cs:get(c) * mult
    assert(math.abs(a - b) <= 1e-5 * math.max(1, math.abs(b)), "group_gauge deviates at col " .. c)
  end
end)

test("identity groups == columns-as-blocks (no relevance)", function ()
  local vals = make_vals()
  local scales = { 1.7, 0.3, 2.5, 0.9, 4.0, 0.1, 1.0, 0.55 }
  local blocksA, pcsA = {}, {}
  for c = 0, C - 1 do blocksA[c + 1] = col_csr(vals, c) end
  for i = 1, C do pcsA[i] = blocksA[i]:sumsq_cols() end
  local Xall = range_csr(vals, 0, C)
  local blA = util.build_blocks(blocksA, scales, nil, N, nil, pcsA)
  local blB = util.build_blocks({ Xall }, scales, nil, N, nil, { Xall:sumsq_cols() },
    { offsets(0, 1, 2, 3, 4, 5, 6, 7, 8) })
  local baked = blB[1].colscale
  for j = 1, C do
    local a, b = baked:get(j - 1), blA[j].scale
    assert(math.abs(a - b) <= 1e-6 * math.max(1, math.abs(b)),
      "baked multiplier deviates at col " .. j)
  end
  local lms = landmarks(6)
  local cA, cB = codes_for(blA, lms), codes_for(blB, lms)
  assert_codes_close(cA, cB, "identity")
end)

test("two 4-col groups == two blocks (relevance + exponent)", function ()
  local vals = make_vals()
  math.randomseed(9)
  local wall = fvec.create(C)
  for c = 0, C - 1 do wall:set(c, math.random()) end
  local w1, w2 = fvec.create(4), fvec.create(4)
  for c = 0, 3 do w1:set(c, wall:get(c)); w2:set(c, wall:get(4 + c)) end
  local scales, exps = { 1.9, 0.4 }, { 2.1, 0.7 }
  local B1, B2 = range_csr(vals, 0, 4), range_csr(vals, 4, 8)
  local Xall = range_csr(vals, 0, C)
  local blA = util.build_blocks({ B1, B2 }, scales, exps, N,
    { w1, w2 }, { B1:sumsq_cols(), B2:sumsq_cols() })
  local blB = util.build_blocks({ Xall }, scales, exps, N,
    { wall }, { Xall:sumsq_cols() }, { offsets(0, 4, 8) })
  local baked = blB[1].colscale
  for g = 1, 2 do
    for c = 0, 3 do
      local a = baked:get((g - 1) * 4 + c)
      local b = blA[g].scale * blA[g].colscale:get(c)
      assert(math.abs(a - b) <= 1e-5 * math.max(1, math.abs(b)),
        "baked multiplier deviates at group " .. g .. " col " .. c)
    end
  end
  local lms = landmarks(6)
  local cA, cB = codes_for(blA, lms), codes_for(blB, lms)
  assert_codes_close(cA, cB, "grouped-relevance")
end)

test("colscale fold: prescaled block + c == raw block + (c .* w)", function ()
  local vals = make_vals()
  math.randomseed(11)
  local w, c = fvec.create(C), fvec.create(C)
  for i = 0, C - 1 do w:set(i, math.random() * 2 + 0.1); c:set(i, math.random() * 2 + 0.1) end
  local Xa = range_csr(vals, 0, C); Xa:bns(w)
  local blA = { { x = Xa, n_tokens = C, scale = 1.0, colscale = c } }
  local cw = fvec.create(C)
  for i = 0, C - 1 do cw:set(i, c:get(i)) end
  cw:scalev(w)
  local blB = { { x = range_csr(vals, 0, C), n_tokens = C, scale = 1.0, colscale = cw } }
  local lms = landmarks(6)
  local cA, cB = codes_for(blA, lms), codes_for(blB, lms)
  assert_codes_close(cA, cB, "colscale-fold")
end)

local function ignores_colscale (kernel_args, scale2, tag)
  local vals = make_vals()
  math.randomseed(13)
  local c1, c2, garbage = fvec.create(4), fvec.create(4), fvec.create(4)
  for i = 0, 3 do
    c1:set(i, math.random() + 0.1)
    c2:set(i, math.random() + 0.1)
    garbage:set(i, math.random() * 20 + 0.1)
  end
  local blA = { { x = range_csr(vals, 0, 4), n_tokens = 4, scale = 1.0, colscale = c1 },
                { x = range_csr(vals, 4, 8), n_tokens = 4, scale = scale2, colscale = c2 } }
  local lms = landmarks(6)
  local fit = { blocks = blA, landmarks = lms, n_landmarks = lms:size() }
  for k, v in pairs(kernel_args) do fit[k] = v end
  local _, enc = spectral.encode(fit)
  local A = enc:encode({ blocks = blA })
  local blG = { { x = range_csr(vals, 0, 4), colscale = garbage },
                { x = range_csr(vals, 4, 8), colscale = garbage } }
  local blN = { { x = range_csr(vals, 0, 4) }, { x = range_csr(vals, 4, 8) } }
  assert_codes_close(A, enc:encode({ blocks = blG }), tag .. " (garbage colscale ignored)")
  assert_codes_close(A, enc:encode({ blocks = blN }), tag .. " (no colscale ignored)")
end

test("enc:encode ignores the encode-time colscale (fit colscale is authoritative)", function ()
  ignores_colscale({ kernel = "cosine" }, 1.0, "cosine scale=1")
  ignores_colscale({ kernel = "cosine" }, 22.0, "cosine scale=22")
  ignores_colscale({ kernel = "matern", nu = 2, gamma = 0.25 }, 1.0, "matern scale=1")
  ignores_colscale({ kernel = "matern", nu = 2, gamma = 0.25 }, 22.0, "matern scale=22")
end)

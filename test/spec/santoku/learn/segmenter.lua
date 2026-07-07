local segmenter = require("santoku.learn.segmenter")
local ivec = require("santoku.ivec")
local test = require("santoku.test")

local texts = {
  "the cat sat on the mat",
  "a dog ran fast",
  "the red fox jumped over the lazy dog",
  "cats and dogs run in the park",
}

local function iv (t)
  local v = ivec.create()
  for _, x in ipairs(t) do v:push(x) end
  return v
end

local function iveq (a, b)
  if a:size() ~= b:size() then return false end
  for i = 0, a:size() - 1 do
    if a:get(i) ~= b:get(i) then return false end
  end
  return true
end

local function word_gold (ts)
  local off, st, en = { 0 }, {}, {}
  for d, t in ipairs(ts) do
    local i = 1
    while true do
      local s, e = t:find("%S+", i)
      if not s then break end
      st[#st + 1] = s - 1
      en[#en + 1] = e
      i = e + 1
    end
    off[d + 1] = #st
  end
  return iv(off), iv(st), iv(en)
end

test("segmenter", function ()

  test("train/segment/persist round-trip", function ()
    local go, gs, ge = word_gold(texts)
    local seg = segmenter.create({ context = "left" })
    local ck, rec = seg:train({ texts = texts, n = #texts,
      gold_offsets = go, gold_starts = gs, gold_ends = ge })
    assert(ck >= 1)
    assert(rec >= 0 and rec <= 1)
    local curve, nseen = seg:compression_curve({ texts = texts, n = #texts })
    assert(nseen >= 1 and #curve == nseen)
    local off, st, en, cl = seg:segment({ texts = texts, n = #texts })
    assert(off:size() == #texts + 1)
    assert(st:size() > 0 and st:size() == en:size() and st:size() == cl:size())
    local off2, st2, en2, cl2 = seg:segment({ texts = texts, n = #texts })
    assert(iveq(off, off2) and iveq(st, st2) and iveq(en, en2) and iveq(cl, cl2))
    local tmp = os.tmpname()
    seg:persist(tmp)
    local seg2 = segmenter.load(tmp)
    os.remove(tmp)
    local off3, st3, en3, cl3 = seg2:segment({ texts = texts, n = #texts })
    assert(iveq(off, off3) and iveq(st, st3) and iveq(en, en3) and iveq(cl, cl3))
  end)

  test("context both", function ()
    local go, gs, ge = word_gold(texts)
    local seg = segmenter.create({ context = "both" })
    local ck = seg:train({ texts = texts, n = #texts,
      gold_offsets = go, gold_starts = gs, gold_ends = ge })
    assert(ck >= 1)
    local off, st, en, cl = seg:segment({ texts = texts, n = #texts, drop_sep = true })
    assert(off:size() == #texts + 1)
    assert(st:size() > 0 and st:size() == en:size() and st:size() == cl:size())
    for i = 0, st:size() - 1 do
      assert(en:get(i) > st:get(i))
    end
  end)

end)

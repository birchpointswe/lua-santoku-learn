local tokenizer = require("santoku.learn.tokenizer")
local re = require("santoku.re")
local tbl = require("santoku.table")
local test = require("santoku.test")
local teq = tbl.equals

test("tokenizer.extract", function ()

  test("word runs, raw byte spans, parallel over docs", function ()
    local prog = re.prog("[A-Za-z0-9]+")
    local off, s, e = tokenizer.extract({
      n = 3, texts = { "U.N. official", "", "ab cd12" }, pattern = prog })
    -- doc1: "U"[0,1) "N"[2,3) "official"[5,13); doc2: none; doc3: "ab"[0,2) "cd12"[3,7)
    assert(teq(off:table(), { 0, 3, 3, 5 }))
    assert(teq(s:table(), { 0, 2, 5, 0, 3 }))
    assert(teq(e:table(), { 1, 3, 13, 2, 7 }))
  end)

  test("named groups tag in one pass, untagged = n_tags", function ()
    local P = "{:caps: [A-Z]+ :} ![A-Za-z0-9]"
      .. " / {:num: [0-9]+ :} ![A-Za-z0-9]"
      .. " / [A-Za-z0-9]+ / [^A-Za-z0-9 ]+"
    local tags = re.tags(P)
    local prog = re.prog(P)
    local off, s, e, ty = tokenizer.extract({
      n = 1, texts = { "UN 1996 x" }, pattern = prog })
    assert(teq(off:table(), { 0, 3 }))
    assert(teq(s:table(), { 0, 3, 8 }))
    assert(teq(e:table(), { 2, 7, 9 }))
    assert(teq(ty:table(), { tags.caps, tags.num, 2 }))   -- x hits an unnamed branch
  end)

  test("util.shape_spans: boundaries = word/punct runs, classes refine", function ()
    local util = require("santoku.learn.util")
    local sh = util.SHAPE_TAGS
    local texts = { "U.N. McGrath's 1990s (why) & 1996 12 iPhone F1" }
    local S = util.shape_spans(texts, 1)
    -- token boundaries identical to the plain word/punct segmentation
    local prog = re.prog("[A-Za-z0-9]+ / [^A-Za-z0-9%s]+")
    local off, s, e = tokenizer.extract({ n = 1, texts = texts, pattern = prog })
    assert(teq(S:offsets():table(), off:table()))
    assert(teq(S:col("s"):table(), s:table()))
    assert(teq(S:col("e"):table(), e:table()))
    assert(teq(S:col("ty"):table(), {
      sh.u1, sh.dot, sh.u1, sh.dot,             -- U . N .
      sh.icap, sh.apos, sh.low,                 -- McGrath ' s
      sh.dsuf,                                  -- 1990s
      sh.paren, sh.low, sh.paren,               -- ( why )
      sh.amp,                                   -- &
      sh.d4, sh.d2,                             -- 1996 12
      sh.alnum, sh.alnum,                       -- iPhone F1
    }))
  end)

end)

local test = require("santoku.test")
local err = require("santoku.error")
local assert = err.assert
local spans = require("santoku.spans")
local ivec = require("santoku.ivec")
local ner = require("santoku.learn.ner")
require("santoku.csr")  -- installs csr metatable (__index) for the gaz:block round-trip below
local tbl = require("santoku.table")
local teq = tbl.equals

test("ner: type_labels", function ()
  local C = spans.create({ "s", "e" })
  C:push(0, 2):push(0, 5):push(2, 5):doc()
  local G = spans.create({ "s", "e", "ty" })
  G:push(0, 2, 1):push(2, 5, 3):doc()
  local lab = C:type_labels(G, 4)
  assert(teq(lab:table(), { 1, 4, 3 }))   -- (0,5) matches no gold -> reject(=4)
end)

test("ner: miss_report", function ()
  local gaz = spans.create({ "s", "e", "ty" })
  gaz:doc()                                -- one empty doc
  local bio = spans.create({ "s", "e", "ty" })
  bio:push(0, 2, 0):push(6, 8, 0):doc()
  local gold = spans.create({ "s", "e", "ty" })
  gold:push(0, 2, 1):push(5, 9, 2):doc()
  local m = ner.miss_report({ gaz = gaz, bio = bio, gold = gold, n_types = 4 })
  assert(m.gold == 2)
  assert(m.covered == 0)
  assert(m.wrong_type == 1)                -- bio (0,2) exact boundary, wrong type for gold (0,2,1)
  assert(m.under == 1 and m.under_bio == 1 and m.under_gaz == 0)  -- bio (6,8) inside gold (5,9)
  assert(m.under_by_type[2] == 1)
end)

test("ner: decode_report", function ()
  local C = spans.create({ "s", "e" })
  C:push(0, 2):push(5, 9):push(3, 4):doc()
  local G = spans.create({ "s", "e", "ty" })
  G:push(0, 2, 1):push(5, 9, 2):doc()
  local pred = ivec.create({ 1, 4, 0 })    -- stride 1: cand0 -> type1, cand1 -> reject(4), cand2 unused
  local r = ner.decode_report({ cand = C, gold = G, pred = pred, pred_stride = 1, n_types = 4 })
  assert(r.gold == 2 and r.in_pool == 2 and r.not_in_pool == 0)
  assert(r.correct == 1 and r.false_reject == 1 and r.mistype == 0)
  assert(r.reject_by_type[2] == 1)
end)

test("ner: char gaz persist round-trip", function ()
  local texts = { "Barack Obama visited Paris", "Paris Hilton and Obama" }
  local gold = spans.create({ "s", "e", "ty" })
  gold:push(7, 12, 0):push(21, 26, 1):doc()  -- doc1: Obama(PER=0), Paris(LOC=1)
  gold:push(0, 5, 1):push(17, 22, 0):doc()   -- doc2: Paris(LOC=1), Obama(PER=0)
  local gaz = ner.build_char_gaz({ texts = texts, gold = gold,
    n_types = 2, ngram_min = 3, ngram_max = 5 })
  local cand = spans.create({ "s", "e" })
  cand:push(7, 12):push(21, 26):doc()
  cand:push(0, 5):push(17, 22):doc()
  local A = gaz:block(texts, cand, nil)
  local path = os.tmpname() .. ".gaz"
  gaz:persist(path)
  local B = ner.load_gaz(path):block(texts, cand, nil)
  os.remove(path)
  assert(A:eq(B), "gaz block diverges after persist/load")
end)

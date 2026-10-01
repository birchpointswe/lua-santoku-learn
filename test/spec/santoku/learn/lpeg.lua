-- SPDX-License-Identifier: MIT
-- SPDX-FileCopyrightText: 2024 Birch Point SWE
local test = require("santoku.test")
local lp = require("santoku.lpeg")

test("annotation flow", function ()

  test("extract + spans + match_tags + inject", function ()
    local aho = require("santoku.learn.aho")
    local ivec = require("santoku.ivec")
    local html = 'hello <span class="gold">world</span> foo bar baz'
    local text, existing = lp.html_extract(html)
    assert(text == "hello world foo bar baz")
    local ids = ivec.create({ 1, 2, 3 })
    local ac = aho.create({ ids = ids, patterns = { "world", "foo", "baz" }, names = { "w", "f", "b" } })
    local S = ac:predict({
      texts = { text }, longest = true,
      exclude = lp.html_spans(existing)
    })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 2)
    local pred = lp.html_match_tags(mids, starts, ends, { [1] = "f", [2] = "b" }, "predicted ")
    for _, t in ipairs(pred) do existing[#existing + 1] = t end
    local result = lp.html_inject(text, existing)
    assert(result:find("gold"))
    assert(result:find("predicted f"))
    assert(result:find("predicted b"))
    assert(not result:find("predicted w"))
  end)

end)

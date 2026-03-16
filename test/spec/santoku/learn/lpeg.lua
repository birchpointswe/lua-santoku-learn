local test = require("santoku.test")
local lp = require("santoku.learn.lpeg")

test("json_fields", function ()

  test("simple string fields", function ()
    local line = '{"title":"hello","body":"world","other":123}'
    local results = {}
    for s, e in lp.json_fields(line, {"title", "body"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 2)
    assert(results[1] == "hello")
    assert(results[2] == "world")
  end)

  test("array values", function ()
    local line = '{"tags":["a","b","c"],"x":1}'
    local results = {}
    for s, e in lp.json_fields(line, {"tags"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 3)
    assert(results[1] == "a")
    assert(results[2] == "b")
    assert(results[3] == "c")
  end)

  test("nested objects skipped", function ()
    local line = '{"a":"yes","nested":{"a":"no"},"b":"also"}'
    local results = {}
    for s, e in lp.json_fields(line, {"a", "b"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 2)
    assert(results[1] == "yes")
    assert(results[2] == "also")
  end)

  test("escaped strings", function ()
    local line = '{"val":"hello\\"world"}'
    local results = {}
    for s, e in lp.json_fields(line, {"val"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 1)
    assert(results[1] == 'hello\\"world')
  end)

  test("empty string skipped", function ()
    local line = '{"a":"","b":"ok"}'
    local results = {}
    for s, e in lp.json_fields(line, {"a", "b"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 1)
    assert(results[1] == "ok")
  end)

  test("missing fields", function ()
    local line = '{"x":1}'
    local results = {}
    for s, e in lp.json_fields(line, {"title"}) do
      results[#results + 1] = line:sub(s, e)
    end
    assert(#results == 0)
  end)

end)

test("html_text", function ()

  test("simple tags", function ()
    local html = "<p>hello</p> <b>world</b>"
    local results = {}
    for text in lp.html_text(html) do
      results[#results + 1] = text
    end
    assert(#results == 2)
    assert(results[1] == "hello")
    assert(results[2] == " world")
  end)

  test("script and style stripped", function ()
    local html = "before<script>var x=1;</script>after<style>.a{}</style>end"
    local results = {}
    for text in lp.html_text(html) do
      results[#results + 1] = text
    end
    assert(#results == 3)
    assert(results[1] == "before")
    assert(results[2] == "after")
    assert(results[3] == "end")
  end)

  test("comments stripped", function ()
    local html = "a<!-- comment -->b"
    local results = {}
    for text in lp.html_text(html) do
      results[#results + 1] = text
    end
    assert(#results == 2)
    assert(results[1] == "a")
    assert(results[2] == "b")
  end)

end)

test("html_extract", function ()

  test("strips tags and tracks positions", function ()
    local html = 'hello <span class="author">John</span> world'
    local text, tags = lp.html_extract(html)
    assert(text == "hello John world")
    assert(#tags == 1)
    assert(tags[1].name == "span")
    assert(tags[1].attrs["class"] == "author")
    assert(text:sub(tags[1].s, tags[1].e) == "John")
  end)

  test("multiple tags", function ()
    local html = '<span class="title">BOOK</span> by <span class="author">Smith</span>'
    local text, tags = lp.html_extract(html)
    assert(text == "BOOK by Smith")
    assert(#tags == 2)
    assert(tags[1].attrs["class"] == "title")
    assert(tags[2].attrs["class"] == "author")
    assert(text:sub(tags[1].s, tags[1].e) == "BOOK")
    assert(text:sub(tags[2].s, tags[2].e) == "Smith")
  end)

end)

test("html_tags", function ()

  test("iterates tags", function ()
    local html = '<b>bold</b> and <i x="1">italic</i>'
    local results = {}
    for name, attrs in lp.html_tags(html) do
      results[#results + 1] = { name = name, attrs = attrs }
    end
    assert(#results == 2)
    assert(results[1].name == "b")
    assert(results[2].name == "i")
    assert(results[2].attrs.x == "1")
  end)

end)

test("html_spans", function ()

  test("converts tag positions to aho spans", function ()
    local tags = {
      { s = 1, e = 4 },
      { s = 10, e = 15 },
    }
    local spans = lp.html_spans(tags)
    assert(spans:size() == 2)
    local a0, b0 = spans:get(0)
    assert(a0 == 0)
    assert(b0 == 4)
    local a1, b1 = spans:get(1)
    assert(a1 == 9)
    assert(b1 == 15)
  end)

end)

test("html_match_tags", function ()

  test("converts ivec results to tag records", function ()
    local mock = {
      _data = {},
      size = function (self) return #self._data end,
      get = function (self, i) return self._data[i + 1] end,
    }
    local ids = setmetatable({ _data = { 10, 20 } }, { __index = mock })
    local starts = setmetatable({ _data = { 0, 8 } }, { __index = mock })
    local ends = setmetatable({ _data = { 4, 15 } }, { __index = mock })
    local names = { [10] = "alpha", [20] = "beta" }
    local tags = lp.html_match_tags(ids, starts, ends, names, "pred ")
    assert(#tags == 2)
    assert(tags[1].name == "span")
    assert(tags[1].s == 1)
    assert(tags[1].e == 4)
    assert(tags[1].attrs.class == "pred alpha")
    assert(tags[2].s == 9)
    assert(tags[2].e == 15)
    assert(tags[2].attrs.class == "pred beta")
  end)

  test("works without names or prefix", function ()
    local mock = {
      _data = {},
      size = function (self) return #self._data end,
      get = function (self, i) return self._data[i + 1] end,
    }
    local ids = setmetatable({ _data = { 5 } }, { __index = mock })
    local starts = setmetatable({ _data = { 2 } }, { __index = mock })
    local ends = setmetatable({ _data = { 7 } }, { __index = mock })
    local tags = lp.html_match_tags(ids, starts, ends)
    assert(#tags == 1)
    assert(tags[1].attrs.class == "5")
    assert(tags[1].s == 3)
    assert(tags[1].e == 7)
  end)

end)

test("html_inject", function ()

  test("round-trip extract then inject", function ()
    local html = 'hello <span class="author">John</span> world'
    local text, tags = lp.html_extract(html)
    local rebuilt = lp.html_inject(text, tags)
    assert(rebuilt:find("John"))
    assert(rebuilt:find("hello"))
    assert(rebuilt:find("world"))
    assert(rebuilt:find("span"))
  end)

  test("canonicalization via text override", function ()
    local html = '<span class="author">J. Smith</span>'
    local text, tags = lp.html_extract(html)
    assert(text == "J. Smith")
    tags[1].text = "John Smith"
    local rebuilt = lp.html_inject(text, tags)
    assert(rebuilt:find("John Smith"))
    assert(not rebuilt:find("J%. Smith"))
  end)

end)

test("annotation flow", function ()

  test("extract + spans + match_tags + inject", function ()
    local aho = require("santoku.learn.aho")
    local ivec = require("santoku.ivec")
    local html = 'hello <span class="gold">world</span> foo bar baz'
    local text, existing = lp.html_extract(html)
    assert(text == "hello world foo bar baz")
    local ids = ivec.create({ 1, 2, 3 })
    local ac = aho.create({ ids = ids, patterns = { "world", "foo", "baz" }, names = { "w", "f", "b" } })
    local _, mids, starts, ends = ac:predict({
      texts = { text }, longest = true,
      exclude = lp.html_spans(existing)
    })
    assert(mids:size() == 2)
    local pred = lp.html_match_tags(mids, starts, ends, { [2] = "f", [3] = "b" }, "predicted ")
    for _, t in ipairs(pred) do existing[#existing + 1] = t end
    local result = lp.html_inject(text, existing)
    assert(result:find("gold"))
    assert(result:find("predicted f"))
    assert(result:find("predicted b"))
    assert(not result:find("predicted w"))
  end)

end)

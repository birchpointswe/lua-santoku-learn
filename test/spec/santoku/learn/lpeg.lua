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
    for s, e in lp.html_text(html) do
      results[#results + 1] = html:sub(s, e)
    end
    assert(#results == 2)
    assert(results[1] == "hello")
    assert(results[2] == " world")
  end)

  test("script and style stripped", function ()
    local html = "before<script>var x=1;</script>after<style>.a{}</style>end"
    local results = {}
    for s, e in lp.html_text(html) do
      results[#results + 1] = html:sub(s, e)
    end
    assert(#results == 3)
    assert(results[1] == "before")
    assert(results[2] == "after")
    assert(results[3] == "end")
  end)

  test("comments stripped", function ()
    local html = "a<!-- comment -->b"
    local results = {}
    for s, e in lp.html_text(html) do
      results[#results + 1] = html:sub(s, e)
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

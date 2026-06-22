require("santoku.error")
local aho = require("santoku.learn.aho")
local pvec = require("santoku.pvec")
local test = require("santoku.test")


test("aho", function ()

  test("basic matching", function ()
    local ac = aho.create({ patterns = { "foo", "bar", "baz" } })
    local S = ac:predict({ texts = { "foo bar baz" } })
    local offsets, mids, starts, ends = S:offsets(), S:col("id"), S:col("s"), S:col("e")
    assert(offsets:size() == 2)
    assert(offsets:get(0) == 0)
    assert(offsets:get(1) == 3)
    assert(mids:size() == 3)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 3)
    assert(mids:get(1) == 1)
    assert(starts:get(1) == 4)
    assert(ends:get(1) == 7)
    assert(mids:get(2) == 2)
    assert(starts:get(2) == 8)
    assert(ends:get(2) == 11)
  end)

  test("case insensitive", function ()
    local ac = aho.create({ patterns = { "hello" }, normalize = true })
    local S = ac:predict({ texts = { "HELLO world" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 5)
  end)

  test("case insensitive pattern", function ()
    local ac = aho.create({ patterns = { "WORLD" }, normalize = true })
    local S = ac:predict({ texts = { "hello world" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 6)
    assert(ends:get(0) == 11)
  end)

  test("diacritic text matches plain pattern", function ()
    local ac = aho.create({ patterns = { "cafe" }, normalize = true })
    local S = ac:predict({ texts = { "caf\195\169" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 5)
  end)

  test("diacritic pattern matches plain text", function ()
    local ac = aho.create({ patterns = { "caf\195\169" }, normalize = true })
    local S = ac:predict({ texts = { "cafe latte" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 4)
  end)

  test("diacritic both sides", function ()
    local ac = aho.create({ patterns = { "r\195\169sum\195\169" }, normalize = true })
    local S = ac:predict({ texts = { "my R\195\137SUM\195\137 here" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 3)
    assert(ends:get(0) == 11)
  end)

  test("position after multibyte prefix", function ()
    local ac = aho.create({ patterns = { "world" }, normalize = true })
    local S = ac:predict({ texts = { "h\195\169llo world" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 7)
    assert(ends:get(0) == 12)
  end)

  test("longest match filtering", function ()
    local ac = aho.create({ patterns = { "new", "new york" } })
    local S = ac:predict({ texts = { "new york city" }, longest = true })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 1)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 8)
  end)

  test("overlapping without longest", function ()
    local ac = aho.create({ patterns = { "ab", "abc" } })
    local S = ac:predict({ texts = { "abcd" } })
    assert(S:col("id"):size() == 2)
  end)

  test("multiple texts", function ()
    local ac = aho.create({ patterns = { "hi" } })
    local S = ac:predict({ texts = { "hi there", "say hi", "none" } })
    local offsets = S:offsets()
    assert(offsets:size() == 4)
    assert(offsets:get(0) == 0)
    assert(offsets:get(1) == 1)
    assert(offsets:get(2) == 2)
    assert(offsets:get(3) == 2)
    assert(S:col("id"):size() == 2)
  end)

  test("no matches", function ()
    local ac = aho.create({ patterns = { "xyz" } })
    local S = ac:predict({ texts = { "abc def" } })
    local offsets = S:offsets()
    assert(S:col("id"):size() == 0)
    assert(offsets:get(0) == 0)
    assert(offsets:get(1) == 0)
  end)


  test("latin extended-a", function ()
    local ac = aho.create({ patterns = { "scev" }, normalize = true })
    local S = ac:predict({ texts = { "\197\161\196\141ev" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 6)
  end)

  test("combining marks dropped", function ()
    local ac = aho.create({ patterns = { "cafe" }, normalize = true })
    local S = ac:predict({ texts = { "cafe\204\129" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 1)
    assert(mids:get(0) == 0)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 6)
  end)

  test("empty text", function ()
    local ac = aho.create({ patterns = { "x" } })
    local S = ac:predict({ texts = { "" } })
    local offsets = S:offsets()
    assert(S:col("id"):size() == 0)
    assert(offsets:get(0) == 0)
    assert(offsets:get(1) == 0)
  end)

  test("multiple matches same text", function ()
    local ac = aho.create({ patterns = { "ab" } })
    local S = ac:predict({ texts = { "ab ab ab" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 3)
    assert(starts:get(0) == 0)
    assert(ends:get(0) == 2)
    assert(starts:get(1) == 3)
    assert(ends:get(1) == 5)
    assert(starts:get(2) == 6)
    assert(ends:get(2) == 8)
  end)

  test("tag basic", function ()
    local ac = aho.create({ patterns = { "foo", "bar" } })
    local result = ac:tag({ texts = { "foo and bar" }, fmt = "[%match:%id]" })
    assert(result[1] == "[foo:0] and [bar:1]")
  end)

  test("tag with names", function ()
    local ac = aho.create({
      patterns = { "nyc", "la" },
      names = { "New York City", "Los Angeles" }
    })
    local result = ac:tag({ texts = { "visit nyc or la" }, fmt = '<span title="%name">%match</span>' })
    assert(result[1] == 'visit <span title="New York City">nyc</span> or <span title="Los Angeles">la</span>')
  end)

  test("tag preserves original case", function ()
    local ac = aho.create({ patterns = { "hello" }, normalize = true })
    local result = ac:tag({ texts = { "say HELLO there" }, fmt = "[%match]" })
    assert(result[1] == "say [HELLO] there")
  end)

  test("tag preserves diacritics in match", function ()
    local ac = aho.create({ patterns = { "cafe" }, normalize = true })
    local result = ac:tag({ texts = { "at caf\195\169 now" }, fmt = "[%match]" })
    assert(result[1] == "at [caf\195\169] now")
  end)

  test("tag no matches returns original", function ()
    local ac = aho.create({ patterns = { "xyz" } })
    local result = ac:tag({ texts = { "abc def" }, fmt = "[%match]" })
    assert(result[1] == "abc def")
  end)

  test("tag multiple texts", function ()
    local ac = aho.create({ patterns = { "hi" } })
    local result = ac:tag({ texts = { "hi there", "say hi" }, fmt = "(%match)" })
    assert(result[1] == "(hi) there")
    assert(result[2] == "say (hi)")
  end)

  test("tag literal percent", function ()
    local ac = aho.create({ patterns = { "x" } })
    local result = ac:tag({ texts = { "x" }, fmt = "100%% %match" })
    assert(result[1] == "100% x")
  end)

  test("tag longest filtering", function ()
    local ac = aho.create({
      patterns = { "new", "new york" },
      names = { "new", "New York" }
    })
    local result = ac:tag({ texts = { "new york city" }, fmt = "[%name]", longest = true })
    assert(result[1] == "[New York] city")
  end)

  test("tag html escape match", function ()
    local ac = aho.create({ patterns = { "a&b" } })
    local result = ac:tag({ texts = { "see a&b here" }, fmt = "<b>%hmatch</b>" })
    assert(result[1] == "see <b>a&amp;b</b> here")
  end)

  test("tag html escape name", function ()
    local ac = aho.create({
      patterns = { "foo" },
      names = { '<script>alert("x")</script>' }
    })
    local result = ac:tag({ texts = { "foo" }, fmt = '%hname' })
    assert(result[1] == '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;')
  end)

  test("tag html escape vs raw", function ()
    local ac = aho.create({ patterns = { "x<y" } })
    local result = ac:tag({ texts = { "x<y" }, fmt = "%match|%hmatch" })
    assert(result[1] == "x<y|x&lt;y")
  end)

  test("tag html escape single quotes", function ()
    local ac = aho.create({
      patterns = { "it" },
      names = { "it's" }
    })
    local result = ac:tag({ texts = { "it works" }, fmt = "<span title='%hname'>%hmatch</span>" })
    assert(result[1] == "<span title='it&#39;s'>it</span> works")
  end)


  test("predict with exclude", function ()
    local ac = aho.create({ patterns = { "foo", "bar", "baz" } })
    local exc = pvec.create()
    exc:push(0, 3)
    local S = ac:predict({
      texts = { "foo bar baz" },
      exclude = exc
    })
    local mids = S:col("id")
    assert(mids:size() == 2)
    assert(mids:get(0) == 1)
    assert(mids:get(1) == 2)
  end)

  test("tag with exclude", function ()
    local ac = aho.create({ patterns = { "foo", "bar" } })
    local exc = pvec.create()
    exc:push(0, 3)
    local result = ac:tag({
      texts = { "foo and bar" },
      fmt = "[%match]",
      exclude = exc
    })
    assert(result[1] == "foo and [bar]")
  end)

  test("exclude partial overlap drops match", function ()
    local ac = aho.create({ patterns = { "bar baz" } })
    local exc = pvec.create()
    exc:push(0, 5)
    local S = ac:predict({
      texts = { "foo bar baz" },
      exclude = exc
    })
    assert(S:col("id"):size() == 0)
  end)

  test("exclude_hits returned for recognized regions", function ()
    local ac = aho.create({ patterns = { "foo", "bar", "baz" } })
    local exc = pvec.create()
    exc:push(0, 3)
    exc:push(4, 7)
    exc:push(8, 11)
    local S, exc_hits = ac:predict({
      texts = { "foo bar baz" },
      exclude = exc
    })
    assert(S:col("id"):size() == 0)
    assert(exc_hits:size() == 3)
    assert(exc_hits:get(0) == 1)
    assert(exc_hits:get(1) == 1)
    assert(exc_hits:get(2) == 1)
  end)

  test("exclude_hits zero for unrecognized regions", function ()
    local ac = aho.create({ patterns = { "foo" } })
    local exc = pvec.create()
    exc:push(0, 3)
    exc:push(4, 11)
    local S, exc_hits = ac:predict({
      texts = { "foo hello world" },
      exclude = exc
    })
    assert(S:col("id"):size() == 0)
    assert(exc_hits:size() == 2)
    assert(exc_hits:get(0) == 1)
    assert(exc_hits:get(1) == 0)
  end)

  test("exclude_hits not returned when no exclude", function ()
    local ac = aho.create({ patterns = { "foo" } })
    local S, exc_hits = ac:predict({
      texts = { "foo bar" },
    })
    assert(S:col("id"):size() == 1)
    assert(exc_hits == nil)
  end)

  test("exclude_hits with mixed recognized and unrecognized", function ()
    local ac = aho.create({ patterns = { "Alice", "Bob" } })
    local exc = pvec.create()
    exc:push(0, 5)
    exc:push(10, 17)
    exc:push(21, 24)
    local S, exc_hits = ac:predict({
      texts = { "Alice and Unknown and Bob" },
      exclude = exc
    })
    assert(S:col("id"):size() == 0)
    assert(exc_hits:size() == 3)
    assert(exc_hits:get(0) == 1)
    assert(exc_hits:get(1) == 0)
    assert(exc_hits:get(2) == 1)
  end)

  test("persist/load roundtrip", function ()
    local ac = aho.create({ patterns = { "foo", "bar", "baz" } })
    local tmp = os.tmpname()
    ac:persist(tmp)
    local ac2 = aho.load(tmp)
    os.remove(tmp)
    local S = ac2:predict({ texts = { "foo bar baz" } })
    local mids, starts, ends = S:col("id"), S:col("s"), S:col("e")
    assert(mids:size() == 3)
    assert(mids:get(0) == 0 and starts:get(0) == 0 and ends:get(0) == 3)
    assert(mids:get(1) == 1 and starts:get(1) == 4 and ends:get(1) == 7)
    assert(mids:get(2) == 2 and starts:get(2) == 8 and ends:get(2) == 11)
  end)

end)

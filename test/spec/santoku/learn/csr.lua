local csr = require("santoku.learn.csr")
local aho = require("santoku.learn.aho")
local ivec = require("santoku.ivec")
local cvec = require("santoku.cvec")
local test = require("santoku.test")

test("csr", function ()
  test("tokenize_annotated", function ()

    test("basic", function ()
      local texts = { "hello world" }
      local doc_span_offsets = ivec.create({ 0, 1 })
      local span_starts = ivec.create({ 0 })
      local span_ends = ivec.create({ 5 })
      local ngram_map, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
      })
      assert(offsets:size() == 2)
      assert(offsets:get(0) == 0)
      assert(offsets:get(1) > 0)
      assert(tokens:size() > 0)
      assert(n_tokens > 0)
      assert(ngram_map ~= nil)
    end)

    test("multiple spans", function ()
      local texts = { "foo bar baz" }
      local doc_span_offsets = ivec.create({ 0, 3 })
      local span_starts = ivec.create({ 0, 4, 8 })
      local span_ends = ivec.create({ 3, 7, 11 })
      local _, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
      })
      assert(offsets:size() == 4)
      assert(offsets:get(0) == 0)
      for i = 0, 2 do
        assert(offsets:get(i + 1) > offsets:get(i))
      end
      assert(tokens:size() == offsets:get(3))
      assert(n_tokens > 0)
    end)

    test("multiple docs", function ()
      local texts = { "abc def", "xyz" }
      local doc_span_offsets = ivec.create({ 0, 2, 3 })
      local span_starts = ivec.create({ 0, 4, 0 })
      local span_ends = ivec.create({ 3, 7, 3 })
      local _, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
      })
      assert(offsets:size() == 4)
      assert(tokens:size() == offsets:get(3))
      assert(n_tokens > 0)
    end)

    test("with aho predict", function ()
      local ids = ivec.create({ 1, 2 })
      local ac = aho.create({ ids = ids, patterns = { "foo", "bar" } })
      local texts = { "foo and bar" }
      local doc_span_offsets, _, span_starts, span_ends = ac:predict({ texts = texts })
      local _, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 5,
      })
      assert(offsets:size() == 3)
      assert(tokens:size() > 0)
      assert(n_tokens > 0)
    end)

    test("existing ngram_map", function ()
      local texts = { "hello world" }
      local doc_span_offsets = ivec.create({ 0, 1 })
      local span_starts = ivec.create({ 0 })
      local span_ends = ivec.create({ 5 })
      local args = {
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
      }
      local ngram_map, offsets1, tokens1, _, n_tokens1 = csr.tokenize_annotated(args)
      args.ngram_map = ngram_map
      local _, offsets2, tokens2, _, n_tokens2 = csr.tokenize_annotated(args)
      assert(n_tokens1 == n_tokens2)
      assert(offsets1:get(1) == offsets2:get(1))
      for i = 0, tokens1:size() - 1 do
        assert(tokens1:get(i) == tokens2:get(i))
      end
    end)

    test("empty spans", function ()
      local texts = { "hello world" }
      local doc_span_offsets = ivec.create({ 0, 0 })
      local _, offsets, tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = ivec.create(),
        span_ends = ivec.create(),
        ngram_min = 3,
        ngram_max = 3,
      })
      assert(offsets:size() == 1)
      assert(offsets:get(0) == 0)
      assert(tokens:size() == 0)
    end)

    test("terminals", function ()
      local texts = { "hello world" }
      local doc_span_offsets = ivec.create({ 0, 1 })
      local span_starts = ivec.create({ 0 })
      local span_ends = ivec.create({ 5 })
      local _, off_no = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
      })
      local _, off_t, tokens_t, _, n_tokens_t = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
        terminals = true,
      })
      assert(off_t:size() == 2)
      assert(tokens_t:size() > 0)
      assert(n_tokens_t > 0)
      assert(off_t:get(1) > off_no:get(1))
    end)

    test("normalize", function ()
      local texts = { "Hello World" }
      local doc_span_offsets = ivec.create({ 0, 1 })
      local span_starts = ivec.create({ 0 })
      local span_ends = ivec.create({ 5 })
      local _, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts,
        doc_span_offsets = doc_span_offsets,
        span_starts = span_starts,
        span_ends = span_ends,
        ngram_min = 3,
        ngram_max = 3,
        normalize = true,
      })
      assert(offsets:size() == 2)
      assert(tokens:size() > 0)
      assert(n_tokens > 0)
    end)

    test("collapse modes differ", function ()
      local texts = { "foo bar baz qux" }
      local doc = ivec.create({ 0, 2 })
      local ss = ivec.create({ 0, 8 })
      local se = ivec.create({ 3, 11 })
      local function total (mode)
        local _, off = csr.tokenize_annotated({
          texts = texts, doc_span_offsets = doc,
          span_starts = ss, span_ends = se,
          ngram_min = 3, ngram_max = 3, collapse = mode,
        })
        return off:get(2)
      end
      local none = total("none")
      assert(none > 0)
      assert(total("focus") > 0)
      assert(total("all") > 0)
      assert(total("all") < none)
      assert(total("spans") ~= none)
    end)

    test("span_types", function ()
      local texts = { "foo bar" }
      local doc = ivec.create({ 0, 2 })
      local ss = ivec.create({ 0, 4 })
      local se = ivec.create({ 3, 7 })
      local types = ivec.create({ 0, 1 })
      local _, offsets, tokens, _, n_tokens = csr.tokenize_annotated({
        texts = texts, doc_span_offsets = doc,
        span_starts = ss, span_ends = se, span_types = types,
        ngram_min = 3, ngram_max = 3,
      })
      assert(offsets:size() == 3)
      assert(tokens:size() > 0)
      assert(n_tokens > 0)
    end)

    test("too many span types without markers errors", function ()
      local texts = { "foo bar" }
      local doc = ivec.create({ 0, 1 })
      local ss = ivec.create({ 0 })
      local se = ivec.create({ 3 })
      local types = ivec.create({ 20 }) -- needs 45 marker bytes; default has 27
      local ok = pcall(function ()
        csr.tokenize_annotated({
          texts = texts, doc_span_offsets = doc,
          span_starts = ss, span_ends = se, span_types = types,
          ngram_min = 3, ngram_max = 3,
        })
      end)
      assert(not ok)
    end)

    test("supplied markers string works", function ()
      local texts = { "foo bar baz" }
      local doc = ivec.create({ 0, 1 })
      local ss = ivec.create({ 0 })
      local se = ivec.create({ 3 })
      local types = ivec.create({ 0 })
      local markers = string.char(200, 201, 202, 203, 204)
      local _, offsets, tokens = csr.tokenize_annotated({
        texts = texts, doc_span_offsets = doc, span_starts = ss, span_ends = se,
        span_types = types, markers = markers, ngram_min = 3, ngram_max = 3,
      })
      assert(offsets:size() == 2)
      assert(tokens:size() > 0)
    end)

    test("cvec input matches texts", function ()
      local text = "hello world"
      local doc = ivec.create({ 0, 1 })
      local ss = ivec.create({ 0 })
      local se = ivec.create({ 5 })
      local _, o1, t1 = csr.tokenize_annotated({
        texts = { text }, doc_span_offsets = doc,
        span_starts = ss, span_ends = se, ngram_min = 3, ngram_max = 3,
      })
      local cv = cvec.from_raw(text)
      local so = ivec.create({ 0, #text })
      local _, o2, t2 = csr.tokenize_annotated({
        cvec = cv, sequence_offsets = so, doc_span_offsets = doc,
        span_starts = ss, span_ends = se, ngram_min = 3, ngram_max = 3,
      })
      assert(o1:get(1) == o2:get(1))
      assert(t1:size() == t2:size())
      for i = 0, t1:size() - 1 do assert(t1:get(i) == t2:get(i)) end
    end)

    test("collapse modes with context spans", function ()
      local text = "John Smith works at Acme Corp"
      local doc = ivec.create({ 0, 1 })
      local fs = ivec.create({ 0 })
      local fe = ivec.create({ 10 })   -- focus "John Smith"
      local ft = ivec.create({ 0 })
      local cdoc = ivec.create({ 0, 1 })
      local cs = ivec.create({ 20 })
      local ce = ivec.create({ 29 })   -- context "Acme Corp"
      local ct = ivec.create({ 1 })
      local function tok (mode)
        local _, off = csr.tokenize_annotated({
          texts = { text }, doc_span_offsets = doc,
          span_starts = fs, span_ends = fe, span_types = ft,
          context_offsets = cdoc, context_starts = cs, context_ends = ce, context_types = ct,
          collapse = mode, ngram_min = 2, ngram_max = 2,
        })
        return off:get(1)
      end
      local none, mark, spans, all = tok("none"), tok("mark"), tok("spans"), tok("all")
      assert(none > 0 and mark > 0 and spans > 0 and all > 0)
      assert(all < mark)       -- all drops all text
      assert(spans < mark)     -- spans drops context surface, mark keeps it
      assert(mark > none)      -- mark adds context markers
    end)

    test("mark without context falls back to focus set", function ()
      local text = "aaa bbb ccc"
      local doc = ivec.create({ 0, 2 })       -- 2 focus spans
      local ss = ivec.create({ 0, 8 })
      local se = ivec.create({ 3, 11 })
      local _, off, tokens = csr.tokenize_annotated({
        texts = { text }, doc_span_offsets = doc,
        span_starts = ss, span_ends = se,
        collapse = "mark", ngram_min = 2, ngram_max = 2,
      })
      assert(off:size() == 3)
      assert(tokens:size() > 0)
    end)

  end)
end)

local csr = require("santoku.learn.csr")
local aho = require("santoku.learn.aho")
local ivec = require("santoku.ivec")
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

  end)
end)

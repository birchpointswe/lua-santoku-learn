local tokenizer = require("santoku.learn.tokenizer")
local csr = require("santoku.csr")

local M = {}

M.featurizer = function (o)
  local tok = tokenizer.create({ ngram_min = o.ngram_min, ngram_max = o.ngram_max, normalize = true })
  local X = tok:fit({ texts = o.texts })
  local w, avgdl = X:bm25()
  X:normalize()
  return {
    X = X,
    transform = function (texts)
      local Y = tok:tokenize({ texts = texts })
      Y:bm25(w, avgdl)
      Y:normalize()
      return Y
    end,
  }
end

M.lexical = function (o)
  local util = require("santoku.learn.util")
  local cn = tokenizer.normalize(o.corpus_texts)
  local qn = tokenizer.normalize(o.query_texts)
  local tok = tokenizer.create({ ngram_min = 1, ngram_max = o.ngram_max or 1, mode = "words" })
  local X = tok:fit({ texts = cn, tokens = util.word_spans(cn, #cn) })
  local Q = tok:tokenize({ texts = qn, tokens = util.word_spans(qn, #qn) })
  return X, Q, tok
end

M.bm25_ranker = function (X, Q)
  local Xb = X:clone()
  local raw, buf = X:values(), Xb:values()
  local n = raw:size()
  return function (k1, b, depth)
    buf:copy(raw, 0, n, 0)
    Xb:bm25(k1, b)
    return Xb:topk(Q, depth or 100)
  end
end

M.rerank = function (o)
  local lex = o.candidates:clone():normalize("max")
  local sem = o.candidates:clone():dots(o.query_codes, o.doc_codes)
  return csr.fuse(lex, sem, { weights = { 1 - o.alpha, o.alpha } })
end

return M

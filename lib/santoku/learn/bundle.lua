local fs = require("santoku.fs")
local str = require("santoku.string")

local M = {}

M.persist = function (opts)
  local dir = opts.dir
  fs.mkdirp(dir)
  local toks = opts.tokenizers or {}
  for i = 1, #toks do
    toks[i]:persist(dir .. "/tokenizer_" .. i .. ".bin")
  end
  opts.encoder:persist(dir .. "/encoder.bin")
  opts.ridge:persist(dir .. "/ridge.bin")
  if opts.decider then
    opts.decider:persist(dir .. "/decider.bin")
  end
  if opts.gaz then
    opts.gaz:persist(dir .. "/gaz.bin")
  end
  if opts.gaz_rms then
    opts.gaz_rms:persist(dir .. "/gaz_rms.bin")
  end
  fs.writefile(dir .. "/manifest.lua", str.format(
    "return {\n  version = 3,\n  n_tokenizers = %d,\n  has_decider = %s,\n  has_gaz = %s,\n  has_gaz_rms = %s,\n}\n",
    #toks,
    opts.decider and "true" or "false",
    opts.gaz and "true" or "false",
    opts.gaz_rms and "true" or "false"))
end

M.load = function (dir)
  local tokenizer = require("santoku.learn.tokenizer")
  local spectral = require("santoku.learn.spectral")
  local ridge = require("santoku.learn.ridge")
  local decide = require("santoku.learn.decide")
  local fvec = require("santoku.fvec")
  local manifest = dofile(dir .. "/manifest.lua")
  local toks = {}
  for i = 1, manifest.n_tokenizers do
    toks[i] = tokenizer.load(dir .. "/tokenizer_" .. i .. ".bin")
  end
  local function wrap (ext)
    local bl = {}
    for i = 1, #ext do
      local e = ext[i]
      bl[i] = { x = type(e) == "table" and e.x or e }
    end
    return bl
  end
  local encoder = spectral.load(dir .. "/encoder.bin")
  local r = ridge.load(dir .. "/ridge.bin")
  local encode = function (ext, out, start, count)
    return encoder:encode({ blocks = wrap(ext), start = start, count = count }, out)
  end
  local decider = manifest.has_decider and decide.load(dir .. "/decider.bin") or nil
  local gaz = manifest.has_gaz and require("santoku.learn.ner").load_gaz(dir .. "/gaz.bin") or nil
  local gaz_rms = manifest.has_gaz_rms and fvec.load(dir .. "/gaz_rms.bin") or nil
  return { tokenizers = toks, encoder = encoder, ridge = r, decider = decider,
    gaz = gaz, gaz_rms = gaz_rms, encode = encode }
end

return M

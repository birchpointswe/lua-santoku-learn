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
  local fh = assert(io.open(dir .. "/manifest.lua", "w"))
  fh:write(str.format(
    "return {\n  version = 2,\n  n_tokenizers = %d,\n  has_decider = %s,\n  has_gaz = %s,\n  has_gaz_rms = %s,\n  w_external = %s,\n  w_path = %s,\n  chol_external = %s,\n  chol_path = %s,\n}\n",
    #toks,
    opts.decider and "true" or "false",
    opts.gaz and "true" or "false",
    opts.gaz_rms and "true" or "false",
    opts.w_path and "true" or "false",
    opts.w_path and str.format("%q", opts.w_path) or "nil",
    opts.chol_path and "true" or "false",
    opts.chol_path and str.format("%q", opts.chol_path) or "nil"))
  fh:close()
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
  local encoder
  if manifest.chol_external then
    encoder = spectral.load(dir .. "/encoder.bin", fvec.mmap_open(manifest.chol_path))
  else
    encoder = spectral.load(dir .. "/encoder.bin")
  end
  local r
  if manifest.w_external then
    r = ridge.load(dir .. "/ridge.bin", fvec.mmap_open(manifest.w_path))
  else
    r = ridge.load(dir .. "/ridge.bin")
  end
  local decider = manifest.has_decider and decide.load(dir .. "/decider.bin") or nil
  local gaz = manifest.has_gaz and require("santoku.learn.ner").load_gaz(dir .. "/gaz.bin") or nil
  local gaz_rms = manifest.has_gaz_rms and fvec.load(dir .. "/gaz_rms.bin") or nil
  local function encode (ext, out)
    local bl = {}
    for i = 1, #ext do
      local e = ext[i]
      bl[i] = { x = type(e) == "table" and e.x or e }
    end
    return encoder:encode({ blocks = bl }, out)
  end
  return { tokenizers = toks, encoder = encoder, ridge = r, decider = decider,
    gaz = gaz, gaz_rms = gaz_rms, encode = encode }
end

return M

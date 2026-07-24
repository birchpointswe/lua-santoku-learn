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
  local E = (type(opts.encoder) == "table" and opts.encoder.is_ensemble) and opts.encoder or nil
  if E then
    -- seed ensemble: re-build each base and embed it whole. The single-file external
    -- w/chol pairing has no per-model analogue, so w_path/chol_path are ignored here.
    for s = 0, E.K - 1 do
      local _, r, enc = E.build(s)
      enc:persist(dir .. "/encoder_" .. s .. ".bin")
      r:persist(dir .. "/ridge_" .. s .. ".bin")
      E.release(s)
    end
  else
    opts.encoder:persist(dir .. "/encoder.bin")
    opts.ridge:persist(dir .. "/ridge.bin")
  end
  if opts.decider then
    opts.decider:persist(dir .. "/decider.bin")
  end
  if opts.gaz then
    opts.gaz:persist(dir .. "/gaz.bin")
  end
  if opts.gaz_rms then
    opts.gaz_rms:persist(dir .. "/gaz_rms.bin")
  end
  local w_ext = not E and opts.w_path or nil
  local chol_ext = not E and opts.chol_path or nil
  local fh = assert(io.open(dir .. "/manifest.lua", "w"))
  fh:write(str.format(
    "return {\n  version = 2,\n  n_tokenizers = %d,\n  seed_ensemble = %s,\n  n_labels = %s,\n  has_decider = %s,\n  has_gaz = %s,\n  has_gaz_rms = %s,\n  w_external = %s,\n  w_path = %s,\n  chol_external = %s,\n  chol_path = %s,\n}\n",
    #toks,
    E and tostring(E.K) or "nil",
    E and E.n_labels and tostring(E.n_labels) or "nil",
    opts.decider and "true" or "false",
    opts.gaz and "true" or "false",
    opts.gaz_rms and "true" or "false",
    w_ext and "true" or "false",
    w_ext and str.format("%q", w_ext) or "nil",
    chol_ext and "true" or "false",
    chol_ext and str.format("%q", chol_ext) or "nil"))
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
  local function wrap (ext)
    local bl = {}
    for i = 1, #ext do
      local e = ext[i]
      bl[i] = { x = type(e) == "table" and e.x or e }
    end
    return bl
  end
  local encoder, r, encode
  if manifest.seed_ensemble then
    -- lazy build-load-discard mirror of optimize's deploy ensemble; consumed by
    -- util.predict_tiled via ridge.is_ensemble. encode is a passthrough (unused there).
    local E = { is_ensemble = true, K = manifest.seed_ensemble, n_labels = manifest.n_labels }
    E.build = function (s)
      local enc = spectral.load(dir .. "/encoder_" .. s .. ".bin")
      local rr = ridge.load(dir .. "/ridge_" .. s .. ".bin")
      return (function (ext, out) return enc:encode({ blocks = wrap(ext) }, out) end), rr, enc
    end
    E.release = function () collectgarbage("collect") end
    encoder, r = E, E
    encode = function (x) return x end
  else
    if manifest.chol_external then
      encoder = spectral.load(dir .. "/encoder.bin", fvec.mmap_open(manifest.chol_path))
    else
      encoder = spectral.load(dir .. "/encoder.bin")
    end
    if manifest.w_external then
      r = ridge.load(dir .. "/ridge.bin", fvec.mmap_open(manifest.w_path))
    else
      r = ridge.load(dir .. "/ridge.bin")
    end
    encode = function (ext, out)
      return encoder:encode({ blocks = wrap(ext) }, out)
    end
  end
  local decider = manifest.has_decider and decide.load(dir .. "/decider.bin") or nil
  local gaz = manifest.has_gaz and require("santoku.learn.ner").load_gaz(dir .. "/gaz.bin") or nil
  local gaz_rms = manifest.has_gaz_rms and fvec.load(dir .. "/gaz_rms.bin") or nil
  return { tokenizers = toks, encoder = encoder, ridge = r, decider = decider,
    gaz = gaz, gaz_rms = gaz_rms, encode = encode }
end

return M

local fs = require("santoku.fs")

local src, dst = arg[1], arg[2]

for _, lbl in ipairs({ "pos", "neg" }) do
  local outd = fs.join(dst, lbl)
  fs.mkdirp(outd)
  for _, split in ipairs({ "train", "test" }) do
    local ind = fs.join(src, split, lbl)
    local names = {}
    for fp in fs.files(ind) do
      names[#names + 1] = fp
    end
    for _, fp in ipairs(names) do
      fs.mv(fp, fs.join(outd, split .. "_" .. fs.basename(fp)))
    end
  end
end

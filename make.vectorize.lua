local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.flatten({ { "-O3", "-march=native", "-fno-lto", "-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize" }, base.env.cflags })
base.env.ldflags = arr.flatten({ { "-O3", "-march=native", "-fno-lto", "-Wl,--build-id" }, base.env.ldflags })
return base

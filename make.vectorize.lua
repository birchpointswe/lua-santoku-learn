local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.extend({ "-O3", "-march=native", "-fno-lto", "-fdata-sections", "-ffunction-sections", "-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize" }, base.env.cflags)
base.env.ldflags = arr.extend({ "-O3", "-march=native", "-fno-lto", "-Wl,--gc-sections", "-Wl,--build-id"}, base.env.ldflags)
return base

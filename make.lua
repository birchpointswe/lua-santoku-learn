local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.extend({ "-O3", "-march=native", "-fdata-sections", "-ffunction-sections", }, base.env.cflags)
base.env.ldflags = arr.extend({ "-O3", "-march=native", "-Wl,--gc-sections" }, base.env.ldflags)
return base

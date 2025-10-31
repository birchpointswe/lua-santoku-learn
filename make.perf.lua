local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.extend({ "-g3", "-O3", "-march=native", "-fdata-sections", "-ffunction-sections", "-fno-omit-frame-pointer", "-fno-inline" }, base.env.cflags)
base.env.ldflags = arr.extend({ "-g3", "-O3", "-march=native", "-Wl,--gc-sections" }, base.env.ldflags)
return base

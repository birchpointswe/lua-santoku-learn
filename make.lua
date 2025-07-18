local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.extend({ "-O3", "-march=native" }, base.env.cflags)
base.env.ldflags = arr.extend({ "-O3", "-march=native" }, base.env.ldflags)
return base

local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.flatten({ { "-g3", "-O3", "-march=native", "-fno-omit-frame-pointer" }, base.env.cflags })
base.env.ldflags = arr.flatten({ { "-g3", "-O3", "-march=native" }, base.env.ldflags })
return base

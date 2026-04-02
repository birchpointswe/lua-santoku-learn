local arr = require("santoku.array")
local fs = require("santoku.fs")
local base = fs.runfile("make.common.lua")
base.env.cflags = arr.flatten({ { "-O3" }, base.env.cflags })
base.env.ldflags = arr.flatten({ { "-O3" }, base.env.ldflags })
base.env.native = base.env.native or {}
base.env.native.cflags = arr.flatten({ { "-march=native" }, base.env.native.cflags or {} })
base.env.native.ldflags = arr.flatten({ { "-march=native" }, base.env.native.ldflags or {} })
return base

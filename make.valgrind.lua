local arr = require("santoku.array")
local fs = require("santoku.fs")
local sys = require("santoku.system")
local base = fs.runfile("make.common.lua")
base.env.blas_target = "GENERIC"
base.env.primme_cflags = "-O2 -fPIC"
base.env.primme_ldflags = "-O2 -fPIC"
base.env.cflags = arr.extend({ "-g3", "-O0", "-fno-inline", "-fno-omit-frame-pointer" }, base.env.cflags)
base.env.ldflags = arr.extend({ "-g3", "-O0", }, base.env.ldflags)
return base

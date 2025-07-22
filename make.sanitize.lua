local arr = require("santoku.array")
local fs = require("santoku.fs")
local err = require("santoku.error")
local env = require("santoku.env")
local str = require("santoku.string")
local sys = require("santoku.system")
local base = fs.runfile("make.common.lua")
local ld_preload = env.var("LD_PRELOAD", nil)
local asan = sys.sh({ "sh", "-c", str.format([[
  %s -fsanitize=address -xc /dev/null -### 2>&1 | grep -Pom1 '"\K[^"]*asan[^"]*.so(?=")'
]], env.var("CC", "clang")) })()
err.assert(asan, "Couldn't determine asan lib to preload")
if ld_preload then
  ld_preload = ld_preload .. ":" .. asan
else
  ld_preload = asan
end
local asan_options = env.var("ASAN_OPTIONS", "detect_stack_use_after_return=1:strict_string_checks=1")
local ubsan_options = env.var("UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")
base.env.test.env_vars = {
  ASAN_OPTIONS = asan_options,
  UBSAN_OPTIONS = ubsan_options,
  LD_PRELOAD = ld_preload
}
base.env.cflags = arr.extend({
  "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
  "-g3", "-O1", "-fno-omit-frame-pointer"
}, base.env.cflags)
base.env.ldflags = arr.extend({
  "-fsanitize=address,undefined",
  "-g3", "-O1"
}, base.env.ldflags)
return base

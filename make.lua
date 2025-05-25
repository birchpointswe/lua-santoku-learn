local env = {

  name = "santoku-tsetlin",
  version = "0.0.77-1",
  variable_prefix = "TK_TSETLIN",
  license = "MIT",
  public = true,

  cflags = {
    "-march=native", "-std=gnu11", "-O3", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
  },

  ldflags = {
    "-march=native", "-O3", "-lm", "-lpthread", "-lnuma"
  },

  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.266-1",
  },

  test = {
    sanitize = {
      cflags = { "-fsanitize=address,undefined" },
      ldflags = { "-fsanitize=address,undefined" },
    },
    dependencies = {
      "luacov >= 0.15.0-1",
      "santoku-matrix >= 0.0.43-1",
      "santoku-fs >= 0.0.34-1",
      "lua-cjson >= 2.1.0.10-1",
    }
  },

}

env.homepage = "https://github.com/treadwelllane/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return {
  type = "lib",
  env = env,
}

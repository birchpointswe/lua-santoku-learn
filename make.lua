local env = {

  name = "santoku-tsetlin",
  version = "0.0.77-1",
  variable_prefix = "TK_TSETLIN",
  license = "MIT",
  public = true,

  cflags = { "-march=native", "-fopenmp", "-std=gnu11", "-O3", "-Wall", "-Wextra", "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow", "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable" },
  ldflags = { "-march=native", "-fopenmp", "-O3", "-lm", "-lpthread", "-lnuma" },

  rules = {
    ["tsetlin/capi.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    },
    ["evaluator/capi.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    },
    ["threshold.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    },
    ["graph.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    },
    ["spectral.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    },
    ["corex.c$"] = {
      cflags = { "-I$(shell luarocks show santoku --rock-dir)/include/" }
    }
  },

  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.258-1",
  },

  test = {
    cflags = { "-g3" },
    ldflags = { "-g3" },
    dependencies = {
      "luacov >= 0.15.0-1",
      "santoku-matrix >= 0.0.36-1",
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

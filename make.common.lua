local env = {

  name = "santoku-tsetlin",
  version = "0.0.102-1",
  variable_prefix = "TK_TSETLIN",
  license = "MIT",
  public = true,

  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra", "-pthread",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    -- TODO: Not all libs likely need these includes
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-threads --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
  },

  ldflags = {
    "-lm", "-pthread"
  },

  rules = {
    ["spectral%.c"] = {
      cflags = { "$(shell pkg-config --cflags openblas) -I$(PWD)/deps/primme/primme/include" },
      ldflags = { "$(PWD)/deps/primme/primme/lib/libprimme.a -llapack -llapacke $(shell pkg-config --libs openblas)" },
    },
    ["itq%.c"] = {
      cflags = { "$(shell pkg-config --cflags openblas)" },
      ldflags = { "$(shell pkg-config --libs openblas) -llapacke" },
    },
  },

  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.280-1",
    "santoku-threads >= 0.0.12-1",
    "santoku-matrix >= 0.0.93-1",
    "santoku-system >= 0.0.56-1",
  },

  test = {

    -- TODO: Support injecting the poor mans signal tracer during test
    -- cinject = { "<santoku/execinfo.h>" },
    -- cflags = { "-g", "-O0", "-fno-omit-frame-pointer" },
    -- ldflags = { "-g", "-O0" },

    dependencies = {
      "luacov >= 0.15.0-1",
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

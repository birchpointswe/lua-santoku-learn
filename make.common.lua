local env = {
  name = "santoku-learn",
  version = "0.0.85-1",
  variable_prefix = "TK_LEARN",
  license = "MIT",
  public = true,
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-I$(shell luarocks show santoku-lpeg --rock-dir)/include/",
  },
  ldflags = {
    "-lm",
  },
  native = {
    cflags = {
      "-fopenmp", "$(MATHLIBS_CFLAGS)",
    },
    ldflags = {
      "-fopenmp", "$(MATHLIBS_LDFLAGS)",
    },
  },
  build = {
    wasm = {
      ldflags = {
        "-sWASM_BIGINT",
      },
    },
  },
  test = {
    wasm = {
      ldflags = {
        "-sWASM_BIGINT",
      },
    },
    dependencies = {
      "santoku-system >= 0.0.63-1",
    },
  },
  dependencies = {
    "lua == 5.1",
    "santoku >= 0.0.334-1",
    "santoku-matrix >= 0.0.336-1",
    "santoku-fs >= 0.0.45-1",
    "santoku-lpeg >= 0.0.7-1",
    "lua-cjson >= 2.1.0.10-1",
  },
}

env.homepage = "https://github.com/birchpointswe/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }

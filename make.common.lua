local rock = require("santoku.make.rock")

local env = {
  name = "santoku-learn",
  version = "3.1.0-1",
  variable_prefix = "TK_LEARN",
  license = "MIT",
  public = true,
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    rock.include("santoku"),
    rock.include("santoku-matrix"),
    rock.include("santoku-lpeg"),
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
      "santoku-system >= 2.0.0, < 3.0.0",
    },
  },
  dependencies = {
    "lua == 5.1",
    "santoku >= 2.0.0, < 3.0.0",
    "santoku-matrix >= 2.2.0, < 3.0.0",
    "santoku-fs >= 2.0.0, < 3.0.0",
    "santoku-lpeg >= 2.0.0, < 3.0.0",
  },
}

env.homepage = "https://github.com/birchpointswe/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }

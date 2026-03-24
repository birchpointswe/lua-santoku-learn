local env = {
  name = "santoku-learn",
  version = "0.0.30-1",
  variable_prefix = "TK_LEARN",
  license = "MIT",
  public = true,
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-fopenmp", "$(MATHLIBS_CFLAGS)"
  },
  ldflags = {
    "-lm", "-fopenmp", "$(MATHLIBS_LDFLAGS)"
  },
  dependencies = {
    "lua == 5.1",
    "santoku >= 0.0.322-1",
    "santoku-matrix >= 0.0.300-1",
    "santoku-fs >= 0.0.43-1",
    "lpeg >= 1.1.0-2",
    "lua-cjson >= 2.1.0.10-1",
  },
  test = {
    dependencies = {
      "santoku-system >= 0.0.62-1",
    }
  },
}

env.homepage = "https://github.com/treadwelllane/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }

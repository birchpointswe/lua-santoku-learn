local env = {
  name = "santoku-learn",
  version = "0.0.21-1",
  variable_prefix = "TK_LEARN",
  license = "MIT",
  public = true,
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-fopenmp", "$(shell pkg-config --cflags blas lapack lapacke)"
  },
  ldflags = {
    "-lm", "-fopenmp", "$(shell pkg-config --libs blas lapack lapacke)",
    "-Wl,-z,nodelete"
  },
  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.321-1",
    "santoku-matrix >= 0.0.298-1",
    "santoku-fs >= 0.0.41-1",
    "lpeg >= 1.1.0-2",
    "lua-cjson >= 2.1.0.10-1",
  },
  test = {
    dependencies = {
      "santoku-system >= 0.0.61-1",
    }
  },
}

env.homepage = "https://github.com/treadwelllane/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }

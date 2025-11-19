local env = {
  name = "santoku-tsetlin",
  version = "0.0.225-1",
  variable_prefix = "TK_TSETLIN",
  license = "MIT",
  public = true,
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wsign-compare", "-Wsign-conversion", "-Wstrict-overflow",
    "-Wpointer-sign", "-Wno-unused-parameter", "-Wno-unused-but-set-variable",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-fopenmp", "$(shell pkg-config --cflags blas lapack)"
  },
  ldflags = {
    "-lm", "-fopenmp", "$(shell pkg-config --cflags blas lapack)"
  },
  rules = {
    ["graph%.c"] = {
      cflags = {
        "-isystem$(PWD)/deps/primme/primme/include",
      },
      ldflags = {
        "$(PWD)/deps/primme/primme/lib/libprimme.a",
        "-fopenmp", "$(shell pkg-config --libs blas lapack)"
      },
    },
    ["spectral%.c"] = {
      cflags = {
        "-isystem$(PWD)/deps/primme/primme/include",
        "-fopenmp", "$(shell pkg-config --cflags blas lapack)"
      },
      ldflags = {
        "$(PWD)/deps/primme/primme/lib/libprimme.a",
        "-fopenmp", "$(shell pkg-config --libs blas lapack)"
      },
    },
    ["itq%.c"] = {
      cflags = {
        "-fopenmp", "$(shell pkg-config --cflags blas lapack lapacke)"
      },
      ldflags = {
        "-fopenmp", "$(shell pkg-config --libs blas lapack lapacke)"
      },
    },
    ["tch%.c"] = {
      cflags = { "-fopenmp" },
      ldflags = { "-fopenmp" },
    },
  },
  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.294-1",
    "santoku-matrix >= 0.0.175-1",
  },
  test = {
    dependencies = {
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

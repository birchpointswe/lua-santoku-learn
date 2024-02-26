local env = {

  name = "santoku-tsetlin",
  version = "0.0.1-1",
  variable_prefix = "TK_TSETLIN",
  public = true,

  dependencies = {
    "lua == 5.1",
    "santoku >= 0.0.196-1",
  },

  test = {
    dependencies = {
      "luacov == 0.15.0-1",
      "santoku-bitmap >= 0.0.2-1",
      "santoku-fs >= 0.0.29-1",
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

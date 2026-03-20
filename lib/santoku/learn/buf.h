#ifndef TK_BUF_H
#define TK_BUF_H

#include <lua.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>

#define TK_IVEC_BUF(var, pos, sz) \
  tk_ivec_t *var; int var##_idx; \
  if (lua_isuserdata(L, (pos))) { \
    var = tk_ivec_peek(L, (pos), #var); \
    tk_ivec_ensure(var, (uint64_t)(sz)); \
    var->n = (uint64_t)(sz); \
    lua_pushvalue(L, (pos)); \
  } else { \
    var = tk_ivec_create(L, (uint64_t)(sz)); \
  } \
  var##_idx = lua_gettop(L)

#define TK_DVEC_BUF(var, pos, sz) \
  tk_dvec_t *var; int var##_idx; \
  if (lua_isuserdata(L, (pos))) { \
    var = tk_dvec_peek(L, (pos), #var); \
    tk_dvec_ensure(var, (uint64_t)(sz)); \
    var->n = (uint64_t)(sz); \
    lua_pushvalue(L, (pos)); \
  } else { \
    var = tk_dvec_create(L, (uint64_t)(sz)); \
  } \
  var##_idx = lua_gettop(L)

#define TK_FVEC_BUF(var, pos, sz) \
  tk_fvec_t *var; int var##_idx; \
  if (lua_isuserdata(L, (pos))) { \
    var = tk_fvec_peek(L, (pos), #var); \
    tk_fvec_ensure(var, (uint64_t)(sz)); \
    var->n = (uint64_t)(sz); \
    lua_pushvalue(L, (pos)); \
  } else { \
    var = tk_fvec_create(L, (uint64_t)(sz)); \
  } \
  var##_idx = lua_gettop(L)

#endif

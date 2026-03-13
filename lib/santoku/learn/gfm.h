#ifndef TK_GFM_H
#define TK_GFM_H

#include <lua.h>
#include <lauxlib.h>
#include <stdint.h>
#include <stdbool.h>

#define TK_GFM_MT "tk_gfm_t"

typedef struct {
  int64_t nl;
  double *platt_a;
  double *platt_b;
  double alpha;
  bool destroyed;
} tk_gfm_t;

static inline tk_gfm_t *tk_gfm_peek (lua_State *L, int i) {
  return (tk_gfm_t *)luaL_checkudata(L, i, TK_GFM_MT);
}

#endif

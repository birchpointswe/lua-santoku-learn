#ifndef TK_DECIDE_H
#define TK_DECIDE_H

#include <lua.h>
#include <lauxlib.h>
#include <stdint.h>
#include <stdbool.h>

#define TK_DECIDE_MT "tk_decide_t"



typedef struct {
  int64_t nl;
  bool single;
  bool span;
  double threshold;
  double *offsets;
  double reject_offset;
  int64_t reject;
  bool destroyed;
} tk_decide_t;

static inline tk_decide_t *tk_decide_peek (lua_State *L, int i) {
  return (tk_decide_t *)luaL_checkudata(L, i, TK_DECIDE_MT);
}

#endif

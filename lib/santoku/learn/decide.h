#ifndef TK_DECIDE_H
#define TK_DECIDE_H

#include <lua.h>
#include <lauxlib.h>
#include <stdint.h>
#include <stdbool.h>

#define TK_DECIDE_MT "tk_decide_t"

// Three decode policies (mutually exclusive): multilabel (global threshold), single-label (per-label
// offsets + argmax), span (per-candidate argmax with a REJECT offset, resolved by nms_dp into spans).
typedef struct {
  int64_t nl;
  bool single;          // single-label per-label offsets + argmax
  bool span;            // span decode: nms_dp over candidates with a REJECT decision offset
  double threshold;     // multilabel cut
  double *offsets;      // single-label per-label offsets (length nl)
  double reject_offset; // span: amount subtracted from the REJECT score before argmax/nms_dp
  int64_t reject;       // span: REJECT class index
  bool destroyed;
} tk_decide_t;

static inline tk_decide_t *tk_decide_peek (lua_State *L, int i) {
  return (tk_decide_t *)luaL_checkudata(L, i, TK_DECIDE_MT);
}

#endif

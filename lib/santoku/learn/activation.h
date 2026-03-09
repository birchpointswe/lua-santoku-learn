#ifndef TK_ACTIVATION_H
#define TK_ACTIVATION_H

#include <math.h>
#include <string.h>
#include <stdint.h>

static inline uint8_t tk_activation_mode_from_str (const char *ms)
{
  if (!ms) return 0;
  if (strcmp(ms, "relu") == 0) return 1;
  if (strcmp(ms, "sigmoid") == 0) return 2;
  if (strcmp(ms, "tanh") == 0) return 3;
  if (strcmp(ms, "gelu") == 0) return 4;
  if (strcmp(ms, "softplus") == 0) return 5;
  if (strcmp(ms, "elu") == 0) return 6;
  if (strcmp(ms, "sin") == 0) return 7;
  if (strcmp(ms, "linear") == 0) return 8;
  if (strcmp(ms, "selu") == 0) return 9;
  if (strcmp(ms, "mish") == 0) return 10;
  if (strcmp(ms, "rff") == 0) return 11;
  if (strcmp(ms, "gaussian") == 0) return 12;
  if (strcmp(ms, "welsch") == 0) return 13;
  if (strcmp(ms, "sexp") == 0) return 14;
  if (strcmp(ms, "swish") == 0) return 15;
  if (strcmp(ms, "silu") == 0) return 15;
  return 0;
}

static inline double tk_activate (double v, uint8_t mode, int64_t idx)
{
  switch (mode) {
    case 1: return v > 0.0 ? v : 0.0;
    case 2: return 1.0 / (1.0 + exp(-v));
    case 3: return tanh(v);
    case 4: { double t = 0.7978845608 * (v + 0.044715 * v * v * v);
              return 0.5 * v * (1.0 + tanh(t)); }
    case 5: return log(1.0 + exp(v));
    case 6: return v > 0.0 ? v : exp(v) - 1.0;
    case 7: return sin(v);
    case 8: return v;
    case 9: return v > 0.0 ? 1.0507009873554805 * v
              : 1.0507009873554805 * 1.6732632423543773 * (exp(v) - 1.0);
    case 10: return v * tanh(log(1.0 + exp(v)));
    case 11: return (idx % 2 == 0) ? sin(v) : cos(v);
    case 12: return exp(-v * v);
    case 13: return 1.0 - exp(-v * v * 0.5);
    case 14: return fmax(v, 0.0) + log(1.0 + exp(-fabs(v)));
    case 15: return v / (1.0 + exp(-v));
    default: return v;
  }
}

static inline void tk_activate_vec (double *data, int64_t n, uint8_t mode)
{
  if (mode == 0) return;
  for (int64_t i = 0; i < n; i++)
    data[i] = tk_activate(data[i], mode, i);
}

#endif

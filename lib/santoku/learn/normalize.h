#ifndef TK_TEXT_NORMALIZE_H
#define TK_TEXT_NORMALIZE_H

#include <stdint.h>
#include <stddef.h>

static const char tk_text_c3[64] = {
  'a','a','a','a','a','a','a','c',
  'e','e','e','e','i','i','i','i',
  'd','n','o','o','o','o','o', 0,
  'o','u','u','u','u','y', 0, 's',
  'a','a','a','a','a','a','a','c',
  'e','e','e','e','i','i','i','i',
  'd','n','o','o','o','o','o', 0,
  'o','u','u','u','u','y', 0, 'y'
};

static const char tk_text_c4[64] = {
  'a','a','a','a','a','a','c','c',
  'c','c','c','c','c','c','d','d',
  'd','d','e','e','e','e','e','e',
  'e','e','e','e','g','g','g','g',
  'g','g','g','g','h','h','h','h',
  'i','i','i','i','i','i','i','i',
  'i','i', 0,  0, 'j','j','k','k',
  'k','l','l','l','l','l','l','l'
};

static const char tk_text_c5[64] = {
  'l','l','l','n','n','n','n','n',
  'n','n','n','n','o','o','o','o',
  'o','o','o','o','r','r','r','r',
  'r','r','s','s','s','s','s','s',
  's','s','t','t','t','t','t','t',
  'u','u','u','u','u','u','u','u',
  'u','u','u','u','w','w','y','y',
  'y','z','z','z','z','z','z','s'
};

typedef struct {
  uint8_t bytes[4];
  int n_out;
  int n_in;
} tk_norm_result_t;

static inline tk_norm_result_t tk_text_normalize_next (const char *in, size_t pos, size_t len) {
  tk_norm_result_t r;
  r.n_out = 0;
  r.n_in = 1;
  uint8_t c = (uint8_t)in[pos];
  if (c < 0x80) {
    r.bytes[0] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    r.n_out = 1;
  } else if ((c == 0xC3 || c == 0xC4 || c == 0xC5) &&
             pos + 1 < len && ((uint8_t)in[pos + 1] & 0xC0) == 0x80) {
    uint8_t c2 = (uint8_t)in[pos + 1];
    const char *tbl = (c == 0xC3) ? tk_text_c3 : (c == 0xC4) ? tk_text_c4 : tk_text_c5;
    char base = tbl[c2 - 0x80];
    if (base) {
      r.bytes[0] = (uint8_t)base;
      r.n_out = 1;
    } else {
      r.bytes[0] = c;
      r.bytes[1] = c2;
      r.n_out = 2;
    }
    r.n_in = 2;
  } else if (c == 0xC2 && pos + 1 < len && (uint8_t)in[pos + 1] == 0xA0) {
    r.bytes[0] = ' ';
    r.n_out = 1;
    r.n_in = 2;
  } else if (c == 0xCC && pos + 1 < len && ((uint8_t)in[pos + 1] & 0xC0) == 0x80) {
    r.n_in = 2;
  } else if (c == 0xCD && pos + 1 < len &&
             (uint8_t)in[pos + 1] >= 0x80 && (uint8_t)in[pos + 1] <= 0xAF) {
    r.n_in = 2;
  } else if ((c & 0xE0) == 0xC0 && pos + 1 < len) {
    r.bytes[0] = c;
    r.bytes[1] = (uint8_t)in[pos + 1];
    r.n_out = 2;
    r.n_in = 2;
  } else if ((c & 0xF0) == 0xE0 && pos + 2 < len) {
    for (int k = 0; k < 3; k++)
      r.bytes[k] = (uint8_t)in[pos + (size_t)k];
    r.n_out = 3;
    r.n_in = 3;
  } else if ((c & 0xF8) == 0xF0 && pos + 3 < len) {
    for (int k = 0; k < 4; k++)
      r.bytes[k] = (uint8_t)in[pos + (size_t)k];
    r.n_out = 4;
    r.n_in = 4;
  } else {
    r.bytes[0] = c;
    r.n_out = 1;
  }
  return r;
}

#endif

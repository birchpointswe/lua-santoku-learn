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

static inline int64_t tk_text_normalize (const char *in, size_t len, uint8_t *out, int64_t *pm) {
  int64_t j = 0;
  for (size_t i = 0; i < len; ) {
    uint8_t c = (uint8_t)in[i];
    if (c < 0x80) {
      if (pm) pm[j] = (int64_t)i;
      out[j++] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
      i++;
    } else if ((c == 0xC3 || c == 0xC4 || c == 0xC5) &&
               i + 1 < len && ((uint8_t)in[i + 1] & 0xC0) == 0x80) {
      uint8_t c2 = (uint8_t)in[i + 1];
      const char *tbl = (c == 0xC3) ? tk_text_c3 : (c == 0xC4) ? tk_text_c4 : tk_text_c5;
      char base = tbl[c2 - 0x80];
      if (base) {
        if (pm) pm[j] = (int64_t)i;
        out[j++] = (uint8_t)base;
      } else {
        if (pm) pm[j] = (int64_t)i;
        out[j++] = c;
        if (pm) pm[j] = (int64_t)(i + 1);
        out[j++] = c2;
      }
      i += 2;
    } else if (c == 0xC2 && i + 1 < len && (uint8_t)in[i + 1] == 0xA0) {
      if (pm) pm[j] = (int64_t)i;
      out[j++] = ' ';
      i += 2;
    } else if (c == 0xCC && i + 1 < len && ((uint8_t)in[i + 1] & 0xC0) == 0x80) {
      i += 2;
    } else if (c == 0xCD && i + 1 < len &&
               (uint8_t)in[i + 1] >= 0x80 && (uint8_t)in[i + 1] <= 0xAF) {
      i += 2;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
      if (pm) pm[j] = (int64_t)i;
      out[j++] = c;
      if (pm) pm[j] = (int64_t)(i + 1);
      out[j++] = (uint8_t)in[i + 1];
      i += 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
      for (int k = 0; k < 3; k++) {
        if (pm) pm[j] = (int64_t)(i + (size_t)k);
        out[j++] = (uint8_t)in[i + (size_t)k];
      }
      i += 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
      for (int k = 0; k < 4; k++) {
        if (pm) pm[j] = (int64_t)(i + (size_t)k);
        out[j++] = (uint8_t)in[i + (size_t)k];
      }
      i += 4;
    } else {
      if (pm) pm[j] = (int64_t)i;
      out[j++] = c;
      i++;
    }
  }
  if (pm) pm[j] = (int64_t)len;
  return j;
}

#endif

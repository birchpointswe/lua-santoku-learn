#ifndef TK_TEXT_NORMALIZE_H
#define TK_TEXT_NORMALIZE_H

#include <stdint.h>
#include <stddef.h>

// C3: U+00C0-00FF Latin-1 Supplement
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

// C4: U+0100-013F Latin Extended-A
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

// C5: U+0140-017F Latin Extended-A continued
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

// C6: U+0180-01BF Latin Extended-B
static const char tk_text_c6[64] = {
  'b', 0,  'b','b', 0,  0,  0,  'c',
   0,  'd','d', 0,  0,  0,  0,  0,
  'e', 'f','f', 0,  'g', 0,  0,  0,
  'i','i','k', 0,  0,  0,  0,  'n',
  'o','o', 0,  0,  'p', 0,  0,  0,
   0,  0,  0,  0,  't','t','t','u',
  'u', 0,  'y','y', 'z','z', 0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// C7: U+01C0-01FF Latin Extended-B continued
static const char tk_text_c7[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0, 'a','a','i',
  'i','o','o','u','u','u','u','u',
  'u','u','u','u','u', 0, 'a','a',
  'a','a','g','g','g','g','k','k',
  'o','o','o','o', 0,  0, 'j', 0,
  'g', 0,  0,  'n','n','a','a','a',
  'a','o','o','a','a', 0,  0,  0
};

// C8: U+0200-023F Latin Extended-B continued
static const char tk_text_c8[64] = {
  'a','a','a','a','e','e','e','e',
  'i','i','i','i','o','o','o','o',
  'r','r','r','r','u','u','u','u',
  's','s','t','t', 0,  0,  'h', 0,
   0, 'z', 0,  0,  0,  0,  'a','a',
   0,  0,  0,  0, 'o','o', 0,  0,
   0,  0,  0, 'y','y', 0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// C9: U+0240-027F IPA Extensions
static const char tk_text_c9[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0, 'b', 0, 'c','d','d',
   0,  0, 'e', 0,  0,  0,  0,  0,
  'g', 0,  0, 'h','h', 0, 'i','i',
   0,  0,  0,  0, 'l','l', 0,  0,
  'm', 0, 'n', 0, 'o', 0,  0,  0,
   0, 'r','r','r', 0,  0,  0,  0
};

// CE: U+0380-03BF Greek and Coptic
// Multi-char Θ(0x98/0xB8) and Ξ(0x9E/0xBE) → 0 in table, handled separately
static const char tk_text_ce[64] = {
   0,  0,  0,  0,  0,  0, 'a', 0,
  'e','e','i', 0, 'o', 0, 'y','o',
  'i','a','b','g','d','e','z','e',
   0, 'i','k','l','m','n', 0, 'o',
  'p', 0,  0, 's','t','y','f','x',
   0, 'o', 0,  0,  0,  0,  0,  0,
  'a','b','g','d','e','z','e', 0,
  'i','k','l','m','n', 0, 'o', 0
};

// CF: U+03C0-03FF Greek continued
// Multi-char Ψ(0x88) → 0 in table, handled separately
static const char tk_text_cf[64] = {
  'p','r','s','s','t','y','f','x',
   0, 'o', 0,  0,  0,  0,  0,  0,
  'b', 0, 'f', 0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// D0: U+0400-043F Cyrillic
// Multi-char: Ж(0x96) Ц(0xA6) Ч(0xA7) Ш(0xA8) Щ(0xA9) Ъ(0xAA) Ь(0xAC) Ю(0xAE) Я(0xAF) → 0
static const char tk_text_d0[64] = {
  'e', 0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
  'a','b','v','g','d','e', 0, 'z',
  'i','y','k','l','m','n','o','p',
  'r','s','t','u','f','h', 0,  0,
   0,  0,  0, 'y', 0, 'e', 0,  0,
  'a','b','v','g','d','e', 0, 'z',
  'i','y','k','l','m','n','o','p'
};

// D1: U+0440-047F Cyrillic continued
// Multi-char: ц(0x86) ч(0x87) ш(0x88) щ(0x89) ъ(0x8A) ь(0x8C) ю(0x8E) я(0x8F) → 0
static const char tk_text_d1[64] = {
  'r','s','t','u','f','h', 0,  0,
   0,  0,  0, 'y', 0, 'e', 0,  0,
  'e', 0, 'g', 0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0, 'y','y', 0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// D2: U+0480-04BF Cyrillic Extended
static const char tk_text_d2[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0, 'g', 0, 'g', 0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0, 'k', 0, 'k', 0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0, 'n', 0, 'n', 0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// D3: U+04C0-04FF Cyrillic Extended continued
static const char tk_text_d3[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
  'a', 'a', 0,  0,  0,  0,  0,  0,
  'a','a', 0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
  'o','o', 0,  0,  0,  0,'u','u',
   0,  0,  0,  0,  0,  0,  0,  0,
  'h','h', 0,  0,  0,  0,  0,  0
};

// D4: U+0500-053F Cyrillic Supplement + Armenian capitals start
// U+0531-053F = Armenian Ա-Կ at indices 0xB1-0xBF
static const char tk_text_d4[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0, 'a','b','g','d','e','z','e',
  'y','t','z','i','l','x','t','k'
};

// D5: U+0540-057F Armenian continued
// U+0540-0556 = Armenian capitals Հ-Ֆ at indices 0x80-0x96
// U+0561-057F = Armenian lowercase ա-ֆ at indices 0xA1-0xBF (partial)
static const char tk_text_d5[64] = {
  'h','d','g','c','m','y','n','s',
  'o','c','p','j','r','s','v','t',
  'r','c','w','p','k','o','f', 0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0, 'a','b','g','d','e','z','e',
  'y','t','z','i','l','x','t','k',
  'h','d','g','c','m','y','n','s',
  'o','c','p','j','r','s','v','t'
};

// E1 B8: U+1E00-1E3F Latin Extended Additional
static const char tk_text_e1_b8[64] = {
  'a','a','b','b','b','b','b','b',
  'c','c','d','d','d','d','d','d',
  'd','d','d','d','e','e','e','e',
  'e','e','e','e','e','e','f','f',
  'g','g','h','h','h','h','h','h',
  'h','h','h','h','i','i','i','i',
  'k','k','k','k','l','l','l','l',
  'l','l','l','l','m','m','m','m'
};

// E1 B9: U+1E40-1E7F Latin Extended Additional continued
static const char tk_text_e1_b9[64] = {
  'm','m','n','n','n','n','n','n',
  'o','o','o','o','o','o','o','o',
  'p','p','p','p','r','r','r','r',
  'r','r','r','r','s','s','s','s',
  's','s','s','s','s','s','t','t',
  't','t','t','t','t','t','u','u',
  'u','u','u','u','u','u','u','u',
  'v','v','v','v','w','w','w','w'
};

// E1 BA: U+1E80-1EBF Latin Ext Additional + Vietnamese
static const char tk_text_e1_ba[64] = {
  'w','w','w','w','w','w','x','x',
  'x','x','y','y','y','y','z','z',
  'z','z','z','z','h','t','w','y',
  'a', 0,  0,  0,  0,  0,  0,  0,
  'a','a','a','a','a','a','a','a',
  'a','a','a','a','a','a','e','e',
  'e','e','e','e','e','e','e','e',
  'i','i','o','o','o','o','o','o'
};

// E1 BB: U+1EC0-1EFF Vietnamese continued
static const char tk_text_e1_bb[64] = {
  'o','o','o','o','o','o','o','o',
  'o','o','o','o','u','u','u','u',
  'u','u','u','u','u','u','u','u',
  'u','u','y','y','y','y','y','y',
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0
};

// E2 80: U+2000-203F General Punctuation
static const char tk_text_e2_80[64] = {
  ' ',' ',' ',' ',' ',' ',' ',' ',
  ' ',' ',' ', 0,  0,  0,  0,  0,
  '-','-','-','-','-','-', 0,  0,
  '\'','\'','\'','\'','"','"','"','"',
   0,  0,  0,  0, '.', 0, '.', 0,
   0,  0,  0,  0,  ' ',' ', 0,  0,
   0,  0,'\'','\'','\'','\'','\'','\'',
   0, '\'','\'', 0,  0,  0,  0,  0
};

typedef struct {
  uint8_t bytes[4];
  int n_out;
  int n_in;
} tk_norm_result_t;

static inline tk_norm_result_t tk_text_normalize_next (const char *in, size_t pos, size_t len) {
  tk_norm_result_t r = {0};
  r.n_out = 0;
  r.n_in = 1;
  uint8_t c = (uint8_t)in[pos];

  if (c < 0x80) {

    r.bytes[0] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    r.n_out = 1;

  } else if ((c & 0xE0) == 0xC0 && pos + 1 < len && ((uint8_t)in[pos + 1] & 0xC0) == 0x80) {

    uint8_t c2 = (uint8_t)in[pos + 1];
    r.n_in = 2;

    if (c == 0xC2) {

      switch (c2) {
        case 0xA0: r.bytes[0] = ' '; r.n_out = 1; break;
        case 0xAA: r.bytes[0] = 'a'; r.n_out = 1; break;
        case 0xAD: r.n_out = 0; break;
        case 0xB2: r.bytes[0] = '2'; r.n_out = 1; break;
        case 0xB3: r.bytes[0] = '3'; r.n_out = 1; break;
        case 0xB5: r.bytes[0] = 'u'; r.n_out = 1; break;
        case 0xB9: r.bytes[0] = '1'; r.n_out = 1; break;
        case 0xBA: r.bytes[0] = 'o'; r.n_out = 1; break;
        default: r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; break;
      }

    } else if (c == 0xC3) {

      if (c2 == 0x86) { r.bytes[0] = 'a'; r.bytes[1] = 'e'; r.n_out = 2; }
      else if (c2 == 0xA6) { r.bytes[0] = 'a'; r.bytes[1] = 'e'; r.n_out = 2; }
      else {
        char base = tk_text_c3[c2 - 0x80];
        if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
        else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
      }

    } else if (c == 0xC4) {

      if (c2 == 0xB2) { r.bytes[0] = 'i'; r.bytes[1] = 'j'; r.n_out = 2; }
      else if (c2 == 0xB3) { r.bytes[0] = 'i'; r.bytes[1] = 'j'; r.n_out = 2; }
      else {
        char base = tk_text_c4[c2 - 0x80];
        if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
        else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
      }

    } else if (c == 0xC5) {

      if (c2 == 0x92) { r.bytes[0] = 'o'; r.bytes[1] = 'e'; r.n_out = 2; }
      else if (c2 == 0x93) { r.bytes[0] = 'o'; r.bytes[1] = 'e'; r.n_out = 2; }
      else {
        char base = tk_text_c5[c2 - 0x80];
        if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
        else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
      }

    } else if (c == 0xC6) {

      char base = tk_text_c6[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xC7) {

      char base = tk_text_c7[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xC8) {

      char base = tk_text_c8[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xC9) {

      char base = tk_text_c9[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xCC) {

      r.n_out = 0;

    } else if (c == 0xCD && c2 >= 0x80 && c2 <= 0xAF) {

      r.n_out = 0;

    } else if (c == 0xCE) {

      if (c2 == 0x98 || c2 == 0xB8) { r.bytes[0] = 't'; r.bytes[1] = 'h'; r.n_out = 2; }
      else if (c2 == 0x9E || c2 == 0xBE) { r.bytes[0] = 'k'; r.bytes[1] = 's'; r.n_out = 2; }
      else {
        char base = tk_text_ce[c2 - 0x80];
        if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
        else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
      }

    } else if (c == 0xCF) {

      if (c2 == 0x88) { r.bytes[0] = 'p'; r.bytes[1] = 's'; r.n_out = 2; }
      else {
        char base = tk_text_cf[c2 - 0x80];
        if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
        else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
      }

    } else if (c == 0xD0) {

      switch (c2) {
        case 0x96: r.bytes[0] = 'z'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0xA6: r.bytes[0] = 't'; r.bytes[1] = 's'; r.n_out = 2; break;
        case 0xA7: r.bytes[0] = 'c'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0xA8: r.bytes[0] = 's'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0xA9: r.bytes[0] = 's'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0xAA: r.n_out = 0; break;
        case 0xAC: r.n_out = 0; break;
        case 0xAE: r.bytes[0] = 'y'; r.bytes[1] = 'u'; r.n_out = 2; break;
        case 0xAF: r.bytes[0] = 'y'; r.bytes[1] = 'a'; r.n_out = 2; break;
        case 0xB6: r.bytes[0] = 'z'; r.bytes[1] = 'h'; r.n_out = 2; break;
        default: {
          char base = tk_text_d0[c2 - 0x80];
          if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
          else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
          break;
        }
      }

    } else if (c == 0xD1) {

      switch (c2) {
        case 0x86: r.bytes[0] = 't'; r.bytes[1] = 's'; r.n_out = 2; break;
        case 0x87: r.bytes[0] = 'c'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0x88: r.bytes[0] = 's'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0x89: r.bytes[0] = 's'; r.bytes[1] = 'h'; r.n_out = 2; break;
        case 0x8A: r.n_out = 0; break;
        case 0x8C: r.n_out = 0; break;
        case 0x8E: r.bytes[0] = 'y'; r.bytes[1] = 'u'; r.n_out = 2; break;
        case 0x8F: r.bytes[0] = 'y'; r.bytes[1] = 'a'; r.n_out = 2; break;
        default: {
          char base = tk_text_d1[c2 - 0x80];
          if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
          else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }
          break;
        }
      }

    } else if (c == 0xD2) {

      char base = tk_text_d2[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xD3) {

      char base = tk_text_d3[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xD4) {

      char base = tk_text_d4[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else if (c == 0xD5) {

      char base = tk_text_d5[c2 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.n_out = 2; }

    } else {

      r.bytes[0] = c;
      r.bytes[1] = c2;
      r.n_out = 2;

    }

  } else if ((c & 0xF0) == 0xE0 && pos + 2 < len) {

    uint8_t c2 = (uint8_t)in[pos + 1];
    uint8_t c3 = (uint8_t)in[pos + 2];
    r.n_in = 3;

    if (c == 0xE1 && c2 == 0xB8 && (c3 & 0xC0) == 0x80) {

      char base = tk_text_e1_b8[c3 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3; r.n_out = 3; }

    } else if (c == 0xE1 && c2 == 0xB9 && (c3 & 0xC0) == 0x80) {

      char base = tk_text_e1_b9[c3 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3; r.n_out = 3; }

    } else if (c == 0xE1 && c2 == 0xBA && (c3 & 0xC0) == 0x80) {

      char base = tk_text_e1_ba[c3 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3; r.n_out = 3; }

    } else if (c == 0xE1 && c2 == 0xBB && (c3 & 0xC0) == 0x80) {

      char base = tk_text_e1_bb[c3 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3; r.n_out = 3; }

    } else if (c == 0xE1 && (c2 == 0xAA || c2 == 0xAB)) {

      r.n_out = 0;

    } else if (c == 0xE1 && c2 == 0xB7 && c3 >= 0x80 && c3 <= 0xBF) {

      r.n_out = 0;

    } else if (c == 0xE2 && c2 == 0x80 && (c3 & 0xC0) == 0x80) {

      char base = tk_text_e2_80[c3 - 0x80];
      if (base) { r.bytes[0] = (uint8_t)base; r.n_out = 1; }
      else { r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3; r.n_out = 3; }

    } else if (c == 0xEF && (c2 == 0xBC || c2 == 0xBD)) {

      uint16_t cp = ((uint16_t)(c & 0x0F) << 12) | ((uint16_t)(c2 & 0x3F) << 6) | (c3 & 0x3F);
      if (cp >= 0xFF01 && cp <= 0xFF5E) {
        uint8_t ascii = (uint8_t)(cp - 0xFEE0);
        r.bytes[0] = (ascii >= 'A' && ascii <= 'Z') ? ascii + 32 : ascii;
        r.n_out = 1;
      } else {
        r.bytes[0] = c; r.bytes[1] = c2; r.bytes[2] = c3;
        r.n_out = 3;
      }

    } else {

      for (int k = 0; k < 3; k++)
        r.bytes[k] = (uint8_t)in[pos + (size_t)k];
      r.n_out = 3;

    }

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

#include "siphash.h"
namespace cuckoogpu {

void setkeys(siphash_keys *keys, const char *keybuf) {
  keys->k0 = htole64(((u64 *)keybuf)[0]);
  keys->k1 = htole64(((u64 *)keybuf)[1]);
  keys->k2 = htole64(((u64 *)keybuf)[2]);
  keys->k3 = htole64(((u64 *)keybuf)[3]);
}

u64 siphash24(const siphash_keys *keys, const u64 nonce) {
  u64 v0 = keys->k0, v1 = keys->k1, v2 = keys->k2, v3 = keys->k3 ^ nonce;
  SIPROUND; SIPROUND;
  SIPROUND; SIPROUND;
  v0 ^= nonce;
  v2 ^= 0xff;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  SIPROUND; SIPROUND; SIPROUND; SIPROUND;
  return (v0 ^ v1) ^ (v2  ^ v3);
}

};

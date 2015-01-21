// Copyright (c) 2013 Pieter Wuille
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef _SECP256K1_ECMULT_
#define _SECP256K1_ECMULT_

#include "num.h"
#include "group.h"

static void secp256k1_ecmult_start(void);
static void secp256k1_ecmult_stop(void);

/** Multiply with the generator: R = a*G */
static void secp256k1_ecmult_gen(secp256k1_gej_t *r, const secp256k1_num_t *a);
/** Double multiply: R = na*A + ng*G */
static void secp256k1_ecmult(secp256k1_gej_t *r, const secp256k1_gej_t *a, const secp256k1_num_t *na, const secp256k1_num_t *ng);

#endif

// -----------------------------------------------------------------------------
// @file g_random.c
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_random.h"

// Variant for 32-bit microcontrollers of the Xoshiro256+ algorithm
//
// Source: https://prng.di.unimi.it/

static uint32_t _state[8] = {
    0xBAD5EED1, 0x062081DE, 0xEAD3D6C8, 0x7F4A7C15, 0x3D627E37, 0xA5A5A5A5, 0x12345678, 0x87654321};

static inline uint32_t _rotate_left(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

void g_random_seed(uint32_t seed) {
    for (int i = 0; i < 8; i++) {
        seed ^= seed >> 13;
        seed ^= seed << 17;
        seed ^= seed >> 5;
        _state[i] = seed * (0x2545F491 + 1) + i;
    }

    // Discard the first few values to improve randomness
    for (int i = 0; i < 16; i++) {
        (void)g_random_next();
    }
}

uint32_t g_random_next(void) {
    const uint32_t r = _rotate_left(_state[1] * 5, 7) * 9;
    const uint32_t t = _state[1] << 9;

    _state[2] ^= _state[0];
    _state[5] += _state[1]; // use ADD instead of XOR
    _state[1] ^= _state[2];
    _state[7] ^= _state[3];
    _state[3] += _state[4]; // use ADD instead of XOR
    _state[4] ^= _state[5];
    _state[0] ^= _state[6];
    _state[6] ^= _state[7];

    _state[6] ^= t;
    _state[2] = _rotate_left(_state[2], 11);

    return r;
}

float g_random_range(float min, float max) {
    if (min >= max)
        return min;

    float random = (float)g_random_next() / (float)0xFFFFFFFF;
    return min + (random * (max - min));
}

// -----------------------------------------------------------------------------
// End of File

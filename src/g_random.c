// -----------------------------------------------------------------------------
// @file g_random.c
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_random.h"

// Xoshiro256+ random number generator
//
// Source: https://prng.di.unimi.it/

static uint32_t xoshiro_state[8] = {
    0xBAD5EED1, 0x062081DE, 0xEAD3D6C8, 0x7F4A7C15, 0x3D627E37, 0xA5A5A5A5, 0x12345678, 0x87654321};

static inline uint32_t rotate_left(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint32_t g_random_uint32(void) {
    const uint32_t result = rotate_left(xoshiro_state[1] * 5, 7) * 9;
    const uint32_t t      = xoshiro_state[1] << 9;

    xoshiro_state[2] ^= xoshiro_state[0];
    xoshiro_state[5] ^= xoshiro_state[1];
    xoshiro_state[1] ^= xoshiro_state[2];
    xoshiro_state[7] ^= xoshiro_state[3];
    xoshiro_state[3] ^= xoshiro_state[4];
    xoshiro_state[4] ^= xoshiro_state[5];
    xoshiro_state[0] ^= xoshiro_state[6];
    xoshiro_state[6] ^= xoshiro_state[7];

    xoshiro_state[6] ^= t;
    xoshiro_state[2] = rotate_left(xoshiro_state[2], 11);

    return result;
}

void g_random_seed(uint32_t seed) {
    for (int i = 0; i < 8; i++) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        xoshiro_state[i] = seed * 0x2545F491 + i;
    }

    // Discard the first 16 values
    for (int i = 0; i < 16; i++) {
        (void)g_random_uint32();
    }
}

float g_random_range(float min, float max) {
    if (min >= max)
        return min;

    float random = (float)g_random_uint32() / (float)0xFFFFFFFF;
    return min + (random * (max - min));
}

// -----------------------------------------------------------------------------
// End Of File

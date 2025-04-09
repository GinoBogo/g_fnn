// -----------------------------------------------------------------------------
// @file g_random.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef G_RANDOM_H
#define G_RANDOM_H

#include <stdint.h> // uint32_t

void g_random_seed(uint32_t seed);

uint32_t g_random_next(void);

float g_random_range(float min, float max);

#endif // G_RANDOM_H

// -----------------------------------------------------------------------------
// End Of File

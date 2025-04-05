#include <math.h>   // sqrtf, logf
#include <stdlib.h> // RAND_MAX, rand

// Box-Muller transform for normal distribution
static float rand_normal(void) {
    static float n2        = 0.0;
    static int   n2_cached = 0;

    if (!n2_cached) {
        float x, y, r, n1;
        do {
            x = 2.0f * rand() / (float)RAND_MAX - 1.0f;
            y = 2.0f * rand() / (float)RAND_MAX - 1.0f;
            r = x * x + y * y;
        } while (r == 0.0f || r >= 1.0f);

        float d = sqrtf(-2.0f * logf(r) / r);

        n1 = x * d;
        n2 = y * d;

        n2_cached = 1;
        return n1;
    } else {
        n2_cached = 0;
        return n2;
    }
}

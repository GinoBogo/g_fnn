// -----------------------------------------------------------------------------
// @file g_network.h
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef G_NETWORK_H
#define G_NETWORK_H

#include "g_layer.h"

// -----------------------------------------------------------------------------

typedef struct g_network_t {
    // variables
    g_pages_t *pages;
    g_layers_t layers;

    // functions
    bool (*Create)(struct g_network_t *self, g_pages_t *pages);
    void (*Destroy)(struct g_network_t *self);
    void (*Init_Weights)(struct g_network_t *self, float bias);
    void (*Step_Forward)(struct g_network_t *self);
    void (*Step_Errors)(struct g_network_t *self, f_vector_t *actual_outputs);
    void (*Step_Backward)(struct g_network_t *self);

    // intrinsic
    bool _is_safe;
} g_network_t;

typedef struct g_networks_t {
    g_network_t *ptr;
    int          len;
} g_networks_t;

// -----------------------------------------------------------------------------

extern void g_network_link(g_network_t *self);

extern bool g_network_pages_check(g_pages_t *pages);

#endif // G_NETWORK_H

// -----------------------------------------------------------------------------
// End Of File

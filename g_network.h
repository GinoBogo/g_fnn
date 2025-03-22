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
    g_layers_data_t *data;
    g_layers_t       layers;

    // functions
    bool (*Create)(struct g_network_t *self, g_layers_data_t *data);
    void (*Destroy)(struct g_network_t *self);
    void (*Weights_Init)(struct g_network_t *self);
    void (*Step_Fwd)(struct g_network_t *self);

    // intrinsic
    bool _is_safe;
} g_network_t;

typedef struct g_networks_t {
    g_network_t *ptr;
    int          len;
} g_networks_t;

// -----------------------------------------------------------------------------

extern void g_network_link(g_network_t *self);

extern bool g_network_data_check(g_layers_data_t *data);

#endif // G_NETWORK_H

// -----------------------------------------------------------------------------
// End Of File

// -----------------------------------------------------------------------------
// @file g_layer.h
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef G_LAYER_H
#define G_LAYER_H

#include "g_neuron.h"

// -----------------------------------------------------------------------------

typedef struct g_layer_t {
    // variables
    g_layer_data_t *data;
    int             l_id; // layer index
    g_neurons_t     neurons;

    // functions
    bool (*Create)(struct g_layer_t *self, g_layer_data_t *data, int l_id);
    void (*Destroy)(struct g_layer_t *self);
    void (*Init_Weights)(struct g_layer_t *self);
    void (*Step_Forward)(struct g_layer_t *self);
    void (*Step_Errors)(struct g_layer_t *self, struct g_layer_t *next);
    void (*Step_Backward)(struct g_layer_t *self);

    // intrinsic
    bool _is_safe;
} g_layer_t;

typedef struct g_layers_t {
    g_layer_t *ptr;
    int        len;
} g_layers_t;

// -----------------------------------------------------------------------------

extern void g_layer_link(g_layer_t *self);

extern void g_layer_data_reset(g_layer_data_t *data);

extern bool g_layer_data_check(g_layer_data_t *data, int l_id);

#endif // G_LAYER_H

// -----------------------------------------------------------------------------
// End Of File

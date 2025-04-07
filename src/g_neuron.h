// -----------------------------------------------------------------------------
// @file g_neuron.h
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef G_NEURON_H
#define G_NEURON_H

#include <stdbool.h> // bool

#include "g_page.h" // g_page_t

// -----------------------------------------------------------------------------

typedef struct g_neuron_t {
    // variables
    int       n_id; // neuron index
    g_page_t *page;

    // functions
    bool (*Create)(struct g_neuron_t *self, g_page_t *page, int n_id);
    void (*Destroy)(struct g_neuron_t *self);
    void (*Step_Forward_Z)(struct g_neuron_t *self);
    void (*Step_Forward_Y)(struct g_neuron_t *self);

    // intrinsic
    bool _is_safe;
} g_neuron_t;

typedef struct g_neurons_t {
    g_neuron_t *ptr;
    int         len;
} g_neurons_t;

// -----------------------------------------------------------------------------

extern void g_neuron_link(g_neuron_t *self);

extern bool g_neuron_page_check(g_page_t *page, int n_id);

#endif // G_NEURON_H

// -----------------------------------------------------------------------------
// End Of File

// -----------------------------------------------------------------------------
// @file g_network.c
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_network.h"

#include <assert.h> // assert
#include <stdlib.h> // NULL, calloc, free
#include <time.h>   // time

#include "g_random.h" // g_random_seed

// -----------------------------------------------------------------------------

static void __unsafe_reset(g_network_t *self) {
    assert(self != NULL);
    // variables
    self->pages      = NULL;
    self->layers.ptr = NULL;
    self->layers.len = 0;

    // intrinsic
    self->_is_safe = false;
}

static bool Create(struct g_network_t *self, g_pages_t *pages) {
    bool rvalue = self != NULL;

    if (rvalue) {
        rvalue = g_network_pages_check(pages);

        const int L = rvalue ? pages->len : 0;

        if (rvalue) {
            self->layers.ptr = calloc(L, sizeof(g_layer_t));
            self->layers.len = L;

            rvalue = self->layers.ptr != NULL;
        }

        if (rvalue) {
            for (int k = 0; k < L; ++k) {
                g_layer_t *layer = &self->layers.ptr[k];
                g_page_t  *page  = &pages->ptr[k];

                g_layer_link(layer);

                rvalue = layer->Create(layer, page, k);

                if (!rvalue) {
                    break; // exit loop if layer creation fails
                }
            }
        }

        // check if layers are connected (it requires layer->Create first)
        if (rvalue) {
            for (int i = 0, j = 1; j < L; ++i, ++j) {
                f_vector_t *Yi = &pages->ptr[i].y;
                f_vector_t *Xj = &pages->ptr[j].x;

                rvalue = rvalue && (Yi->ptr == Xj->ptr);
                rvalue = rvalue && (Yi->len == Xj->len);
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->pages = pages;
        } else {
            self->Destroy(self);
        }
    }

    return rvalue;
}

static void Destroy(struct g_network_t *self) {
    if (self != NULL) {
        if (self->layers.ptr != NULL) {
            const int L = self->layers.len;

            for (int k = 0; k < L; ++k) {
                g_layer_t *layer = &self->layers.ptr[k];

                if (layer != NULL) {
                    layer->Destroy(layer);
                }
            }

            free(self->layers.ptr);
        }

        __unsafe_reset(self);
    }
}

static void Init_Weights(struct g_network_t *self, float bias) {
    if ((self != NULL) && self->_is_safe) {
        g_random_seed(time(NULL));

        const int L = self->layers.len;

        for (int k = 0; k < L; ++k) {
            g_layer_t *layer = &self->layers.ptr[k];

            layer->Init_Weights(layer, bias);
        }
    }
}

static void Step_Forward(struct g_network_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int L = self->layers.len;

        for (int k = 0; k < L; ++k) {
            g_layer_t *layer_k = &self->layers.ptr[k];

            layer_k->Step_Forward(layer_k);
        }
    }
}

static void Step_Errors(struct g_network_t *self, f_vector_t *actual_outputs) {
    if ((self != NULL) && self->_is_safe) {
        if (actual_outputs != NULL) {
            const int L = self->layers.len;

            g_layer_t *layer_L = &self->layers.ptr[L - 1];

            const int P = layer_L->page->y.len;

            if (P == actual_outputs->len) {
                float *Y_L     = layer_L->page->y.ptr;
                float *dE_dy_L = layer_L->page->de_dy.ptr;

                // MSE: we treat each output of the last layer as independent
                // from the other outputs. This simplification allows us to
                // calculate the error for each output independently, without
                // scaling it by the total number of outputs.

                for (int j = 0; j < P; ++j) {
                    dE_dy_L[j] = 2.0f * (Y_L[j] - actual_outputs->ptr[j]);
                }

                for (int k = L - 2; k >= 0; --k) {
                    g_layer_t *layer_k0 = &self->layers.ptr[k + 0];
                    g_layer_t *layer_k1 = &self->layers.ptr[k + 1];

                    layer_k0->Step_Errors(layer_k0, layer_k1);
                }
            }
        }
    }
}

static void Step_Backward(struct g_network_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int L = self->layers.len;

        for (int k = L - 1; k >= 0; --k) {
            g_layer_t *layer_k = &self->layers.ptr[k];

            layer_k->Step_Backward(layer_k);
        }
    }
}

bool g_network_pages_check(g_pages_t *pages) {
    bool rvalue = pages != NULL;

    if (rvalue) {
        rvalue = rvalue && (pages->ptr != NULL);
        rvalue = rvalue && (pages->len > 1); // at least 2 layers
    }

    if (rvalue) {
        const int L = pages->len;

        for (int k0 = 0; k0 < L - 1; ++k0) {
            for (int k1 = k0 + 1; k1 < L; ++k1) {
                rvalue = rvalue && (&pages->ptr[k0] != &pages->ptr[k1]);
            }
        }
    }

    return rvalue;
}

void g_network_link(g_network_t *self) {
    if (self != NULL) {
        // variables & intrinsic
        __unsafe_reset(self);

        // functions
        self->Create        = Create;
        self->Destroy       = Destroy;
        self->Init_Weights  = Init_Weights;
        self->Step_Forward  = Step_Forward;
        self->Step_Errors   = Step_Errors;
        self->Step_Backward = Step_Backward;
    }
}

// -----------------------------------------------------------------------------
// End Of File

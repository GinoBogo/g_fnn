// -----------------------------------------------------------------------------
// @file g_network.c
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_network.h"

#include <assert.h> // assert
#include <stdlib.h> // NULL, calloc, free, srand
#include <time.h>   // time

static void __unsafe_reset(g_network_t *self) {
    assert(self != NULL);
    // variables
    self->data       = NULL;
    self->layers.ptr = NULL;
    self->layers.len = 0;

    // intrinsic
    self->_is_safe = false;
}

static bool Create(struct g_network_t *self, g_layers_data_t *data) {
    bool rvalue = false;

    if (self != NULL) {
        rvalue = g_network_data_check(data);

        if (rvalue) {
            const int N = data->len;

            self->layers.ptr = calloc(N, sizeof(g_layer_t));
            self->layers.len = N;

            rvalue &= self->layers.ptr != NULL;
        }

        if (rvalue) {
            const int N = data->len;

            for (int l = 0; l < N; ++l) {
                g_layer_t      *layer      = &self->layers.ptr[l];
                g_layer_data_t *layer_data = &data->ptr[l];

                g_layer_link(layer);

                rvalue &= layer->Create(layer, layer_data, l);

                if (!rvalue) {
                    break; // exit loop if layer creation fails
                }
            }
        }

        if (rvalue) {
            const int N = data->len;

            for (int i = 0, j = 1; j < N; ++i, ++j) {
                f_vector_t *Yi = &data->ptr[i].y;
                f_vector_t *Xj = &data->ptr[j].x;

                rvalue &= Yi->ptr == Xj->ptr;
                rvalue &= Yi->len == Xj->len;
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->data = data; // "shallow copy"
        } else {
            self->Destroy(self);
        }
    }

    return rvalue;
}

static void Destroy(struct g_network_t *self) {
    if (self != NULL) {
        if (self->layers.ptr != NULL) {
            const int N = self->layers.len;

            for (int i = 0; i < N; ++i) {
                g_layer_t *layer = &self->layers.ptr[i];

                if (layer != NULL) {
                    layer->Destroy(layer);
                }
            }

            free(self->layers.ptr);
        }

        __unsafe_reset(self);
    }
}

static void Init_Weights(struct g_network_t *self) {
    if ((self != NULL) && self->_is_safe) {
        srand(time(NULL));

        const int N = self->layers.len;

        for (int i = 0; i < N; ++i) {
            g_layer_t *layer = &self->layers.ptr[i];

            layer->Init_Weights(layer);
        }
    }
}

static void Step_Forward(struct g_network_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int N = self->layers.len;

        for (int i = 0; i < N; ++i) {
            g_layer_t *layer = &self->layers.ptr[i];

            layer->Step_Forward(layer);
        }
    }
}

static void Step_Errors(struct g_network_t *self, f_vector_t *actual_outputs) {
    if ((self != NULL) && self->_is_safe) {
        if (actual_outputs != NULL) {
            const int L = self->layers.len;

            g_layer_t *output_layer = &self->layers.ptr[L - 1];

            const int N = output_layer->data->y.len;

            if (N == actual_outputs->len) {
                float *Y     = output_layer->data->y.ptr;
                float *dE_dy = output_layer->data->de_dy.ptr;

                for (int i = 0; i < N; ++i) {
                    dE_dy[i] = Y[i] - actual_outputs->ptr[i];
                }

                for (int i = L - 2; i >= 0; --i) {
                    g_layer_t *hidden_layer = &self->layers.ptr[i];

                    hidden_layer->Step_Errors(hidden_layer, &self->layers.ptr[i + 1]);
                }
            }
        }
    }
}

static void Step_Backward(struct g_network_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int N = self->layers.len;

        for (int i = N - 1; i >= 0; --i) {
            g_layer_t *layer = &self->layers.ptr[i];

            layer->Step_Backward(layer);
        }
    }
}

bool g_network_data_check(g_layers_data_t *data) {
    bool rvalue = false;

    if (data != NULL) {
        rvalue |= data != NULL;

        if (rvalue) {
            rvalue &= data->ptr != NULL;
            rvalue &= data->len > 0;
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

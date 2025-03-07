// -----------------------------------------------------------------------------
// @file g_layer.c
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_layer.h"

#include <assert.h> // assert
#include <math.h>   // expf
#include <stdlib.h> // NULL, calloc, free

#include "g_neuron.h"

static void __unsafe_reset(g_layer_t *self) {
    assert(self != NULL);
    // variables
    self->data        = NULL;
    self->l_id        = -1;
    self->neurons.ptr = NULL;
    self->neurons.len = 0;

    // intrinsic
    self->_is_safe = false;
}

static bool Create(struct g_layer_t *self, g_layer_data_t *data, int l_id) {
    bool rvalue = false;

    if (self != NULL) {
        rvalue = g_layer_data_check(data, l_id);

        const int N = data->y.len;

        if (rvalue) {
            self->neurons.ptr = calloc(N, sizeof(g_neuron_t));
            self->neurons.len = N;

            rvalue &= self->neurons.ptr != NULL;
        }

        if (rvalue) {
            for (int i = 0; i < N; ++i) {
                g_neuron_t *neuron = &self->neurons.ptr[i];

                g_neuron_link(neuron);

                rvalue &= neuron->Create(neuron, data, i);

                if (!rvalue) {
                    break; // exit loop if neuron creation fails
                }
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->data = data; // "shallow copy"
            self->l_id = l_id;
        } else {
            self->Destroy(self);
        }
    }

    return rvalue;
}

static void Destroy(struct g_layer_t *self) {
    if (self != NULL) {
        if (self->neurons.ptr != NULL) {
            const int N = self->neurons.len;

            for (int i = 0; i < N; ++i) {
                g_neuron_t *neuron = &self->neurons.ptr[i];

                neuron->Destroy(neuron);
            }

            free(self->neurons.ptr);
        }

        __unsafe_reset(self);
    }
}

static void Step_Fwd(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int N = self->neurons.len;

        g_neuron_t *neuron = self->neurons.ptr;

        for (int i = 0; i < N; ++i) {
            neuron[i].Step_Z(&neuron[i]);
        }

        if (neuron->data->af_type == SOFTMAX) {
            float sum_exp = 0.0;

            for (int i = 0; i < N; ++i) {
                sum_exp += expf(neuron->data->z.ptr[i]);
            }

            neuron->data->af_args.ptr[0] = sum_exp;
            neuron->data->af_args.len    = 1;
        }

        for (int i = 0; i < N; ++i) {
            neuron[i].Step_Y(&neuron[i]);
        }
    }
}

void g_layer_link(g_layer_t *self) {
    if (self != NULL) {
        // variables & intrinsic
        __unsafe_reset(self);

        // functions
        self->Create   = Create;
        self->Destroy  = Destroy;
        self->Step_Fwd = Step_Fwd;
    }
}

void g_layer_data_reset(g_layer_data_t *data) {
    if (data != NULL) {
        // forward propagation
        data->x.ptr = NULL;
        data->x.len = 0;
        data->w.ptr = NULL;
        data->w.row = 0;
        data->w.col = 0;
        data->z.ptr = NULL;
        data->z.len = 0;
        data->y.ptr = NULL;
        data->y.len = 0;

        // backward propagation
        data->dy_dz.ptr = NULL;
        data->dy_dz.len = 0;
        data->de_dy.ptr = NULL;
        data->de_dy.len = 0;

        data->af_type     = UNKNOWN;
        data->af_call     = NULL;
        data->af_args.ptr = NULL;
        data->af_args.len = 0;
    }
}

bool g_layer_data_check(g_layer_data_t *data, int l_id) {
    bool rvalue = false;

    if (data != NULL) {
        rvalue |= l_id >= 0;
        rvalue &= data->x.ptr != NULL;
        rvalue &= data->w.ptr != NULL;
        rvalue &= data->z.ptr != NULL;
        rvalue &= data->y.ptr != NULL;

        if (rvalue) {
            rvalue &= data->x.ptr != data->z.ptr;
            rvalue &= data->x.ptr != data->y.ptr;
            rvalue &= data->z.ptr != data->y.ptr;
        }

        if (rvalue) {
            const int N = data->w.row;

            for (int i = 0; i < N; ++i) {
                float *data__w_ptr = f_matrix_row(&data->w, i);

                rvalue &= data__w_ptr != NULL;
                rvalue &= data__w_ptr != data->x.ptr;
                rvalue &= data__w_ptr != data->z.ptr;
                rvalue &= data__w_ptr != data->y.ptr;
            }
        }

        if (rvalue) {
            rvalue &= data->x.len > 0;
            rvalue &= data->w.col == data->x.len + 1;
            rvalue &= data->w.row == data->y.len;
            rvalue &= data->w.row == data->z.len;
        }
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File

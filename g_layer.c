// -----------------------------------------------------------------------------
// @file g_layer.c
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_layer.h"

#include <assert.h> // assert
#include <math.h>   // expf, sqrtf
#include <stdlib.h> // NULL, RAND_MAX, calloc, free, rand

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

static void __xavier_init(float *weights, int n_in, int n_out) {
    float std_dev = sqrtf(6.0f / (n_in + n_out));

    for (int i = 0; i < n_in; i++) {
        // rand() does not generate uniform random numbers [0, 1]
        float number = (float)rand() / (float)RAND_MAX;

        weights[i] = (number * (2.0f * std_dev)) - std_dev;
    }
}

static bool Create(struct g_layer_t *self, g_layer_data_t *data, int l_id) {
    bool rvalue = false;

    if (self != NULL) {
        rvalue = g_layer_data_check(data, l_id);

        if (rvalue) {
            const int N = data->y.len;

            self->neurons.ptr = calloc(N, sizeof(g_neuron_t));
            self->neurons.len = N;

            rvalue &= self->neurons.ptr != NULL;
        }

        if (rvalue) {
            const int N = data->y.len;

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

                if (neuron != NULL) {
                    neuron->Destroy(neuron);
                }
            }

            free(self->neurons.ptr);
        }

        __unsafe_reset(self);
    }
}

static void Init_Weights(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int N = self->neurons.len;

        for (int i = 0; i < N; ++i) {
            float *weights_ptr = f_matrix_row(&self->data->w, i);
            int    weights_len = self->data->w.col;

            __xavier_init(weights_ptr, weights_len, N);
        }
    }
}

static void Step_Forward(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int N = self->neurons.len;

        g_neuron_t *neuron = self->neurons.ptr;

        for (int i = 0; i < N; ++i) {
            neuron[i].Step_Z(&neuron[i]);
        }

        if (neuron->data->af_type == SOFTMAX) {
            float sum_exp = 0.0f;

            const float *Z = neuron->data->z.ptr;

            for (int i = 0; i < N; ++i) {
                sum_exp += expf(Z[i]);
            }

            neuron->data->af_args.ptr[0] = sum_exp;
            neuron->data->af_args.len    = 1;
        }

        for (int i = 0; i < N; ++i) {
            neuron[i].Step_Y(&neuron[i]);
        }
    }
}

static void Step_Errors(struct g_layer_t *self, struct g_layer_t *next) {
    if ((self != NULL) && self->_is_safe) {
        if ((self != next) && (next != NULL)) {
            const int P0 = self->data->de_dy.len;
            const int P1 = next->data->de_dy.len;

            float *L0_dE_dy = self->data->de_dy.ptr;
            float *L1_dE_dy = next->data->de_dy.ptr;
            float *L1_dY_dz = next->data->dy_dz.ptr;

            for (int i = 0; i < P0; ++i) {
                L0_dE_dy[i] = 0.0f;

                for (int j = 0; j < P1; ++j) {
                    float L1_w_ij = *f_matrix_at(&next->data->w, j, i);

                    L0_dE_dy[i] += L1_dE_dy[j] * L1_dY_dz[j] * L1_w_ij;
                }
            }
        }
    }
}

static void Step_Backward(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
    }
}

void g_layer_link(g_layer_t *self) {
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
        // forward propagation
        rvalue &= data->x.ptr != NULL;
        rvalue &= data->w.ptr != NULL;
        rvalue &= data->z.ptr != NULL;
        rvalue &= data->y.ptr != NULL;

        // backward propagation
        rvalue &= data->dy_dz.ptr != NULL;
        rvalue &= data->de_dy.ptr != NULL;

        if (rvalue) {
            // forward propagation
            rvalue &= data->x.ptr != data->z.ptr;
            rvalue &= data->x.ptr != data->y.ptr;
            rvalue &= data->z.ptr != data->y.ptr;

            // backward propagation
            rvalue &= data->dy_dz.ptr != data->de_dy.ptr;

            rvalue &= data->dy_dz.ptr != data->x.ptr;
            rvalue &= data->dy_dz.ptr != data->z.ptr;
            rvalue &= data->dy_dz.ptr != data->y.ptr;

            rvalue &= data->de_dy.ptr != data->x.ptr;
            rvalue &= data->de_dy.ptr != data->z.ptr;
            rvalue &= data->de_dy.ptr != data->y.ptr;
        }

        if (rvalue) {
            const int N = data->w.row;

            for (int i = 0; i < N; ++i) {
                float *data__w_ptr = f_matrix_row(&data->w, i);

                rvalue &= data__w_ptr != NULL;

                // forward propagation
                rvalue &= data__w_ptr != data->x.ptr;
                rvalue &= data__w_ptr != data->z.ptr;
                rvalue &= data__w_ptr != data->y.ptr;

                // backward propagation
                rvalue &= data__w_ptr != data->dy_dz.ptr;
                rvalue &= data__w_ptr != data->de_dy.ptr;
            }
        }

        if (rvalue) {
            // forward propagation
            rvalue &= data->x.len > 0;
            rvalue &= data->w.col == data->x.len + 1;
            rvalue &= data->w.row == data->z.len;
            rvalue &= data->w.row == data->y.len;

            // backward propagation
            rvalue &= data->dy_dz.len == data->z.len;
            rvalue &= data->de_dy.len == data->y.len;
        }
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File

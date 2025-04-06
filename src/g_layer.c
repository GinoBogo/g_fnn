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

static void __he_uniform_init(float *weights, int fan_in) {
    const float std_dev = sqrtf(6.0f / fan_in);

    for (int i = 0; i < fan_in; ++i) {
        const float number = (float)rand() / (float)RAND_MAX;

        weights[i] = (number * (2.0f * std_dev)) - std_dev;
    }

    weights[fan_in] = 0.5f; // bias term
}

static void __xavier_uniform_init(float *weights, int fan_in, int fan_out) {
    const float std_dev = sqrtf(6.0f / (fan_in + fan_out));

    for (int i = 0; i < fan_in; ++i) {
        const float number = (float)rand() / (float)RAND_MAX;

        weights[i] = (number * (2.0f * std_dev)) - std_dev;
    }

    weights[fan_in] = 0.5f; // bias term
}

static bool Create(struct g_layer_t *self, g_layer_data_t *data, int l_id) {
    bool rvalue = self != NULL;

    if (rvalue) {
        rvalue = g_layer_data_check(data, l_id);

        const int P = rvalue ? data->y.len : 0;

        if (rvalue) {
            self->neurons.ptr = calloc(P, sizeof(g_neuron_t));
            self->neurons.len = P;

            rvalue = self->neurons.ptr != NULL;
        }

        if (rvalue) {
            for (int j = 0; j < P; ++j) {
                g_neuron_t *neuron = &self->neurons.ptr[j];

                g_neuron_link(neuron);

                rvalue = neuron->Create(neuron, data, j);

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
            const int P = self->neurons.len;

            for (int j = 0; j < P; ++j) {
                g_neuron_t *neuron = &self->neurons.ptr[j];

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
        const int fan_in  = self->data->x.len;
        const int fan_out = self->data->y.len;

        const g_act_func_type_t af_type = self->data->af_type;

        for (int j = 0; j < fan_out; ++j) {
            float *Wj = f_matrix_row(&self->data->w, j);

            switch (af_type) {
                case RELU:
                case LEAKY_RELU:
                case PRELU:
                case SWISH:
                case ELU:
                    __he_uniform_init(Wj, fan_in);
                    break;

                case TANH:
                case SIGMOID:
                case SOFTMAX:
                    __xavier_uniform_init(Wj, fan_in, fan_out);
                    break;
                default:
                    for (int i = 0; i <= fan_in; ++i) {
                        Wj[i] = 0.0f; // weights + bias
                    }
                    break;
            }
        }
    }
}

static void Step_Forward(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        const int P = self->neurons.len;

        g_neuron_t *neuron = self->neurons.ptr;

        for (int j = 0; j < P; ++j) {
            neuron[j].Step_Forward_Z(&neuron[j]);
        }

        if (self->data->af_type == SOFTMAX) {
            float sum_exp = 0.0f;

            const float *Z = self->data->z.ptr;

            for (int j = 0; j < P; ++j) {
                sum_exp += expf(Z[j]);
            }

            self->data->af_args.ptr[0] = sum_exp;
            self->data->af_args.len    = 1;
        }

        for (int j = 0; j < P; ++j) {
            neuron[j].Step_Forward_Y(&neuron[j]);
        }
    }
}

static void Step_Errors(struct g_layer_t *self, struct g_layer_t *next) {
    if ((self != NULL) && self->_is_safe) {
        if ((self != next) && (next != NULL) && next->_is_safe) {
            const int P0 = self->data->de_dy.len;
            const int P1 = next->data->de_dy.len;

            float *dE_dy_k0 = self->data->de_dy.ptr;
            float *dE_dy_k1 = next->data->de_dy.ptr;
            float *dy_dz_k1 = next->data->dy_dz.ptr;

            for (int j = 0; j < P0; ++j) {
                dE_dy_k0[j] = 0.0f;

                for (int i = 0; i < P1; ++i) {
                    float w_k1_ji = *f_matrix_at(&next->data->w, i, j);

                    dE_dy_k0[j] += dE_dy_k1[i] * dy_dz_k1[i] * w_k1_ji;
                }
            }
        }
    }
}

static void Step_Backward(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        float *dE_dy = self->data->de_dy.ptr;
        float *dy_dz = self->data->dy_dz.ptr;

        const int   P  = self->data->y.len; // number of neurons
        const int   N  = self->data->x.len; // number of inputs (all neurons)
        const float lr = self->data->lr;    // learning rate

        for (int j = 0; j < P; ++j) {
            const float dE_dz_j = dE_dy[j] * dy_dz[j];

            float *Xj = &self->data->x.ptr[j];
            float *Wj = f_matrix_row(&self->data->w, j);

            for (int i = 0; i < N; ++i) {
                Wj[i] -= lr * dE_dz_j * Xj[i];
            }

            Wj[N] -= lr * dE_dz_j;
        }
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
        data->l_id = -1;
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
    bool rvalue = data != NULL;

    if (rvalue) {
        rvalue = data->l_id == l_id;
        // forward propagation
        rvalue = rvalue && (data->x.ptr != NULL);
        rvalue = rvalue && (data->w.ptr != NULL);
        rvalue = rvalue && (data->z.ptr != NULL);
        rvalue = rvalue && (data->y.ptr != NULL);

        // backward propagation
        rvalue = rvalue && (data->dy_dz.ptr != NULL);
        rvalue = rvalue && (data->de_dy.ptr != NULL);

        if (rvalue) {
            // forward propagation
            rvalue = rvalue && (data->x.ptr != data->z.ptr);
            rvalue = rvalue && (data->x.ptr != data->y.ptr);
            rvalue = rvalue && (data->z.ptr != data->y.ptr);

            // backward propagation
            rvalue = rvalue && (data->dy_dz.ptr != data->de_dy.ptr);

            rvalue = rvalue && (data->dy_dz.ptr != data->x.ptr);
            rvalue = rvalue && (data->dy_dz.ptr != data->z.ptr);
            rvalue = rvalue && (data->dy_dz.ptr != data->y.ptr);

            rvalue = rvalue && (data->de_dy.ptr != data->x.ptr);
            rvalue = rvalue && (data->de_dy.ptr != data->z.ptr);
            rvalue = rvalue && (data->de_dy.ptr != data->y.ptr);
        }

        if (rvalue) {
            const int P = data->w.row;

            for (int j = 0; j < P; ++j) {
                float *data__w_ptr = f_matrix_row(&data->w, j);

                rvalue = data__w_ptr != NULL;

                if (!rvalue) {
                    break; // exit loop if null pointer
                }

                // forward propagation
                rvalue = rvalue && (data__w_ptr != data->x.ptr);
                rvalue = rvalue && (data__w_ptr != data->z.ptr);
                rvalue = rvalue && (data__w_ptr != data->y.ptr);

                // backward propagation
                rvalue = rvalue && (data__w_ptr != data->dy_dz.ptr);
                rvalue = rvalue && (data__w_ptr != data->de_dy.ptr);

                if (!rvalue) {
                    break; // exit loop if weights are invalid
                }
            }
        }

        if (rvalue) {
            // forward propagation
            rvalue = data->x.len > 0;
            rvalue = rvalue && (data->w.col == data->x.len + 1);
            rvalue = rvalue && (data->w.row == data->z.len);
            rvalue = rvalue && (data->w.row == data->y.len);

            // backward propagation
            rvalue = rvalue && (data->dy_dz.len == data->z.len);
            rvalue = rvalue && (data->de_dy.len == data->y.len);
        }
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File

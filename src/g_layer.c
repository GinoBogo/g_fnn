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
#include <stdlib.h> // NULL, calloc, free

#include "g_random.h" // g_random_range

// -----------------------------------------------------------------------------

static void __unsafe_reset(g_layer_t *self) {
    assert(self != NULL);
    // variables
    self->page        = NULL;
    self->l_id        = -1;
    self->neurons.ptr = NULL;
    self->neurons.len = 0;

    // intrinsic
    self->_is_safe = false;
}

static void __he_uniform_init(float *weights, int fan_in, float bias) {
    const float std_dev = sqrtf(6.0f / fan_in);

    for (int i = 0; i < fan_in; ++i) {
        weights[i] = g_random_range(-std_dev, std_dev);
    }

    weights[fan_in] = bias;
}

static void __xavier_uniform_init(float *weights, int fan_in, int fan_out, float bias) {
    const float std_dev = sqrtf(6.0f / (fan_in + fan_out));

    for (int i = 0; i < fan_in; ++i) {
        weights[i] = g_random_range(-std_dev, std_dev);
    }

    weights[fan_in] = bias;
}

static bool Create(struct g_layer_t *self, g_page_t *page, int l_id) {
    bool rvalue = self != NULL;

    if (rvalue) {
        rvalue = g_layer_page_check(page, l_id);

        const int P = rvalue ? page->y.len : 0;

        if (rvalue) {
            self->neurons.ptr = calloc(P, sizeof(g_neuron_t));
            self->neurons.len = P;

            rvalue = self->neurons.ptr != NULL;
        }

        if (rvalue) {
            for (int j = 0; j < P; ++j) {
                g_neuron_t *neuron = &self->neurons.ptr[j];

                g_neuron_link(neuron);

                rvalue = neuron->Create(neuron, page, j);

                if (!rvalue) {
                    break; // exit loop if neuron creation fails
                }
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->page = page; // "shallow copy"
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

static void Init_Weights(struct g_layer_t *self, float bias) {
    if ((self != NULL) && self->_is_safe) {
        const int fan_in  = self->page->x.len;
        const int fan_out = self->page->y.len;

        const g_act_func_type_t af_type = self->page->af_type;

        for (int j = 0; j < fan_out; ++j) {
            float *Wj = f_matrix_row(&self->page->w, j);

            switch (af_type) {
                case RELU:
                case LEAKY_RELU:
                case PRELU:
                case SWISH:
                case ELU:
                    __he_uniform_init(Wj, fan_in, bias);
                    break;

                case TANH:
                case SIGMOID:
                case SOFTMAX:
                    __xavier_uniform_init(Wj, fan_in, fan_out, bias);
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

        if (self->page->af_type == SOFTMAX) {
            const float *Z = self->page->z.ptr;

            float Z_max = Z[0];
            for (int j = 1; j < P; ++j) {
                if (Z[j] > Z_max)
                    Z_max = Z[j];
            }

            float sum_exp = expf(Z[0] - Z_max);
            for (int j = 1; j < P; ++j) {
                sum_exp += expf(Z[j] - Z_max);
            }

            self->page->af_args.ptr[0] = sum_exp;
            self->page->af_args.ptr[1] = Z_max;
            self->page->af_args.len    = 2;
        }

        for (int j = 0; j < P; ++j) {
            neuron[j].Step_Forward_Y(&neuron[j]);
        }
    }
}

static void Step_Errors(struct g_layer_t *self, struct g_layer_t *next) {
    if ((self != NULL) && self->_is_safe) {
        if ((self != next) && (next != NULL) && next->_is_safe) {
            const int P0 = self->page->de_dy.len;
            const int P1 = next->page->de_dy.len;

            float *dE_dy_k0 = self->page->de_dy.ptr;
            float *dE_dy_k1 = next->page->de_dy.ptr;
            float *dy_dz_k1 = next->page->dy_dz.ptr;

            for (int j = 0; j < P0; ++j) {
                dE_dy_k0[j] = 0.0f;

                for (int i = 0; i < P1; ++i) {
                    float w_k1_ji = *f_matrix_at(&next->page->w, i, j);

                    dE_dy_k0[j] += dE_dy_k1[i] * dy_dz_k1[i] * w_k1_ji;
                }
            }
        }
    }
}

static void Step_Backward(struct g_layer_t *self) {
    if ((self != NULL) && self->_is_safe) {
        float *dE_dy = self->page->de_dy.ptr;
        float *dy_dz = self->page->dy_dz.ptr;

        const int   P  = self->page->y.len; // number of neurons
        const int   N  = self->page->x.len; // number of inputs (all neurons)
        const float lr = self->page->lr;    // learning rate

        for (int j = 0; j < P; ++j) {
            const float dE_dz_j = dE_dy[j] * dy_dz[j];

            float *Xj = &self->page->x.ptr[j];
            float *Wj = f_matrix_row(&self->page->w, j);

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

void g_layer_page_reset(g_page_t *page) {
    if (page != NULL) {
        page->l_id = -1;
        // forward propagation
        page->x.ptr = NULL;
        page->x.len = 0;
        page->w.ptr = NULL;
        page->w.row = 0;
        page->w.col = 0;
        page->z.ptr = NULL;
        page->z.len = 0;
        page->y.ptr = NULL;
        page->y.len = 0;

        // backward propagation
        page->dy_dz.ptr = NULL;
        page->dy_dz.len = 0;
        page->de_dy.ptr = NULL;
        page->de_dy.len = 0;

        page->af_type     = UNKNOWN;
        page->af_call     = NULL;
        page->af_args.ptr = NULL;
        page->af_args.len = 0;
    }
}

bool g_layer_page_check(g_page_t *page, int l_id) {
    bool rvalue = page != NULL;

    if (rvalue) {
        rvalue = page->l_id == l_id;
        // forward propagation
        rvalue = rvalue && (page->x.ptr != NULL);
        rvalue = rvalue && (page->w.ptr != NULL);
        rvalue = rvalue && (page->z.ptr != NULL);
        rvalue = rvalue && (page->y.ptr != NULL);

        // backward propagation
        rvalue = rvalue && (page->dy_dz.ptr != NULL);
        rvalue = rvalue && (page->de_dy.ptr != NULL);

        if (rvalue) {
            // forward propagation
            rvalue = rvalue && (page->x.ptr != page->z.ptr);
            rvalue = rvalue && (page->x.ptr != page->y.ptr);
            rvalue = rvalue && (page->z.ptr != page->y.ptr);

            // backward propagation
            rvalue = rvalue && (page->dy_dz.ptr != page->de_dy.ptr);

            rvalue = rvalue && (page->dy_dz.ptr != page->x.ptr);
            rvalue = rvalue && (page->dy_dz.ptr != page->z.ptr);
            rvalue = rvalue && (page->dy_dz.ptr != page->y.ptr);

            rvalue = rvalue && (page->de_dy.ptr != page->x.ptr);
            rvalue = rvalue && (page->de_dy.ptr != page->z.ptr);
            rvalue = rvalue && (page->de_dy.ptr != page->y.ptr);
        }

        if (rvalue) {
            const int P = page->w.row;

            for (int j = 0; j < P; ++j) {
                float *page__w_ptr = f_matrix_row(&page->w, j);

                rvalue = page__w_ptr != NULL;

                if (!rvalue) {
                    break; // exit loop if null pointer
                }

                // forward propagation
                rvalue = rvalue && (page__w_ptr != page->x.ptr);
                rvalue = rvalue && (page__w_ptr != page->z.ptr);
                rvalue = rvalue && (page__w_ptr != page->y.ptr);

                // backward propagation
                rvalue = rvalue && (page__w_ptr != page->dy_dz.ptr);
                rvalue = rvalue && (page__w_ptr != page->de_dy.ptr);

                if (!rvalue) {
                    break; // exit loop if weights are invalid
                }
            }
        }

        if (rvalue) {
            // forward propagation
            rvalue = page->x.len > 0;
            rvalue = rvalue && (page->w.col == page->x.len + 1);
            rvalue = rvalue && (page->w.row == page->z.len);
            rvalue = rvalue && (page->w.row == page->y.len);

            // backward propagation
            rvalue = rvalue && (page->dy_dz.len == page->z.len);
            rvalue = rvalue && (page->de_dy.len == page->y.len);
        }
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File

// -----------------------------------------------------------------------------
// @file g_neuron.c
//
// @date December, 2024
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#include "g_neuron.h"

#include <assert.h> // assert
#include <math.h>   // expf, tanhf
#include <stdlib.h> // NULL

// -----------------------------------------------------------------------------

float *f_matrix_row(f_matrix_t *mat, int row) {
    float *rvalue = NULL;

    if ((mat != NULL) && (mat->ptr != NULL)) {
        const bool chk_1 = row >= 0;
        const bool chk_2 = mat->row > row;
        const bool chk_3 = mat->col > 0;

        if (chk_1 && chk_2 && chk_3) {
            rvalue = mat->ptr + (row * mat->col);
        }
    }

    return rvalue;
}

float *f_matrix_at(f_matrix_t *mat, int row, int col) {
    float *rvalue = NULL;

    if ((mat != NULL) && (mat->ptr != NULL)) {
        const bool chk_1 = row >= 0 && col >= 0;
        const bool chk_2 = mat->row > row;
        const bool chk_3 = mat->col > col;

        if (chk_1 && chk_2 && chk_3) {
            rvalue = mat->ptr + (row * mat->col + col);
        }
    }

    return rvalue;
}

f_vector_t f_matrix_vector(f_matrix_t *mat, int row) {
    f_vector_t rvalue;

    rvalue.ptr = f_matrix_row(mat, row);
    rvalue.len = mat->col;

    return rvalue;
}

// -----------------------------------------------------------------------------

static void __af_linear(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    *Y = *Z;

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = 1.0f;
}

static void __af_tanh(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    *Y = tanhf(*Z);

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = 1.0f - ((*Y) * (*Y));
}

static void __af_relu(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    *Y = *Z > 0.0f ? *Z : 0.0f;

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : 0.0f;
}

static void __af_leaky_relu(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    const float alpha = data->af_args.ptr[0];

    *Y = *Z > 0.0f ? *Z : alpha * (*Z);

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? alpha : 0.0f;
}

static void __af_prelu(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    const float beta = data->af_args.ptr[n_id];

    *Y = *Z > 0.0f ? *Z : beta * (*Z);

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? beta : 0.0f;
}

static void __af_swish(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    const float sigma = 1.0f / (1.0f + expf(-(*Z)));

    *Y = (*Z) * sigma;

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = (*Y) + sigma * (1.0f - (*Y));
}

static void __af_elu(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    const float alpha = data->af_args.ptr[0];

    *Y = *Z > 0.0f ? *Z : alpha * (expf(*Z) - 1.0f);

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : (*Y) + alpha;
}

static void __af_sigmoid(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    *Y = 1.0f / (1.0f + expf(-(*Z)));

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    *dY_dZ = (*Y) * (1.0f - (*Y));
}

static void __af_softmax(g_layer_data_t *data, int n_id) {
    float *Z = &data->z.ptr[n_id];
    float *Y = &data->y.ptr[n_id];

    const float sum_exp = data->af_args.ptr[0];

    *Y = expf(*Z) / sum_exp;

    float *dY_dZ = &data->dy_dz.ptr[n_id];

    const float sigma = 1.0f / (1.0f + expf(-(*Z)));

    *dY_dZ = sigma * (1.0f - sigma);
}

static void __unsafe_reset(g_neuron_t *self) {
    assert(self != NULL);
    // variables
    self->data = NULL;
    self->n_id = -1;

    // intrinsic
    self->_is_safe = false;
}

static bool Create(struct g_neuron_t *self, g_layer_data_t *data, int n_id) {
    bool rvalue = false;

    if (self != NULL) {
        rvalue = g_neuron_data_check(data, n_id);

        const bool first_time = rvalue && (data->af_call == NULL);

        if (first_time) {
            switch (data->af_type) {
                case LINEAR: {
                    data->af_call = __af_linear;
                } break;

                case TANH: {
                    data->af_call = __af_tanh;
                } break;

                case RELU: {
                    data->af_call = __af_relu;
                } break;

                case LEAKY_RELU: {
                    data->af_call = __af_leaky_relu;

                    rvalue &= data->af_args.ptr != NULL;
                    rvalue &= data->af_args.len > 0;
                } break;

                case PRELU: {
                    data->af_call = __af_prelu;

                    rvalue &= data->af_args.ptr != NULL;
                    rvalue &= data->af_args.len == data->y.len;
                } break;

                case SWISH: {
                    data->af_call = __af_swish;
                } break;

                case ELU: {
                    data->af_call = __af_elu;

                    rvalue &= data->af_args.ptr != NULL;
                    rvalue &= data->af_args.len > 0;
                } break;

                case SIGMOID: {
                    data->af_call = __af_sigmoid;
                } break;

                case SOFTMAX: {
                    data->af_call = __af_softmax;

                    rvalue &= data->af_args.ptr != NULL;
                    rvalue &= data->af_args.len > 0;
                } break;

                default: { // fallback
                    data->af_call = __af_linear;
                } break;
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->data = data; // "shallow copy"
            self->n_id = n_id;
        } else {
            self->Destroy(self);
        }
    }

    return rvalue;
}

static void Destroy(struct g_neuron_t *self) {
    if (self != NULL) {
        __unsafe_reset(self);
    }
}

static void Step_Z(struct g_neuron_t *self) {
    if ((self != NULL) && self->_is_safe) {
        g_layer_data_t *data = self->data;
        const int       n    = self->n_id;
        const int       N    = data->x.len;

        float *X = data->x.ptr;
        float *W = f_matrix_row(&data->w, n);
        float *Z = data->z.ptr;

        Z[n] = W[N];

        for (int i = 0; i < N; ++i) {
            Z[n] += W[i] * X[i];
        }
    }
}

static void Step_Y(struct g_neuron_t *self) {
    if ((self != NULL) && self->_is_safe) {
        g_layer_data_t *data = self->data;
        const int       n    = self->n_id;

        data->af_call(data, n);
    }
}

void g_neuron_link(g_neuron_t *self) {
    if (self != NULL) {
        // variables & intrinsic
        __unsafe_reset(self);

        // functions
        self->Create  = Create;
        self->Destroy = Destroy;
        self->Step_Z  = Step_Z;
        self->Step_Y  = Step_Y;
    }
}

bool g_neuron_data_check(g_layer_data_t *data, int n_id) {
    bool rvalue = false;

    if (data != NULL) {
        rvalue |= n_id >= 0;
        rvalue &= data->z.len > n_id;
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End Of File

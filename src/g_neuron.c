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

static void __af_linear(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    *Y = *Z;

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = 1.0f;
}

static void __af_tanh(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    *Y = tanhf(*Z);

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = 1.0f - ((*Y) * (*Y));
}

static void __af_relu(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    *Y = *Z > 0.0f ? *Z : 0.0f;

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : 0.0f;
}

static void __af_leaky_relu(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float alpha = page->af_args.ptr[0];

    *Y = *Z > 0.0f ? *Z : alpha * (*Z);

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : alpha;
}

static void __af_prelu(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float beta = page->af_args.ptr[n_id];

    *Y = *Z > 0.0f ? *Z : beta * (*Z);

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : beta;
}

static void __af_swish(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float sigma = 1.0f / (1.0f + expf(-(*Z)));

    *Y = (*Z) * sigma;

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = (*Y) + sigma * (1.0f - (*Y));
}

static void __af_elu(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float alpha = page->af_args.ptr[0];

    *Y = *Z > 0.0f ? *Z : alpha * (expf(*Z) - 1.0f);

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = *Z > 0.0f ? 1.0f : (*Y) + alpha;
}

static void __af_softplus(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float u = expf(*Z);
    const float v = 1.0f + u;

    *Y = logf(v);

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = u / v;
}

static void __af_sigmoid(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    *Y = 1.0f / (1.0f + expf(-(*Z)));

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = (*Y) * (1.0f - (*Y));
}

static void __af_softmax(g_page_t *page, int n_id) {
    float *Z = &page->z.ptr[n_id];
    float *Y = &page->y.ptr[n_id];

    const float sum_exp = page->af_args.ptr[0];
    const float Z_max   = page->af_args.ptr[1];

    *Y = expf(*Z - Z_max) / sum_exp;

    float *dY_dZ = &page->dy_dz.ptr[n_id];

    *dY_dZ = (*Y) * (1.0f - (*Y));
}

static void __unsafe_reset(g_neuron_t *self) {
    assert(self != NULL);
    // variables
    self->page = NULL;
    self->n_id = -1;

    // intrinsic
    self->_is_safe = false;
}

static bool Create(struct g_neuron_t *self, g_page_t *page, int n_id) {
    bool rvalue = false;

    if (self != NULL) {
        rvalue = g_neuron_page_check(page, n_id);

        const bool first_time = rvalue && (page->af_call == NULL);

        if (first_time) {
            switch (page->af_type) {
                case LINEAR: {
                    page->af_call = __af_linear;
                } break;

                case TANH: {
                    page->af_call = __af_tanh;
                } break;

                case RELU: {
                    page->af_call = __af_relu;
                } break;

                case LEAKY_RELU: {
                    page->af_call = __af_leaky_relu;

                    rvalue = rvalue && (page->af_args.ptr != NULL);
                    rvalue = rvalue && (page->af_args.len > 0);
                } break;

                case PRELU: {
                    page->af_call = __af_prelu;

                    rvalue = rvalue && (page->af_args.ptr != NULL);
                    rvalue = rvalue && (page->af_args.len == page->y.len);
                } break;

                case SWISH: {
                    page->af_call = __af_swish;
                } break;

                case ELU: {
                    page->af_call = __af_elu;

                    rvalue = rvalue && (page->af_args.ptr != NULL);
                    rvalue = rvalue && (page->af_args.len > 0);
                } break;

                case SOFTPLUS: {
                    page->af_call = __af_softplus;
                } break;

                case SIGMOID: {
                    page->af_call = __af_sigmoid;
                } break;

                case SOFTMAX: {
                    page->af_call = __af_softmax;
                } break;

                default: { // fallback
                    page->af_call = __af_linear;
                } break;
            }
        }

        self->_is_safe = rvalue;

        if (rvalue) {
            self->page = page; // "shallow copy"
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

static void Step_Forward_Z(struct g_neuron_t *self) {
    if ((self != NULL) && self->_is_safe) {
        g_page_t *page = self->page;
        const int j    = self->n_id;  // j-th neuron
        const int N    = page->x.len; // number of inputs

        float *Xj = page->x.ptr;
        float *Wj = f_matrix_row(&page->w, j);
        float *Z  = page->z.ptr;

        Z[j] = Wj[N]; // bias of j-th neuron

        for (int i = 0; i < N; ++i) {
            Z[j] += Wj[i] * Xj[i]; // i-th input of j-th neuron
        }
    }
}

static void Step_Forward_Y(struct g_neuron_t *self) {
    if ((self != NULL) && self->_is_safe) {
        g_page_t *page = self->page;
        const int j    = self->n_id; // j-th neuron

        page->af_call(page, j);
    }
}

void g_neuron_link(g_neuron_t *self) {
    if (self != NULL) {
        // variables & intrinsic
        __unsafe_reset(self);

        // functions
        self->Create         = Create;
        self->Destroy        = Destroy;
        self->Step_Forward_Z = Step_Forward_Z;
        self->Step_Forward_Y = Step_Forward_Y;
    }
}

bool g_neuron_page_check(g_page_t *page, int n_id) {
    bool rvalue = page != NULL;

    if (rvalue) {
        // prevent out-of-bounds access to Z (and then W and Y)
        rvalue = rvalue && (n_id >= 0);
        rvalue = rvalue && (page->z.len > n_id);
    }

    return rvalue;
}

// -----------------------------------------------------------------------------
// End of File

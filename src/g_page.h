// -----------------------------------------------------------------------------
// @file g_page.h
//
// @date April, 2025
//
// @author Gino Francesco Bogo
// -----------------------------------------------------------------------------

#ifndef G_PAGE_H
#define G_PAGE_H

// -----------------------------------------------------------------------------

typedef struct f_vector_t {
    float *ptr;
    int    len;
} f_vector_t;

typedef struct f_matrix_t {
    float *ptr;
    int    row; // number of neurons in a layer
    int    col; // number of weights per neuron
} f_matrix_t;

extern float *f_matrix_row(f_matrix_t *mat, int row);

extern float *f_matrix_at(f_matrix_t *mat, int row, int col);

// -----------------------------------------------------------------------------
/*
 * CNN  - Convolutional Neural Network
 * FFN  - Feed-Forward Network
 * SFFN - Shallow Feed-Forward Network
 * RNN  - Recurrent Neural Network
 * LSTM - Long Short-Term Memory (networks)
 */

typedef enum g_act_func_type_t {
    LINEAR,     // for input layer (inputs value retaining)
    TANH,       // for hidden layers (SFFN, RNN, LSTM)
    RELU,       // for hidden layers (Deep FFN, CNN)
    LEAKY_RELU, // for hidden layers (Deep FFN, CNN)
    PRELU,      // for hidden layers (Deep FFN, CNN)
    SWISH,      // for hidden layers (Deep FFN, CNN)
    ELU,        // for hidden layers (Deep FFN, CNN)
    SIGMOID,    // for output layer (binary classification)
    SOFTMAX,    // for output layer (multi-class classification)
    UNKNOWN = -1
} g_act_func_type_t;

struct g_page_t; // forward declaration

typedef void (*g_act_func_call_t)(struct g_page_t *page, int n_id);

// -----------------------------------------------------------------------------

typedef struct g_act_func_args_t {
    float *ptr;
    int    len;
} g_act_func_args_t;

// -----------------------------------------------------------------------------

typedef struct g_page_t {
    int l_id; // layer index
    // forward propagation
    f_vector_t x; // X[neuron]
    f_matrix_t w; // W[layer][neuron]
    f_vector_t z; // Z[layer]
    f_vector_t y; // Y[layer]

    // backward propagation
    f_vector_t dy_dz; // dY/dZ[layer]
    f_vector_t de_dy; // dE/dY[layer]
    float      lr;    // learning rate

    // activation function
    g_act_func_type_t af_type;
    g_act_func_call_t af_call;
    g_act_func_args_t af_args;
} g_page_t;

typedef struct g_pages_t {
    g_page_t *ptr;
    int       len;
} g_pages_t;

// -----------------------------------------------------------------------------

extern void g_page_reset(g_page_t *page);

#endif // G_PAGE_H

// -----------------------------------------------------------------------------
// End of File

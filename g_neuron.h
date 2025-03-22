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

extern f_vector_t f_matrix_vector(f_matrix_t *mat, int row);

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

struct g_layer_data_t; // forward declaration

typedef void (*g_act_func_call_t)(struct g_layer_data_t *data, int n_id);

typedef struct g_act_func_args_t {
    float *ptr;
    int    len;
} g_act_func_args_t;

// -----------------------------------------------------------------------------

typedef struct g_layer_data_t {
    // forward propagation
    f_vector_t x; // neuron's inputs
    f_matrix_t w; // neuron's weights
    f_vector_t z; // layer's weighted sums
    f_vector_t y; // layer's outputs

    // backward propagation
    f_vector_t dy_dz; // layer's dy/dz derivatives
    f_vector_t de_dy; // layer's dE/dy derivatives

    g_act_func_type_t af_type;
    g_act_func_call_t af_call;
    g_act_func_args_t af_args;
} g_layer_data_t;

typedef struct g_layers_data_t {
    g_layer_data_t *ptr;
    int             len;
} g_layers_data_t;

// -----------------------------------------------------------------------------

typedef struct g_neuron_t {
    // variables
    g_layer_data_t *data;
    int             n_id; // neuron index

    // functions
    bool (*Create)(struct g_neuron_t *self, g_layer_data_t *data, int n_id);
    void (*Destroy)(struct g_neuron_t *self);
    void (*Step_Z)(struct g_neuron_t *self);
    void (*Step_Y)(struct g_neuron_t *self);

    // intrinsic
    bool _is_safe;
} g_neuron_t;

typedef struct g_neurons_t {
    g_neuron_t *ptr;
    int         len;
} g_neurons_t;

// -----------------------------------------------------------------------------

extern void g_neuron_link(g_neuron_t *self);

extern bool g_neuron_data_check(g_layer_data_t *data, int n_id);

#endif // G_NEURON_H

// -----------------------------------------------------------------------------
// End Of File

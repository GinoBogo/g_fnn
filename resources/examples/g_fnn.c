#include <math.h>   // expf, sqrtf
#include <stdio.h>  // NULL
#include <stdlib.h> // RAND_MAX, calloc, free, malloc, rand, srand
#include <time.h>   // time

extern void load_network_inputs(float **x, int neurons);
extern void load_actual_outputs(float *actual_y, int neurons);

// Activation function: SIGMOID
static float g_sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

// Derivative of SIGMOID
static float d_sigmoid(float y) {
    return y * (1.0f - y);
}

// Activation function: RELU
static float g_relu(float z) {
    return z > 0.0f ? z : 0.0f;
}

// Derivative of RELU
static float d_relu(float y) {
    return y > 0.0f ? 1.0f : 0.0f;
}

// He initialization
static void he_init(float *weights, int size) {
    float std_dev = sqrtf(2.0f / size);

    for (int i = 0; i < size; i++) {
        weights[i] = (((float)rand() / (float)RAND_MAX) * (2.0f * std_dev)) - std_dev;
    }
}

// Free memory if needed
static void free_mem(void *ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

int main(void) {
    // Seed for random number generation
    srand(time(NULL));

    const int L   = 4;               // number of layers
    const int P[] = {7, 20, 20, 10}; // number of neurons in each layer

    // Matrices of pointers: [layer: k][neuron: j][input: i]
    float **x[L];
    float **w[L];
    // Arrays of pointers: [layer: k][neuron: j]
    float *z[L];
    float *y[L];
    float *dE_dy[L];
    float *dy_dz[L];

    // Allocate memory
    for (int k = 0; k < L; k++) {
        if (k == 0) {
            // Input layer (no weights, no biases, no derivatives)
            y[0] = (float *)calloc(P[0], sizeof(float));
        } else {
            // Hidden and output layers
            x[k] = (float **)calloc(P[k], sizeof(float *));
            w[k] = (float **)calloc(P[k], sizeof(float *));

            z[k]     = (float *)calloc(P[k], sizeof(float));
            y[k]     = (float *)calloc(P[k], sizeof(float));
            dE_dy[k] = (float *)calloc(P[k], sizeof(float));
            dy_dz[k] = (float *)calloc(P[k], sizeof(float));

            for (int j = 0; j < P[k]; j++) {
                x[k][j] = (float *)calloc(P[k - 1], sizeof(float));
                w[k][j] = (float *)calloc(P[k - 1] + 1, sizeof(float));
                // Initialize weights
                he_init(w[k][j], P[k - 1] + 1);
            }
        }
    }

    float *actual_y = (float *)calloc(P[L - 1], sizeof(float));

    // Load network inputs directly into y[0] (i.e., input layer outputs)
    load_network_inputs(&y[0], P[0]);

    // Forward pass
    for (int k = 1; k < L; k++) {
        for (int j = 0; j < P[k]; j++) {
            // Initialize z with bias
            z[k][j] = w[k][j][P[k - 1]];

            // Calculate weighted sum of inputs
            for (int i = 0; i < P[k - 1]; i++) {
                z[k][j] += w[k][j][i] * y[k - 1][i];
            }

            // Apply activation function
            y[k][j] = (k == L - 1) ? g_sigmoid(z[k][j]) : g_relu(z[k][j]);
        }
    }

    // Backward pass
    load_actual_outputs(actual_y, P[L - 1]);

    // Calculate dy/dz for all layers
    for (int k = L - 1; k > 0; k--) {
        for (int j = 0; j < P[k]; j++) {
            dy_dz[k][j] = (k == L - 1) ? d_sigmoid(y[k][j]) : d_relu(y[k][j]);
        }
    }

    // Calculate dE/dy for all layers
    for (int k = L - 1; k > 0; k--) {
        if (k == L - 1) {
            // Output layer
            for (int j = 0; j < P[k]; j++) {
                dE_dy[k][j] = y[k][j] - actual_y[j];
            }
        } else {
            // Hidden layers
            for (int i = 0; i < P[k - 1]; i++) {
                dE_dy[k][i] = 0.0f;

                for (int j = 0; j < P[k]; j++) {
                    dE_dy[k][i] += dE_dy[k + 1][j] * dy_dz[k + 1][j] * w[k + 1][j][i];
                }
            }
        }
    }

    // Learning rate
    const float lr[] = {0.03f, 0.02f, 0.01f};

    // Update weights
    for (int k = L - 1; k > 0; k--) {
        for (int j = 0; j < P[k]; j++) {
            float delta = dE_dy[k][j] * dy_dz[k][j];

            // Update bias
            w[k][j][P[k - 1]] -= lr[k - 1] * delta;

            // Update other weights
            for (int i = 0; i < P[k - 1]; i++) {
                w[k][j][i] -= lr[k - 1] * delta * y[k - 1][i];
            }
        }
    }

    // Release memory
    for (int k = 0; k < L; k++) {
        if (k == 0) {
            free_mem(y[0]);
        } else {
            for (int j = 0; j < P[k]; j++) {
                free_mem(x[k][j]);
                free_mem(w[k][j]);
            }
            free_mem(x[k]);
            free_mem(w[k]);

            free_mem(y[k]);
            free_mem(z[k]);
            free_mem(dE_dy[k]);
            free_mem(dy_dz[k]);
        }
    }

    free_mem(actual_y);

    return 0;
}

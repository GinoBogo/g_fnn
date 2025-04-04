#include <math.h>   // expf, sqrtf
#include <stdio.h>  // NULL, size_t
#include <stdlib.h> // RAND_MAX, calloc, free, malloc, rand, srand
#include <time.h>   // time

extern void load_network_inputs(float **x, size_t neurons);
extern void load_actual_outputs(float *actual_y, size_t neurons);

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

// Xavier initialization
static void xavier_init(float *weights, int size) {
    float std_dev = sqrtf(2.0f / size);

    for (int i = 0; i < size; i++) {
        weights[i] = (((float)rand() / (float)RAND_MAX) * (2.0f * std_dev)) - std_dev;
    }
}

int main(void) {
    // Seed for random number generation
    srand(time(NULL));

    const int L   = 4;               // number of layers
    const int P[] = {7, 20, 20, 10}; // number of neurons in each layer

    // Matrices of pointers: [layer][neuron][input]
    float **x[L];
    float **w[L];
    // Arrays of pointers: [layer][neuron]
    float *z[L];
    float *y[L];
    float *dE_dy[L];
    float *dy_dz[L];

    // Allocate memory
    for (int l = 0; l < L; l++) {
        x[l] = (float **)malloc(P[l] * sizeof(float *));

        for (int n = 0; n < P[l]; n++) {
            if (l == 0) {
                x[l][n] = (float *)calloc(1, sizeof(float));
            } else {
                x[l][n] = (float *)calloc(P[l - 1], sizeof(float));
            }
        }

        if (l > 0) {
            w[l] = (float **)malloc(P[l] * sizeof(float *));

            for (int n = 0; n < P[l]; n++) {
                w[l][n] = (float *)calloc(P[l - 1] + 1, sizeof(float));
                // Initialize weights
                xavier_init(w[l][n], P[l - 1] + 1);
            }
        }

        z[l]     = (float *)calloc(P[l], sizeof(float));
        y[l]     = (float *)calloc(P[l], sizeof(float));
        dE_dy[l] = (float *)calloc(P[l], sizeof(float));
        dy_dz[l] = (float *)calloc(P[l], sizeof(float));
    }

    // Fetch inputs and calculate predicted outputs
    load_network_inputs(x[0], P[0]);

    for (int l = 0; l < L; l++) {
        if (l == 0) {
            // Input layer, just copy inputs
            for (int n = 0; n < P[l]; n++) {
                // Assuming input is stored in x[l][n][0]
                y[l][n] = x[l][n][0];
            }
        } else {
            for (int n = 0; n < P[l]; n++) {
                z[l][n] = 0.0f;

                for (int i = 0; i < P[l - 1]; i++) {
                    z[l][n] += y[l - 1][i] * w[l][n][i];
                }

                z[l][n] += w[l][n][P[l - 1]]; // Add bias

                if (l == L - 1) {
                    // Use sigmoid for the output layer
                    y[l][n] = g_sigmoid(z[l][n]);
                } else {
                    // Use ReLU for hidden layers
                    y[l][n] = g_relu(z[l][n]);
                }
            }
        }
    }

    // Load actual outputs
    float *actual_y = (float *)calloc(P[L - 1], sizeof(float));
    load_actual_outputs(actual_y, P[L - 1]);

    // Calculate dE/dy for the last layer
    for (int i = 0; i < P[L - 1]; i++) {
        // Assuming MSE as error function
        dE_dy[L - 1][i] = y[L - 1][i] - actual_y[i];
    }

    free(actual_y);

    // Calculate dy/dz for all layers
    for (int n = 0; n < P[L - 1]; n++) {
        // Use sigmoid for the output layer
        dy_dz[L - 1][n] = d_sigmoid(y[L - 1][n]);
    }

    for (int l = 1; l < L - 1; l++) {
        for (int n = 0; n < P[l]; n++) {
            // Use ReLU for hidden layers
            dy_dz[l][n] = d_relu(y[l][n]);
        }
    }

    // Calculate dE/dy for previous layers
    for (int l = L - 2; l > 0; l--) {
        for (int i = 0; i < P[l]; i++) {
            dE_dy[l][i] = 0.0f;

            for (int j = 0; j < P[l + 1]; j++) {
                dE_dy[l][i] += dE_dy[l + 1][j] * dy_dz[l + 1][j] * w[l + 1][j][i];
            }
        }
    }

    // Calculate output layer errors (dE/dy)
    for (int i = 0; i < P[L - 1]; i++) {
        dE_dy[L - 1][i] = y[L - 1][i] - actual_y[i];
    }

    // Calculate hidden layers errors
    for (int l = L - 2; l > 0; l--) {
        for (int i = 0; i < P[l]; i++) {
            dE_dy[l][i] = 0.0f;
            for (int j = 0; j < P[l + 1]; j++) {
                dE_dy[l][i] += dE_dy[l + 1][j] * dy_dz[l + 1][j] * w[l + 1][j][i];
            }
        }
    }

    // Learning rate
    const float lr[] = {0.01f, 0.01f, 0.01f, 0.01f};

    // Update weights
    for (int l = 1; l < L; l++) {
        for (int j = 0; j < P[l]; j++) {
            float delta = dE_dy[l][j] * dy_dz[l][j];

            // Update bias
            w[l][j][P[l - 1]] -= lr[l] * delta;

            // Update other weights
            for (int i = 0; i < P[l - 1]; i++) {
                w[l][j][i] -= lr[l] * delta * y[l - 1][i];
            }
        }
    }

    // Release memory
    for (int l = 0; l < L; l++) {
        for (int n = 0; n < P[l]; n++) {
            free(x[l][n]);
        }
        free(x[l]);

        if (l > 0) {
            for (int n = 0; n < P[l]; n++) {
                free(w[l][n]);
            }
            free(w[l]);
        }

        free(z[l]);
        free(y[l]);
        free(dE_dy[l]);
        free(dy_dz[l]);
    }

    return 0;
}

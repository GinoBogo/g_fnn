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

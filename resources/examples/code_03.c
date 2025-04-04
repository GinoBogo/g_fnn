// Code to update weights

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

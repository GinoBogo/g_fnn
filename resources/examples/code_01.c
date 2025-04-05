/* Fully-connected Neural Network:
 * every neuron in one layer is connected to every neuron in the next layer.
 */

const int Ph = 7;  // number of neurons in layer h = k-1
const int Pk = 10; // number of neurons in layer k

float X[Pk][Ph];
float W[Pk][Ph + 1];
float Z[Pk];
float Y[Pk];

// Step 1: load X inputs (layer k) from Y outputs (layer k-1)
// ...

// Step 2: weighted sum of inputs and weights
for (int j = 0; j < Pk; ++j) {
    Z[j] = W[j][Ph];

    for (int i = 0; i < Ph; ++i) {
        Z[j] += W[j][i] * X[j][i];
    }
}

// Step 3: output calculation
for (int j = 0; j < Pk; ++j) {
    Y[j] = g(Z[j]);
}

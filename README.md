# Feed-forward Neural Network

![Fully-connected Neural Network](./resources/images/g_ffn_fig00.png)

### Inputs of the $j$-th neuron:

(1) $\ \ \ \ x_{ij}^{(k)} = y_{i}^{(k-1)}$

where:

- $0 \leq i \lt P_{k-1}$
- $0 \leq j \lt P_{k}$

and:

- $P_{k}$ is the number of neurons in layer $k$
- $P_{k-1}$ is the number of neurons in layer $k-1$


### Output of the $j$-th neuron:

(2) $\ \ \ \ y_{j}^{(k)} = g^{k}(z_{j}^{(k)})$

where:

- $g^{(k)}$ is the **Activation Function** of layer $k$

and:

(3) $\ \ \ \ z_{j}^{(k)} = \sum_{i=0}^{P_{k}-1} w_{ij}^{(k)} \cdot x_{ij}^{(k)} + b_{j}^{(k)}$

Using (1) in (3) and writing the **bias** $b_{j}^{(k)}$ as neuron's weight $w_{P_{k}j}^{(k)}$:

(4) $\ \ \ \ z_{j}^{(k)} = \sum_{i=0}^{P_{k}-1} w_{ij}^{(k)} \cdot y_{i}^{(k-1)} + w_{P_{k}j}^{(k)} \cdot 1$

~~~C
/* Fully-connected Neural Network */

float Xj[N];
float Wj[N+1];
float Zj;
float Yj;

/* load X inputs (layer k) from Y outputs (layer k-1) */

Zj = Wj[N];
for (int i = 0; i < N; ++i) {
	Zj += Wj[i] * Xj[i];
}

Yj = g(Zj);
~~~
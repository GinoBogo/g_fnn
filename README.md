# Feed-forward Neural Network

![Neurons Connection](./resources/images/g_ffn_fig00.png)

### Inputs of the $j$-th neuron:

1. $\ \ \ \ x_{ij}^{(k)} = y_{i}^{(k-1)}$

where:

- $0 \leq i \lt P_{k-1}$

- $0 \leq j \lt P_{k}$

and:

- $P_{k}$ is the number of neurons in layer $k$

- $P_{k-1}$ is the number of neurons in layer $k-1$


### Output of the $j$-th neuron:

2. $\ \ \ \ y_{j}^{(k)} = g^{k}(z_{j}^{(k)})$

where:

- $g^{(k)}$ is the **Activation Function** of layer $k$

and:

3. $\ \ \ \ z_{j}^{(k)} = \sum_{i=0}^{P_{k}-1} w_{ij}^{(k)} \cdot x_{ij}^{(k)} + b_{j}^{(k)}$

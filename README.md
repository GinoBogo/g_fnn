# Feed-forward Neural Network
Used in thousands of applications, Feed-forward Neural Networks are fundamental to deep learning. Their main advantage is structural flexibility, making them adaptable to various types of problems. A Feed-forward Neural Network with at least one hidden layer and sufficient neurons can approximate any continuous function, demonstrating its versatility and power as a _universal approximator_ in modeling complex behaviors. Fully-connected Neural Networks are a subset of Feed-forward Neural Networks and will be the focus of the following sections. 

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

Using (1) in (3) and writing the **bias** $b_{j}^{(k)}$ as neuron's weight $w_{P_{k}j}^{(k)}$ we have:

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
Regarding the partial derivatives of $y$ and $z$ we have:

(5) $\ \ \ \ \frac{\partial y_{j}^{(k)}}{\partial z_{j}^{(k)}} = \frac{\partial g^{(k)}(z_{j}^{(k)})}{\partial z_{j}^{(k)}} = g'^{(k)}(z_{j}^{(k)})$


(6) $\ \ \ \ \frac{\partial z_{j}^{(k)}}{\partial w_{ij}^{(k)}} = y_{i}^{(k-1)}$

(7) $\ \ \ \ \frac{\partial z_{j}^{(k)}}{\partial y_{i}^{(k-1)}} = w_{ij}^{(k)}$

### Activation Functions:
Activation Functions are mathematical equations that determine the output of a neural network's node and introduce non-linearity, enabling the network to model complex data patterns.

| Acronym | Full Name                            |
|---------|--------------------------------------|
| CNN     | Convolutional Neural Network         |
| FFN     | Feed-Forward Network                 |
| SFFN    | Shallow Feed-Forward Network         |
| RNN     | Recurrent Neural Network             |
| LSTM    | Long Short-Term Memory (network)     |

Some of the most commonly used Activation Functions and their layer-by-layer applicability are:

| Name       | Applicability                                |
|------------|----------------------------------------------|
| LINEAR     | for input layer (inputs value retaining)     |
| TANH       | for hidden layers (SFFN, RNN, LSTM)          |
| RELU       | for hidden layers (Deep FFN, CNN)            |
| LEAKY_RELU | for hidden layers (Deep FFN, CNN)            |
| PRELU      | for hidden layers (Deep FFN, CNN)            |
| SWISH      | for hidden layers (Deep FFN, CNN)            |
| ELU        | for hidden layers (Deep FFN, CNN)            |
| SIGMOID    | for output layer (binary classification)     |
| SOFTMAX    | for output layer (multi-class classification)|

| Name | Expression | Derivative |
|--|-----|-----|
| LINEAR | $g(z) = z$ | $g'(z) = 1$ |
| TANH | $g(z) = \tanh(z)$ | $g'(z) = 1 - \tanh^2(z)$ |
| RELU | $g(z) = \max(0, z)$ | $g'(z) = 0 \ \ \ \ \text{if } z \leq 0; \ \ \ \ g'(z) = 1 \ \ \ \ \text{if } z > 0$ |
| LEAKY_RELU | $g(z) = \alpha z \ \ \ \ \text{if } z \leq 0; \ \ \ \ g(z) = z  \ \ \ \ \text{if } z > 0$ | $g'(z) = \alpha \ \ \ \ \text{if } z \leq 0; \ \ \ \ g'(z) = 1 \ \ \ \ \text{if } z > 0$ |
| PRELU | $g(z) = \beta z \ \ \ \ \text{if } z \leq 0; \ \ \ \ g(z) = z \ \ \ \ \text{if } z > 0$ | $g'(z) = \beta \ \ \ \ \text{if } z \leq 0; \ \ \ \ g'(z) = 1 \ \ \ \ \text{if } z > 0$ |
| SWISH | $g(z) = z \cdot \sigma(z)$ | $g'(z) = g(z) + \sigma(z) [1 - g(z)]$ |
| ELU | $g(z) = \alpha (e^z - 1) \ \ \ \ \text{if } z \leq 0; \ \ \ \ g(z) = z \ \ \ \ \text{if } z > 0$ | $g'(z) = g(z) + \alpha \ \ \ \ \text{if } z \leq 0; \ \ \ \ g'(z) = 1 \ \ \ \ \text{if } z > 0$ |
| SIGMOID | $g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$ | $g'(z) = \sigma(z) [1 - \sigma(z)]$ |
| SOFTMAX | $g_i(z) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $g'_i(z) = \sigma_i(z) [1 - \sigma_i(z)]$ |

Leaky ReLU and PReLU look identical at first, but Leaky ReLU uses a fixed small slope ($\alpha$, typically 0.01) for negative values, while PReLU learns the slope ($\beta$) during training, providing more flexibility.
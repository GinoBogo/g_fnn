# Feed-forward Neural Network
##### Personal notes and reflections

Used in thousands of applications, Feed-forward Neural Networks are fundamental to deep learning. Their main advantage is structural flexibility, making them adaptable to various types of problems. A Feed-forward Neural Network with at least one hidden layer and sufficient neurons can approximate any continuous function, demonstrating its versatility and power as a _universal approximator_ in modeling complex behaviors. Fully-connected Neural Networks are a subset of Feed-forward Neural Networks and will be the focus of the following sections. 

![Fig. 0](./resources/images/g_ffn_fig00.png)

Fully-connected Neural Networks (also known as Dense Neural Networks) are a type of artificial neural network where every neuron in one layer is connected to every neuron in the next layer.

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

Using (1) in (3) and writing the **bias** $b_{j}^{(k)}$ in terms of neuron's weight $w_{P_{k}j}^{(k)}$ we have:

(4) $\ \ \ \ z_{j}^{(k)} = \sum_{i=0}^{P_{k}-1} w_{ij}^{(k)} \cdot y_{i}^{(k-1)} + w_{P_{k}j}^{(k)} \cdot 1$

~~~C
/* Fully-connected Neural Network:
 * every neuron in one layer is connected to every neuron in the next layer.
 */

float Xj[Pk];
float Wj[Pk+1];
float Zj;
float Yj;

/* step 1: load X inputs (layer k) from Y outputs (layer k-1) */

/* step 2: inputs weighting */
Zj = Wj[N];
for (int i = 0; i < N; ++i) {
	Zj += Wj[i] * Xj[i];
}

/* step 3: output calculation */
Yj = g(Zj);
~~~
Regarding the partial derivatives of $y$ and $z$ we have:

(5) $\ \ \ \ \frac{\partial y_{j}^{(k)}}{\partial z_{j}^{(k)}} = \frac{\partial g^{(k)}(z_{j}^{(k)})}{\partial z_{j}^{(k)}} = g'^{(k)}(z_{j}^{(k)})$


(6) $\ \ \ \ \frac{\partial z_{j}^{(k)}}{\partial w_{ij}^{(k)}} = y_{i}^{(k-1)}$

(7) $\ \ \ \ \frac{\partial z_{j}^{(k)}}{\partial y_{i}^{(k-1)}} = w_{ij}^{(k)}$

Their usefulness will be clear in the following sections.

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

Overview of the above-mentioned Activation Functions: definitions and derivatives for use in neural networks.

| Name | Definition | Derivative |
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

### Error Functions:
Error functions, also known as Loss Functions or Cost Functions, are used to measure the difference between the predicted output of a neural network and the actual output. The goal is to minimize this difference, which is typically achieved through optimization algorithms.

Common Error Functions are:

| Name | Definition | Applicability |
|--|-----|-----|
| **Mean Squared Error**  | $MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2$ | Regression |
| **Mean Absolute Error** | $MAE = \frac{1}{n} \sum_{i=1}^n abs(y_i - \hat{y_i})$ | Regression |
| **Cross Entropy** | $CE = -\sum_{i=1}^n y_i \log(\hat{y_i}) - (1 - y_i) \log(1 - \hat{y_i})$ | Classification (binary) |
| **Binary Cross Entropy** | $BCE = -\sum_{i=1}^n y_i \log(\hat{y_i}) - (1 - y_i) \log(1 - \hat{y_i})$ | Classification (binary) |

where:
- $n$ is the number of samples
- $y_i$ is the actual output
- $\hat{y_i}$ is the predicted output

Next, we will focus on the **Mean Squared Error** (MSE) error function, which is calculated for the output layer neurons.

![Fig. 1](resources/images/g_ffn_fig01.png)

The contribution of the generic $n$-th neuron to the MSE is:

(8) $\ \ \ \ E_n^{(L)} = \frac{1}{2} \left(y_n^{(L)} - y_n\right)^2$

The corresponding $n$-th **gradient** is:

(9) $\ \ \ \ \frac{\partial E_n^{(L)}}{\partial y_n^{(L)}} = y_n^{(L)} - y_n$

where:

- $\frac{1}{2}$ is a normalization factor that does not affect the gradient's meaning,

- $L$ is the number of layers in the neural network.

Intuitively, the gradient $\frac{\partial E}{\partial y}$ measures how the error $E$ changes with variations in the output $y$. Since $E(y,\hat{y})$ is a non-negative function, a negative gradient implies that the error is decreasing. A decreasing error clearly indicates that the network training is progressing as expected.

The MSE for the output layer is:

(10) $\ \ \ \ E_T^{(L)} = \sum_{n=0}^{P_L-1} E_n^{(L)}$

and the $j$-th gradient for the output layer is:

(11) $\ \ \ \ \frac{\partial E_T^{(L)}}{\partial y_j^{(L)}} = \frac{\partial}{\partial y_j^{(L)}} \sum_{n=0}^{P_L-1} E_n^{(L)}$

There are not interactions between the outputs of the neurons in the output layer. This means that the total gradient is simply the sum of the gradients of the output layer neurons. Thus, the equation (11) becomes:

(12) $\ \ \ \ \frac{\partial E_T^{(L)}}{\partial y_j^{(L)}} = \sum_{n=0}^{P_L-1} \frac{\partial E_n^{(L)}}{\partial y_j^{(L)}}$

Using (9) in (12) and considering that $\frac{\partial E_n^{(L)}}{\partial y_j^{(L)}} = 0$ for $j \neq n$, we have:

(13) $\ \ \ \ \frac{\partial E_T^{(L)}}{\partial y_j^{(L)}} \equiv \frac{\partial E_j^{(L)}}{\partial y_j^{(L)}} \Rightarrow y_j^{(L)} - y_j$

Equation (13) tells us how sensitive the error $E_j^{(L)}$ is to changes in the output $y_j^{(L)}$. These changes in $y_j$, in turn, depend on variations in $z_j$, given that $y_j = g(z_j)$. To find the relationship between variations in $E_j^{(L)}$ and variations in $z_j^{(L)}$, it worths recalling the **chain rule** for derivatives:

(14) $\ \ \ \ \frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$

In other words, the derivative of a **composite function** $f\left(g(x)\right)$ with respect to $x$ is the product of the derivative of $f$ with respect to $g$ and the derivative of $g$ with respect to $x$. Applying this rule to the error $E_j^{(L)}$ we have:

(15) $\ \ \ \ \frac{\partial E_j^{(L)}}{\partial z_j^{(L)}} = \frac{\partial E_j^{(L)}}{\partial y_j^{(L)}} \cdot \frac{\partial y_j^{(L)}}{\partial z_j^{(L)}}$

and using (5) in (15):

(16) $\ \ \ \ \frac{\partial E_j^{(L)}}{\partial z_j^{(L)}} = \frac{\partial E_j^{(L)}}{\partial y_j^{(L)}} \cdot g'^{(L)}\left(z_j^{(L)}\right)$

By applying the same method, it is possible to determine the relationship between variations in $E_j^{(L)}$ and variations in $y_i^{(L-1)}$, which correspond to the output of the $i$-th neuron in the previous layer $L-1$. As a result, we have:

(17) $\ \ \ \ \frac{\partial E_j^{(L)}}{\partial y_i^{(L-1)}} = \frac{\partial E_j^{(L)}}{\partial z_j^{(L)}} \cdot \frac{\partial z_j^{(L)}}{\partial y_i^{(L-1)}}$

Using (7) and (16) in (17) we have:

(18) $\ \ \ \ \frac{\partial E_j^{(L)}}{\partial y_i^{(L-1)}} = \frac{\partial E_j^{(L)}}{\partial y_j^{(L)}} \cdot g'^{(L)}\left(z_j^{(L)}\right) \cdot w_{ij}^{(L)}$

Equation (18) represents the amount of MSE variation that the $j$-th neuron in layer $L$ receives from the $i$-th neuron in layer $L-1$. However, the $i$-th neuron also supplies MSE variations to other neurons in layer $L$. Thus, the total amount of MSE variations that depend on the $i$-th neuron is:

(19) $\ \ \ \ \frac{\partial E_i^{(L-1)}}{\partial y_i^{(L-1)}} = \sum_{j=0}^{P_{L-1}-1} \frac{\partial E_j^{(L)}}{\partial y_i^{(L-1)}}$

Using (17) in (19) we have:

(20) $\ \ \ \ \frac{\partial E_i^{(L-1)}}{\partial y_i^{(L-1)}} = \sum_{j=0}^{P_{L-1}-1} \frac{\partial E_j^{(L)}}{\partial y_j^{(L)}} \cdot g'^{(L)}\left(z_j^{(L)}\right) \cdot w_{ij}^{(L)}$

![Fig. 2](resources/images/g_ffn_fig02.png)

### Back-propagation
Back-propagation is a key **learning algorithm** for artificial neural networks that calculates the gradient of the Error Function with respect to the network's weights by applying the **chain rule**. It adjusts the weights iteratively to minimize the error and improve the model's predictions. 
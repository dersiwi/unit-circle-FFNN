# Problem description
Train a neural network to detect, if a point $p = (x,y)^\top \in [0,1]^2$ lies within the unit circle or not. Of course this problem is very easy to decide by using the formula :
$$\sqrt{x^2 + y^2} \leq 1$$

### Problem input
 - points $p$ for our neural network:
    $$p = \begin{pmatrix}
        x\\y
    \end{pmatrix}, \quad x,y \in[0,1]$$
 
 - Radius of the circle $r = 1$

### Why?
This problem might seem like it is a bit overkill to be solved by 'AI', but the main reason is to go through the process of setting up the network by hand, understanding how a feed forward neural network works, learning gradient descent and working out the necessary gradients (by hand). <br>
Furthermore, after the network is setup it really can be trained to solve <i>any</i> problem that that has two parameters as inputs (sort of), given the correcet dataset.

# The network

## Architecture 
The network consists of two layers, not counting the input layer. 

<div align="center">
<img src="https://github.com/dersiwi/unit-circle-FFNN/blob/master/images/netwrkArchitecture_colored.svg">
</div>
The network architecture is arbitrary, the amout of hidden layers and the size of each hidden layer can be chosen at random. Only the input layer defines how many inputs the network has, and the output layer the amount of outputs.


## Forward pass
A forward pass refers to the calclation process of the network. In this network, this works as follows : 

$$y_{prediciton}  =\sigma(\theta_2 ^\top \sigma(\theta_1 x + b_1) + b_2)$$
In order to understand that better here is a breakdown

   1. Multiply the input $x \in \mathbb{R}^2$ with the weight matrix $\theta_1 \in \mathbb{R}^{4 \times 2}$ and add the vector $b_1 \in \mathbb{R}^4$ to each column:
   
   $$a_1 = \theta_1 x  +b_1 \in \mathbb{R}^4$$



   2. Apply the activation function $\sigma$ to each component of $a_1$:
   
   $$h_1 = \sigma(a_1) = \begin{pmatrix}
    \sigma(a_{1,1}) \\
    \sigma(a_{1,2}) \\
    \sigma(a_{1,3}) \\
    \sigma(a_{1,4}) \\
   \end{pmatrix} \in \mathbb{R}^4$$

   3. Multiply the vector $h_1$ with the second weights vector $\theta_2 \in \mathbb{R}^{4}$ and add the bias scalar $b_2 \in \mathbb{R}$:
   $$a_2 = \theta_2^\top h_1 + b_2=\theta_2^\top \sigma(\theta_1 x) + b_2\in \mathbb{R}$$

   4. Apply $\sigma$ function one more time to get the final prediciton of the network:
   
   $$ y_{prediciton} = \sigma(a_2) = \sigma(\theta_2 ^\top \sigma(\theta_1 x + b_1) + b_2)$$

### Interpreting $y_{prediciton}$
In this particular example, given a point $p = (x,y)$ the output $y_{prediction}$ is the networks estimation of how likely it is for that point $p$ to be inside the unit circle.
<br>
This of course is dependent on the labeling of the data.


## Training the network
Training the network means, setting the weights and biases stored in $\theta_1, \theta_2, b_1, b_2$. Now how do we do that? <br>

### Initialization

Of course without any training or testing there is no way to know good values for either $\theta_1, \theta_2, b_1, b_2$, so an easy solution (in the beginning) is just to initialize all values with random numbers in the range of $[0,1]$. 

### The loss-function
The loss function $J$ describes a funciton that takes $m$ guesses by the network and sums up their error squared, also called the mean-squared-error-loss:
$$J = \frac{1}{m} \sum_i (y_{pred} - y_i)^2$$

where $m$ is the amount of training points, $y_{pred}$ is the prediction of the network, given input $x_i$ and $y_i$ is the correct labeling of the datapoint:
$$y_i = 1 \Leftrightarrow \sqrt{x^2 + y^2} \leq 1$$
otherwise, $y_i = 0$. <br>
$y_i$ is called the label of the training data $x_i$ because it provides the corrext slution - meaining it tells us, if $x_i$ is inside or outside the circle.

### Gradient descent 

To set $\theta_1, \theta_2$ and $b_1, b_2$ we can use our loss funciton, which tells us how 'bad' the neural network currently performs. More precisely - our netowrk performing good is directly equivalent to the loss funciton being minimal. So thats what we do; we <ins>minimize</ins> the loss function $J$. <br>

The method we want to do this with is called <i>gradient descent</i>.  <br>
Because $J$ is so complex, it is impossible to solve for the global minimum directly. Gradient descent basically walks in the direction of a mininmum, by calculating the gradients 
$$\frac{\partial J}{\partial \theta_1} \quad \text{ and } \quad \frac{\partial J}{\partial \theta_2}$$
and subtracting them, along with a multiplier $\alpha$  from $\theta_1$ and $\theta_2$:

$$\theta_1 = \theta_1 - \alpha\frac{\partial J}{\partial \theta_1} \quad 
\text{ and } \quad 
\theta_2 = \theta_2 - \alpha \frac{\partial J}{\partial \theta_2}$$

Doing this over and over again iteratively minimizes $J$ (at least locally). Notice, that the same process is also done to the biases $b_1, b_2$.
### The gradients of the loss-function

Now that we know, how our network is going to train itself, we have to do one very important step, which is actually 'calculating' the gradients:

$$
\frac{\partial J}{\partial \theta_2} =
\textcolor{turquoise}{
    \frac{\partial J}{\partial y_{p}} 
    \frac{\partial y_{p}}{\partial a_2} }
\frac{\partial a_2}{\partial \theta_2}
\quad \text{and} \quad 
\frac{\partial J}{\partial b_2} = 
\textcolor{turquoise}{
    \frac{\partial J}{\partial y_{p}} 
    \frac{\partial y_{p}}{\partial a_2} }
\frac{\partial a_2}{\partial b_2}
$$

$$
\frac{\partial J}{\partial \theta_1} = 
\textcolor{violet}{
    \frac{\partial J}{\partial y_{p}} 
    \frac{\partial y_{p}}{\partial a_2} } 
\textcolor{violet}{
    \frac{\partial a_2}{\partial h_1}
    \frac{\partial h_1}{\partial a_1}
} 
\frac{\partial a_1}{\partial \theta_1}
\quad \text{and} \quad 
\frac{\partial J}{\partial \theta_1} = 
\textcolor{violet}{
    \frac{\partial J}{\partial y_{p}} 
    \frac{\partial y_{p}}{\partial a_2} } 
\textcolor{violet}{
    \frac{\partial a_2}{\partial h_1}
    \frac{\partial h_1}{\partial a_1}
} 
\frac{\partial a_1}{\partial b_1}
$$


This is basically just the chain-rule applied to the initial gradients. This type of differentiation might seem strange at first, but if you recall [these calculations](#forward-pass) they might seem more sensible.
Now, because there is a lot of redundance, we can write this a lot more efficiently (this translates directly into the performance of the training process):

$$
\frac{\partial J}{\partial \theta_2} =
\textcolor{turquoise}{\frac{\partial J}{\partial a_2}}
\frac{\partial a_2}{\partial \theta_2}
\quad \text{and} \quad 
\frac{\partial J}{\partial b_2} = 
\textcolor{turquoise}{\frac{\partial J}{\partial a_2}} 
\frac{\partial a_2}{\partial b_2}
\quad \text{,} \quad
\frac{\partial J}{\partial \theta_1} = 
\textcolor{violet}{\frac{\partial J}{\partial a_1}}  
\frac{\partial a_1}{\partial \theta_1}
\quad \text{and} \quad 
\frac{\partial J}{\partial \theta_1} = 
\textcolor{violet}{\frac{\partial J}{\partial a_1}} 
\frac{\partial a_1}{\partial b_1}
$$

That being cleaned up, here are the reappearing derivatives: 

$$
\textcolor{turquoise}{\frac{\partial J}{\partial a_2}} = 2(y_{pred} - y) * \sigma'(a_2), \quad 
\textcolor{violet}{\frac{\partial J}{\partial a_1}}  = 
\textcolor{turquoise}{\frac{\partial J}{\partial a_2}} 
\theta_2^\top \sigma'(a_1) 
$$

The last compoenents of the gradients are : 

$$
\frac{\partial a_1}{\partial \theta_1}
=h_1^\top
, \quad
\frac{\partial a_1}{\partial b_1}
= \mathbb{1}^\top, \quad
\frac{\partial a_2}{\partial \theta_2}
= X^\top
,\quad
\frac{\partial a_2}{\partial b_2}
= \mathbb{1}^\top
$$

Notice that (theoretically) the gradient of $\frac{\partial a_2}{\partial \theta_2}$ is a 3D-tensor, but in this case we do not have to evaluate the tensor itself. <br>
Also, $X \in \mathbb{R}^{m \times 2}$ represents all our training data stored in a matrix.

## Results

<div align="center">

| # | Training Accuracy  | Testing Accuracy | Training samples | Testing Samples |
| --- | :---: | :---: | :---: |:---: |
| 1 | 99.5%  | 98.6%  | 200 | 10.000 |
| 2 | 100%  | 96.25%  | 150 | 10.000 |
| 3 | 78.44%  | 78.86%  | 5000 | 10.000 |

</div>


As one can see the accuracy of the network after (and during) training may vary depending on the amount of data samples used for training. 
Clearly apparent in #3 more training-samples does not automatically lead to better performance (accuracy) of the network. 

### Hyperparameters and seed
Hyperparameters are parameters that are not set by the network during the learning process, for example the amount of epochs. Following parameters were used to get the above results: 

<div align="center">

|Parameter| Setting|
| :---: | :---: |
| Amount of Epochs | 10.000|
| learning rate $\alpha$ | 0.1|
| random_seed | 69|
</div>

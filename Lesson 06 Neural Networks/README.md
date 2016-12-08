# Lesson 6 - Neural Networks

In biology a neural network is a set of **neurons** connected together by links
called **axons**. Neurons, when **excited**, send electric impulses across axons
which generate synapses.

In computer science, a particular type of neuron is called **Perceptron**.

## Perceptron

  * Inputs `x_i`: the input values.
  * Weights `w_i`: which determine the sensitivity of the neuron.
  * Firing threshold `theta`: a value that determines whether the output of the
    neuron is 1 or 0.
  * Output `y`: is computed by summing `x_i * w_i` and if the sum is above
    the firing threshold `theta`, the neuron is set to the active state (1),
    on the other hand, the neuron is set to the idle/inactive state (0).

If we draw on the plane the function `x2 = f(x1)` we find that the activation
threshold is a line that divides the plane in two regions.
The region above the diving line is where the neuron is activated, whereas the
region below the diving line is where the neuron is inactivated.

fig(plane)

### `AND` function

Under the following hypotheses:

  * x1 can assume only binary values (1 and 0),
  * x2 can assume only binary values (1 and 0),
  * w1 and w2 are 1/2
  * theta is 3/4

We can conclude that the activation function of the neuron is exactly the
boolean `AND` function. Of course other values of theta, w1 and w2 would work.

fig(truth table)

### `OR` function

Under the following hypotheses:

  * x1 can assume only binary values (1 and 0),
  * x2 can assume only binary values (1 and 0),
  * w1 and w2 are 3/4
  * theta is 1/2

We can conclude that the activation function of the neuron is exactly the
boolean `OR` function. Of course other values of theta, w1 and w2 would work.

### `NOT` function

Under the following hypotheses:

  * x1 can assume only binary values (1 and 0),
  * w1 is -1
  * theta is -1/2

We can conclude that the activation function of the neuron is exactly the
boolean `NOT` function. Of course other values of theta and w1 would work.

## Figuring the weights in the neural network

### Perceptron Rule - Single Unit (threshold)

Given a training set, we want to set the weights of the network so to capture
the given dataset. In other words, we want to iteratively set the weights of
the network so to minimize its error on the dataset.

In particular, we are going to define the new weight `new_w_i = w_i + delta`
where `delta = learning_rate * (y - y_tilda) * x_i`. `y_tilda` is the output of
the network whereas `y` is the target output.

If we have a dataset that is **linearly separable** then the Perceptron rule
will actually find the plane that separates the values in the dataset with a
finite number of iterations.

**Note** the `y_tilda` function is the step function which is not differentiable.
For this reason, we cannot compute the gradient on this function.

However, we can make the step function differentiable by getting rid of it and
using another function similar to it. The new function is the Sigmoid.

`Sigmoid(a) = 1 / (1 + exp(-a))`

fig(sigmoid)

The derivate of the Sigmoid function is just `Sig(x) * (1 - Sig(x))`.

fig(proof derivate sigmoid)

### Gradient Descent Rule (unthresholded)

This algorithm will work smoothly on **non-linear separable datasets** but
converges to a local optimal.

We want to minimize the error `E(w) = ∑(y - x_i * w_i)^2` by computing the new
weights as `new_w_i = w_i + delta` where
`delta = learning_rate * (y - a) * x_i`. `y_tilda` is the output of
the network, `y` is the target output and `a` is the activation of the neuron.

## Back propagation

Given the nature of the weights when using the Sigmoid function in our neural
network, we can introduce the idea of **Back Propagation** where the information
flows from the input neurons to the output neurons whereas the error flows on
the other direction.

## Optimizing Weights

To boost the gradient descent performances, there are several techniques to
exploit when we want to find the correct values for the weights.

  * **Momentum terms** in the gradient, like in physics, when a ball is rolling,
    it has a momentum. The same idea can apply to the gradient descent where we
    continue to explore one direction even though the gradient in that direction
    is not the steepest gradient we have found.
  * **Higher order derivatives**, computing 2nd and 3rd order derivatives to
    figure which is the better direction to follow when optimizing the weights.
  * **Randomize optimization**.
  * **Penalty for "complexity"**, too many neurons or too many layers in the network.

## Restriction Bias

Representational power, set of hypotheses we will consider.

  * Boolean function: network of threshold-like unit
  * Continuous function: as long as the network is connected, i.e. there are no
    jumps in the network, we can represent the continuous function with a single
    hidden layer network.
  * Arbitrary: multiple hidden layers without jumps.

## Preference Bias

Gives you a metric for the algorithm you have to prefer in training you network.

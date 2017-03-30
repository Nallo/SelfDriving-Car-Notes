# Kalman Filter

_Kalman filtering, is an algorithm that uses a series of measurements observed
over time, containing statistical noise and other inaccuracies, and produces
estimates of unknown variables that tend to be more precise than those based on
a single measurement alone._ - [Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)

In Self Driving cars, the Kalman Filter algorithm is exploited to combine raw
data gathered from:

  * Radar sensors
  * Lidar sensors

In particular, through the Kalman Filter algorithm, a Self Driving car is capable
of understanding its surrounding.

Due to the need of real-time computation, Kalman Filters are typically implemented
using a high-performance programming language such as: C++.

## How It Works

The Kalman Filter represents our distributions by guassians and iterates on two
main cycles.

### 1. Measurement Update

In this phase, two mathematical concepts are taken into account:

  * The product of the measurements
  * The [Bayes Rule](https://en.wikipedia.org/wiki/Bayes%27_rule)

Let assume we have:

  * a **prior probability** which is a Normal Distribution with mean **u1** and
    variance **s1**;
  * a **measurement probability** which is a Normal Distribution with mean **u2**
    and variance **s2**;

We want to compute the **posterior probability** as result of the **measurement
update**. The new Normal Distribution will be a Gaussian with

  * `u = (u1*s2 + u2*s1)/(s1 + s2)`
  * `s = 1 / [(1/s1) + (1/s2)]`

In python this math is coded as follows:

```python

def update(mean1, var1, mean2, var2):
    """ Given the mean and variance of two different Gaussian Distributions,
        it computes the mean and variance of the new Gaussian of the measurement.
    """
    new_mean = (mean1*var2 + mean2*var1) / (var1 + var2)
    new_var = 1 / (1/var1 + 1/var2)
    return [new_mean, new_var]
```

### 2. Motion Update

In this phase, two mathematical concepts are taken into account:

  * Convolution
  * The [Total Probability](https://en.wikipedia.org/wiki/Law_of_total_probability)


When it comes to the motion update, the math is really straight forward.
Lets assume we are in position **x** with a Distribution probability with mean
**u1** and variance **s1**. We move in one direction with a Distribution probability
with mean **u_mov** and variance **s_mov** the new motion update will be a Gaussian
with:

  * `u = u1 + u_mov`
  * `s = s1 + s_mov`

  In python this math is coded as follows:

  ```python

  def predict(mean1, var1, mean2, var2):
      """ Given the mean and variance of two different Gaussian Distributions,
          predict your new mean and variance given the mean and variance of your
          prior belief and the mean and variance of your motion.
      """
      new_mean = mean1 + mean2
      new_var = var1 + var2
      return [new_mean, new_var]
  ```

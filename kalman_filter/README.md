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

### 1. Measurement Update (Prediction)

Let's say we know an object's current position and velocity , which we keep in
the *x* variable. Now one second has passed. We can predict where the object will
be one second later because we knew the object position and velocity one second
ago; we'll just assume the object kept going at the same velocity.

The equation (11) does these prediction calculations for us.

But maybe the object didn't maintain the exact same velocity. Maybe the object
changed direction, accelerated or decelerated. So when we predict the position
one second later, our uncertainty increases. Equation (12) represents this increase
in uncertainty.

Process noise refers to the uncertainty in the prediction step. We assume the
object travels at a constant velocity, but in reality, the object might accelerate
or decelerate. The notation *ν ∼ N(0,Q)* defines the process noise as a gaussian
distribution with mean zero and covariance Q.

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

Now we get some sensor information that tells where the object is relative to the
car. First we compare where we think we are with what the sensor data tells us:
Equation (13).

The *K* matrix, often called the **Kalman filter gain**, combines the uncertainty
of where we think we are *P'* with the uncertainty of our sensor measurement *R*.
If our sensor measurements are very uncertain (*R* is high relative to *P'*),
then the Kalman filter will give more weight to where we think we are: *x'*.
If where we think we are is uncertain (*P'* is high relative to *R*), the Kalman
filter will put more weight on the sensor measurement: *z*.

Measurement noise refers to uncertainty in sensor measurements. The notation
*ω ∼ N(0,R)* defines the measurement noise as a gaussian distribution with mean
zero and covariance *R*. Measurement noise comes from uncertainty in sensor
measurements.

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

### Put everything together

Let assume we have a set of **measurements** and **motions** and we want to compute
the belief the final position. The following snippet does the magic:

```python

measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

#Please print out ONLY the final values of the mean
#and the variance in a list [mu, sig].

# Insert code here
for idx in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[idx], measurement_sig)
    #print "update", [mu, sig]
    [mu, sig] = predict(mu, sig, motion[idx], motion_sig)
    #print "motion", [mu, sig]

print [mu, sig]
```

# Multi dimensions Kalman Filter Design

```python

def kalman_filter(x, P):
    for n in range(len(measurements)):

        # measurement update
        Z = matrix([[measurements[n]]])
        y = Z - H * x   # error calculation
        S = H * P * H.transpose() + R
        K = P * H.transpose() * S.inverse() # kalman gain

        # New state
        x = x + K * y
        P = (I - K*H) * P

        # prediction
        x = F * x + u
        P = F * P * F.transpose()

    return x,P

############################################
### use the code below to test your filter!
############################################

measurements = [1, 2, 3]

x = matrix([[0.], [0.]])               # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]])               # external motion
F = matrix([[1., 1.], [0, 1.]])        # next state function
H = matrix([[1., 0.]])                 # measurement function
R = matrix([[1.]])                     # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]])       # identity matrix

print kalman_filter(x, P)
```

# Kalman Filter Algorithm Map

![kalman filter algo map](imgs/kalman-map.png)

The Kalman Filter will receive data from two different sensors:

  * Lidar (cartesian coordinate system)
  * Radar (polar coordinate system)

**First measurement** - the filter will receive initial measurements of the object's
position relative to the car. These measurements will come from a radar or lidar sensor.

**Initialize state and covariance matrices** - the filter will initialize the object's
position based on the first measurement.

Another sensor measurement will come after a time period Δt.

**Predict** - the algorithm will predict where the object will be after time Δt.
One basic way to predict the object location after Δt is to assume the object's
velocity is constant; thus the object will have moved velocity * Δt.

**Update** - the filter compares the "predicted" location with what the sensor
measurement says. The predicted location and the measured location are combined to give
an updated location.

The car will receive another sensor measurement after a time period Δt. The algorithm
then does another predict and update step.

### Question

*What should a Kalman Filter do if both the radar and laser measurements arrive
at the same time, k+t? Hint: The Kalman filter algorithm predicts -> updates ->
predicts -> updates, etc. If two sensor measurements come in simultaneously, the
time step between the first measurement and the second measurement would be zero.*

Predict the state to k+t then use either one of the sensors to update. Then predict
the state to k+t again and update with the other sensor measurement.

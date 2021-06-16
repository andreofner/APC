""" Differentiable Kalman Filter"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

def plot_filtered(ssm,observations, targets, title = ""):
    """ plot filtered input"""
    predictions_mean = []
    for i in list(observations):
        x_pred = ssm.forward(i)
        predictions_mean.append(x_pred.numpy())
    plt.plot(np.asarray(observations), label="observation")
    plt.plot(np.squeeze(np.asarray(predictions_mean)), label="prediction")
    plt.plot(np.squeeze(np.asarray(targets)), label="target")
    plt.legend()
    plt.title(title)
    plt.show()

def init_weights(size1, size2):
    bound = tf.sqrt(1. / (size1 * size2))
    return tf.random.uniform([size1, size2], minval=-bound, maxval=bound)

def createKF(n=3, m=1):
    kf = KalmanFilter(n = n, m= m)
    return kf

class KalmanFilter(tf.Module):
    def __init__(self, m, n):
        """
        # sizes:
        m : int - measurement size
        n : int - state size
        # variables:
        x : float32 [n, 1]  - initial state
        z : float32 [m, 1] -  measurement
        # trainable variables:
        A : float32 [n, n] -  state transition matrix
        Q : float32 [n, n] -  process noise covariance
        R : float32 [m, m] -  measurement noise covariance
        H : float32 [m, n] -  measurement ingoing transition matrix
        H_out : float32 [m+n, m] -  measurement outgoing transition matrix
        """
        self._m = m
        self._n = n
        self._x = tf.Variable(init_weights(n, 1), dtype=tf.float32, name="x")
        self._A = tf.Variable(init_weights(n, n), dtype=tf.float32, name="A")
        self._P = tf.Variable(init_weights(n, n), dtype=tf.float32, name="P")
        self._Q = tf.Variable(init_weights(n, n), dtype=tf.float32, name="Q") # map from input data
        self._H = tf.Variable(init_weights(m, n), dtype=tf.float32, name="H")
        self._R = tf.Variable(init_weights(m, m), dtype=tf.float32,  name="R")
        self._H_out = tf.Variable(init_weights(n, m), dtype=tf.float32, name="H_out") # map towards output data
        self._in_gain = tf.Variable([1], dtype=tf.float32, name="input_gain")
        self.LML = 0.

    def predict(self):
        """ x_pred: predicted state p_pred: predicted error covariance"""
        x_pred = tf.matmul(self._A, self._x)
        p_pred = tf.matmul(self._A, tf.matmul(self._P, self._A, transpose_b=True)) + self._Q
        return x_pred, p_pred

    def predict_obs(self):
        """ Predict observation from current state"""
        return tf.matmul(tf.transpose(self._x), self._H_out) # todo this hsoul be H_ not H_out?

    def correct(self, z):
        """
        Inputs:
        z: observation

        Returns:
        K: Kalman gain
        x_corr: posterior state mean
        P_corr: posterior state variance
        """
        z = z*self._in_gain
        K = tf.matmul(self._P, tf.matmul(tf.transpose(self._H), tf.linalg.inv(tf.matmul(self._H, tf.matmul(self._P, self._H, transpose_b=True)) + self._R)))
        x_corr = self._x + tf.matmul(K, z - tf.matmul(self._H, self._x))
        P_corr = tf.matmul((1 - tf.matmul(K, self._H)), self._P)
        return K, x_corr, P_corr

    def forward(self, z):
        self._x, self._P = self.predict()
        K, self._x, self._P = self.correct(z)
        return self.predict_obs()

    def marginal_log_likelihood(self, z):
        """ Compute the marginal likelihood p(x[t] | x[:t-1]) for observation """
        y = z - tf.matmul(self._H, self._x)  # Innovation # todo dot or matmul?
        S = tf.matmul(tf.transpose(self._H), tf.linalg.inv(
            tf.matmul(self._H, tf.matmul(self._P, self._H, transpose_b=True)) + self._R))  # Innovation covariance
        d = np.asarray(tf.transpose(self._H_out) * 1 / self._R * z).shape[0]  # todo H_out or H?
        self.LML = self.LML - 0.5 * (tf.transpose(y) + (1 / S) * y + np.log(np.abs(S)) + d * np.log(2 * np.pi))
        return self.LML

amp, freq = 2, 6000
time = np.linspace(0, 100000, 100000)
signal1 = amp*np.sin(2*np.pi*freq*time)[:,np.newaxis]
signal2 = amp*sg.sawtooth(2*np.pi*freq*time, width=0.5)[:,np.newaxis]

inputs_ = np.array_split(signal1, 500, axis=0)
targets_ = np.array_split(signal2, 500, axis=0)
data = np.concatenate([inputs_, targets_], axis = 2)
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.repeat().shuffle(buffer_size=128, reshuffle_each_iteration=True).batch(512).as_numpy_iterator()

def loss(x, y):
  return tf.reduce_mean(tf.math.square(x-y), axis=[0,1])

def grad(model, observations, targets):
    with tf.GradientTape() as tape:
        loss_value = 0.
        for o, t in zip(list(observations), list(targets)):
            x_pred = model.forward(o)
            loss_value += loss(x_pred, t)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

ssm = createKF(n=3, m=1)
observations = np.asarray(inputs_)[0,:,0]
targets = np.asarray(targets_)[0,:,0]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False, epsilon=1e-08,
                                     beta_1=0.9, beta_2=0.999, decay=0.)
loss_list = []
for update in range(500):
    data = dataset.next()
    data_in = data[0,:,:1]
    data_target = data[0,:,1:2]
    loss_value, grads = grad(ssm, data_in, data_target)
    loss_list.append(loss_value)
    optimizer.apply_gradients(zip(grads, ssm.trainable_variables))
    if update % 10 == 0 and update > 1:
        print(loss_value.numpy())
        plot_filtered(ssm, observations, targets,  title=str("update: " + str(update) + " loss: " + str(loss_value.numpy())))

plt.plot(np.asarray(loss_list))
plt.show()

for v in list(ssm.trainable_variables):
    print("k ", v.name, " - ", v.numpy())



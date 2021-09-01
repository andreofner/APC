""" Minimal examples: Gradient based linear predictive coding and Kalman filtering"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.experimental.numpy import dot as tf_dot

order = 5
run_LPC = False
run_KF = False
run_KF_speech = False
run_PC_speech = False
run_HPC_speech = True

if run_LPC:
  """ Linear predictive coding (LPC) with gradients"""
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  A_1 = tf.Variable(initial_value=tf.random.uniform([order-1]), trainable=True, name="A_1")
  A_0 = tf.Variable(initial_value=[-1.], trainable=False, name="A_0")
  input = np.linspace(1, 100, 100)

  for update in range(100):
      with tf.GradientTape(persistent=True) as tape:
          A = tf.concat([A_0, A_1], axis=0)
          E = tf.TensorArray(dtype=tf.float32, size=input.shape[0], clear_after_read=False)
          order = A.shape[0]
          for n in range(input.shape[0]): E = E.write(n, 0.)
          for n in range(order, input.shape[0], 1):
              for i in range(order):
                  e_n = E.read(n) - tf.cast(A[i]*input[n-i], dtype=tf.float32)
                  E = E.write(n, e_n)
          E_mean = tf.reduce_mean(tf.math.square(E.stack()))
          grad = tape.gradient(E_mean, [A_1])
          optimizer.apply_gradients(zip(grad, [A_1]))
          if update % 10 == 0: print("Error:", E_mean.numpy(), " Updated A:", A.numpy())

if run_KF:
  """ Kalman filtering"""
  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.001)
  with tf.GradientTape(persistent=True) as tapeKF:
    input = np.linspace(1, 100, 100)
    xdim = 1
    for update in range(10):
      A = tf.Variable(initial_value=tf.random.uniform([order, order])*0.0001, trainable=True, name="A")
      x = tf.zeros([order])
      C = tf.eye(order)
      B = tf.ones([order])
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(input.shape[0]-1):
        u = input[n]
        y = input[n+1]
        x = tf_dot(A, x) + tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) + tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 1 == 0: print("Error:", E_mean.numpy())

  plt.plot(input, label="True Value")
  plt.plot(ohats, label="Kalman Filter")
  plt.legend()
  plt.show()

if run_KF_speech:
  """ Gradient based Kalman filtering of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 64
  hop_length = 32

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.01)
  with tf.GradientTape(persistent=True) as tapeKF:
    xdim = 1
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    for update in range(1000):
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(data.shape[0]-1):
        u = data[n,0]
        y = data[n,1]
        x = tf_dot(A, x) #+ tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) #+ tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 10 == 0:
        print("Error:", E_mean.numpy())
        plt.plot(data[:,1], label="True Value")
        plt.plot(ohats, label="Kalman Filter")
        plt.legend()
        plt.show()

if run_PC_speech:
  """ Gradient based predictive coding of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 64
  hop_length = 32

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  lr_A = 0.01
  lr_x = 0.01
  approximate_x = False
  updates_A = 1000
  updates_xhat = 50
  optimizer_A = tf.keras.optimizers.Adam(learning_rate=lr_A)
  optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)

  with tf.GradientTape(persistent=True) as tapeKF:
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    xhat = tf.Variable(initial_value=tf.zeros([order]), trainable=True)

  xdim = 1

  for update_A in range(updates_A):
    with tapeKF:
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      MSE_seq = 0.
    for n in range(data.shape[0] - 1):
      with tapeKF:
        y = data[n, 1]
      if approximate_x:
        for update_x in range(updates_xhat):
            with tapeKF:
              ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat)))) #+ tf_dot(B, u))
              ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
              MSE = ex + ey
            grad_x = tapeKF.gradient(MSE, [xhat])
            optimizer_x.apply_gradients(zip(grad_x, [xhat]))
      else:
        with tapeKF:
          x = tf_dot(A, x)
          xhat_proj = tf_dot(A, xhat)
          phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
          K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
          xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
          phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
          ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat))))  # + tf_dot(B, u))
          ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
          MSE = ex + ey
      with tapeKF:
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        MSE_seq = MSE_seq+MSE
    grad = tapeKF.gradient(MSE_seq, [A])
    optimizer_A.apply_gradients(zip(grad, [A]))
    if update_A % 10 == 0:
      print("MSE:", MSE)
      plt.plot(data[:, 1], label="True Value")
      plt.plot(ohats, label="Kalman Filter")
      plt.legend()
      plt.show()

""" Gradient based linear predictive coding and Kalman filtering"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.experimental.numpy import dot as tf_dot

order = 2
run_LPC = False
run_KF = False
run_KF_speech = False
run_PC_speech = False
run_HPC_speech = True

if run_LPC:
  """ Linear predictive coding (LPC) with gradients"""
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  A_1 = tf.Variable(initial_value=tf.random.uniform([order-1]), trainable=True, name="A_1")
  A_0 = tf.Variable(initial_value=[-1.], trainable=False, name="A_0")
  input = np.linspace(1, 100, 100)

  for update in range(100):
      with tf.GradientTape(persistent=True) as tape:
          A = tf.concat([A_0, A_1], axis=0)
          E = tf.TensorArray(dtype=tf.float32, size=input.shape[0], clear_after_read=False)
          order = A.shape[0]
          for n in range(input.shape[0]): E = E.write(n, 0.)
          for n in range(order, input.shape[0], 1):
              for i in range(order):
                  e_n = E.read(n) - tf.cast(A[i]*input[n-i], dtype=tf.float32)
                  E = E.write(n, e_n)
          E_mean = tf.reduce_mean(tf.math.square(E.stack()))
          grad = tape.gradient(E_mean, [A_1])
          optimizer.apply_gradients(zip(grad, [A_1]))
          if update % 10 == 0: print("Error:", E_mean.numpy(), " Updated A:", A.numpy())

if run_KF:
  """ Kalman filtering"""
  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.001)
  with tf.GradientTape(persistent=True) as tapeKF:
    input = np.linspace(1, 100, 100)
    xdim = 1
    for update in range(10):
      A = tf.Variable(initial_value=tf.random.uniform([order, order])*0.0001, trainable=True, name="A")
      x = tf.zeros([order])
      C = tf.eye(order)
      B = tf.ones([order])
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(input.shape[0]-1):
        u = input[n]
        y = input[n+1]
        x = tf_dot(A, x) + tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) + tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 1 == 0: print("Error:", E_mean.numpy())

  plt.plot(input, label="True Value")
  plt.plot(ohats, label="Kalman Filter")
  plt.legend()
  plt.show()

if run_KF_speech:
  """ Gradient based Kalman filtering of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 64
  hop_length = 32

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.01)
  with tf.GradientTape(persistent=True) as tapeKF:
    xdim = 1
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    for update in range(1000):
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(data.shape[0]-1):
        u = data[n,0]
        y = data[n,1]
        x = tf_dot(A, x) #+ tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) #+ tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 10 == 0:
        print("Error:", E_mean.numpy())
        plt.plot(data[:,1], label="True Value")
        plt.plot(ohats, label="Kalman Filter")
        plt.legend()
        plt.show()

if run_PC_speech:
  """ Gradient based predictive coding of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 64
  hop_length = 32

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  lr_A = 0.01
  lr_x = 0.01
  approximate_x = False
  updates_A = 1000
  updates_xhat = 50
  optimizer_A = tf.keras.optimizers.Adam(learning_rate=lr_A)
  optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)

  with tf.GradientTape(persistent=True) as tapeKF:
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    xhat = tf.Variable(initial_value=tf.zeros([order]), trainable=True)

  xdim = 1

  for update_A in range(updates_A):
    with tapeKF:
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      MSE_seq = 0.
    for n in range(data.shape[0] - 1):
      with tapeKF:
        y = data[n, 1]
      if approximate_x:
        for update_x in range(updates_xhat):
            with tapeKF:
              ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat)))) #+ tf_dot(B, u))
              ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
              MSE = ex + ey
            grad_x = tapeKF.gradient(ex, [xhat])
            optimizer_x.apply_gradients(zip(grad_x, [xhat]))
      else:
        with tapeKF:
          x = tf_dot(A, x)
          xhat_proj = tf_dot(A, xhat)
          phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
          K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
          xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
          phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
          ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat))))  # + tf_dot(B, u))
          ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
          MSE = ex + ey
      with tapeKF:
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        MSE_seq = MSE_seq+MSE
    grad = tapeKF.gradient(MSE_seq, [A])
    optimizer_A.apply_gradients(zip(grad, [A]))
    if update_A % 10 == 0:
      print("MSE:", MSE)
      plt.plot(data[:, 1], label="True Value")
      plt.plot(ohats, label="Kalman Filter")
      plt.legend()
      plt.show()


""" Gradient based linear predictive coding and Kalman filtering"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.experimental.numpy import dot as tf_dot

order = 2
run_LPC = False
run_KF = False
run_KF_speech = False
run_PC_speech = False
run_HPC_speech = True

if run_LPC:
  """ Linear predictive coding (LPC) with gradients"""
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  A_1 = tf.Variable(initial_value=tf.random.uniform([order-1]), trainable=True, name="A_1")
  A_0 = tf.Variable(initial_value=[-1.], trainable=False, name="A_0")
  input = np.linspace(1, 100, 100)

  for update in range(100):
      with tf.GradientTape(persistent=True) as tape:
          A = tf.concat([A_0, A_1], axis=0)
          E = tf.TensorArray(dtype=tf.float32, size=input.shape[0], clear_after_read=False)
          order = A.shape[0]
          for n in range(input.shape[0]): E = E.write(n, 0.)
          for n in range(order, input.shape[0], 1):
              for i in range(order):
                  e_n = E.read(n) - tf.cast(A[i]*input[n-i], dtype=tf.float32)
                  E = E.write(n, e_n)
          E_mean = tf.reduce_mean(tf.math.square(E.stack()))
          grad = tape.gradient(E_mean, [A_1])
          optimizer.apply_gradients(zip(grad, [A_1]))
          if update % 10 == 0: print("Error:", E_mean.numpy(), " Updated A:", A.numpy())

if run_KF:
  """ Kalman filtering"""
  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.001)
  with tf.GradientTape(persistent=True) as tapeKF:
    input = np.linspace(1, 100, 100)
    xdim = 1
    for update in range(10):
      A = tf.Variable(initial_value=tf.random.uniform([order, order])*0.0001, trainable=True, name="A")
      x = tf.zeros([order])
      C = tf.eye(order)
      B = tf.ones([order])
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(input.shape[0]-1):
        u = input[n]
        y = input[n+1]
        x = tf_dot(A, x) + tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) + tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 1 == 0: print("Error:", E_mean.numpy())

  plt.plot(input, label="True Value")
  plt.plot(ohats, label="Kalman Filter")
  plt.legend()
  plt.show()

if run_KF_speech:
  """ Gradient based Kalman filtering of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 64
  hop_length = 32

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  optimizerKF = tf.keras.optimizers.Adam(learning_rate=0.01)
  with tf.GradientTape(persistent=True) as tapeKF:
    xdim = 1
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    for update in range(1000):
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      xhat = tf.zeros([order])
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      E_mean = 0.
      for n in range(data.shape[0]-1):
        u = data[n,0]
        y = data[n,1]
        x = tf_dot(A, x) #+ tf_dot(B, u)
        xhat_proj = tf_dot(A, xhat) #+ tf_dot(B, u)
        phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
        K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
        xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
        phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        E_mean = E_mean + tf.reduce_mean(tf.math.square(y-ohat))
      grad = tapeKF.gradient(E_mean, [A])
      optimizerKF.apply_gradients(zip(grad, [A]))
      if update % 10 == 0:
        print("Error:", E_mean.numpy())
        plt.plot(data[:,1], label="True Value")
        plt.plot(ohats, label="Kalman Filter")
        plt.legend()
        plt.show()

if run_PC_speech:
  """ Gradient based predictive coding of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers

  frame_length = 16
  hop_length = 16

  sr, x = wavfile.read(pysptk.util.example_audio_file())
  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # F0 estimation and source excitation generation
  f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
  pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
  source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
  # windowed inputs
  frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames *= pysptk.blackman(frame_length)
  frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
  frames = np.expand_dims(frames, axis=-1)

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())
  data = data

  lr_A = 0.01
  lr_x = 0.01
  approximate_x = False
  updates_A = 1000
  updates_xhat = 50
  optimizer_A = tf.keras.optimizers.Adam(learning_rate=lr_A)
  optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)

  with tf.GradientTape(persistent=True) as tapeKF:
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    xhat = tf.Variable(initial_value=tf.zeros([order]), trainable=True)

  xdim = 1

  for update_A in range(updates_A):
    with tapeKF:
      x = tf.zeros([order])
      B = tf.cast([0., 0.1], dtype=tf.float32)
      H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
      phat = tf.eye(order)
      C = tf.eye(order)
      Q = tf.eye(order)
      R = tf.eye(order)
      ohats = []
      MSE_seq = 0.
    for n in range(data.shape[0] - 1):
      with tapeKF:
        y = data[n, 1]
      if approximate_x:
        for update_x in range(updates_xhat):
            with tapeKF:
              ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat)))) #+ tf_dot(B, u))
              ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
              MSE = ex + ey
            grad_x = tapeKF.gradient(MSE, [xhat])
            optimizer_x.apply_gradients(zip(grad_x, [xhat]))
      else:
        with tapeKF:
          x = tf_dot(A, x)
          xhat_proj = tf_dot(A, xhat)
          phat_proj = tf_dot(A, tf_dot(phat, tf.transpose(A))) + Q
          K = tf_dot(tf_dot(phat_proj, tf.transpose(C)), tf.linalg.inv(tf_dot(tf_dot(C, phat_proj), tf.transpose(C)) + R))
          xhat = xhat_proj + tf_dot(K, y - tf_dot(C, xhat_proj))
          phat = phat_proj - tf_dot(K, tf_dot(C, phat_proj))
          ex = tf.reduce_mean(tf.math.square(xhat - (tf_dot(A, xhat))))  # + tf_dot(B, u))
          ey = tf.reduce_mean(tf.math.square(y - tf_dot(C, xhat)))
          MSE = ex + ey
      with tapeKF:
        ohat = tf_dot(H, xhat)
        ohats.append(ohat.numpy())
        MSE_seq = MSE_seq+MSE
    grad = tapeKF.gradient(MSE_seq, [A])
    optimizer_A.apply_gradients(zip(grad, [A]))
    if update_A % 10 == 0:
      print("MSE:", MSE)
      plt.plot(data[:, 1], label="True Value")
      plt.plot(ohats, label="Kalman Filter")
      plt.legend()
      plt.show()

if run_HPC_speech:
  """ Hierarchical gradient based predictive coding of speech """
  import tensorflow as tf
  import os
  import matplotlib.pyplot as plt
  import simpleaudio as sa
  import pysptk
  from scipy.io import wavfile
  import soundfile as sf
  import librosa
  import librosa.display
  import numpy as np
  from scipy import signal as sg
  import librosa
  import librosa.display
  import matplotlib.pyplot as plt
  from scipy import signal
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import matplotlib.pyplot as plt
  import matplotlib.gridspec as gridspec

  frame_length = 128
  hop_length = 128

  #sr, x = wavfile.read(pysptk.util.example_audio_file())

  sr = 256
  wave_hz = 10
  duration = 1
  t = np.linspace(0, duration, sr, endpoint=False)
  x = signal.square(2 * np.pi * wave_hz * t)
  plt.plot(t, x)
  plt.show()

  #ime = np.arange(0, 10, 0.1)
  #amplitude = np.sin(time)
  #plot.plot(time, amplitude)
  #plt.show()

  x = x.astype(np.float32)
  x /= np.abs(x).max()
  train_target = x
  train_input = x
  fs = sr

  # windowed outputs
  frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
  #frames_se *= pysptk.blackman(frame_length)
  frames_se = np.expand_dims(frames_se, axis=-1)
  #frames_se *= 100
  frames_se += 0.5
  frames = frames_se

  # Dataset from frames
  frames_concat = np.concatenate([frames, frames_se], axis=-1)
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

  dataset = dataset.batch(1).as_numpy_iterator()
  batch_size_ = dataset.next().shape[0]
  dataset = tf.data.Dataset.from_tensor_slices(frames_concat)
  dataset = dataset.batch(1)
  dataset = dataset.as_numpy_iterator()
  data = np.squeeze(dataset.next())

  xdim = 1
  lr_A = 0.01 #0.001
  lr_x = 0.1 #0.1
  lr_F = 0.01 #0.1
  lr_G = 0.01 #0.1

  updates_A_TD = 20 #10
  updates_A = 1 #2
  updates_xhat = 5 #3

  alphas = np.linspace(0,1,updates_A_TD+1)
  alphas_A = np.linspace(0, 1, updates_A + 1)

  optimizer_F = tf.keras.optimizers.Adam(learning_rate=lr_F)
  optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
  optimizer_A = tf.keras.optimizers.Adam(learning_rate=lr_A)
  optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)

  loss_hierarchical = []
  loss_single = []

  plot_interval = 5
  f, axs = plt.subplots(ncols=2, nrows=3, constrained_layout=True)
  #int(updates_A_TD / plot_interval - 1)

  with tf.GradientTape(persistent=True) as tapeKF:
    A_TD = tf.Variable(initial_value=tf.zeros([order, order]) * 0.1, trainable=True, name="A_TD")
    F = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="F")
    G = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="G")
    A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
    B = tf.cast([0., 0.1], dtype=tf.float32)

    xhat_TD = tf.Variable(initial_value=tf.zeros([order]), trainable=True, name="xhat_TD")
    xhat = tf.Variable(initial_value=tf.zeros([order]), trainable=True, name="xhat")

  plt_col = -1
  f, axs = plt.subplots(ncols=2, nrows=3, constrained_layout=True)
  for use_top_down in [True, False]:
    plt_col += 1
    plt_row = -1
    states_l1 = np.zeros([data.shape[0], order])
    for update_A_TD in range(updates_A_TD):
      for update_A in range(updates_A):
        with tapeKF:
          tapeKF.reset()
          A = tf.Variable(initial_value=tf.random.uniform([order, order]) * 0.1, trainable=True, name="A")
          xhat = tf.Variable(initial_value=tf.zeros([order]), trainable=True, name="xhat")
          optimizer_A = tf.keras.optimizers.Adam(learning_rate=lr_A)
          optimizer_x = tf.keras.optimizers.Adam(learning_rate=lr_x)
          optimizer_F = tf.keras.optimizers.Adam(learning_rate=lr_F)
          optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
        with tapeKF:
          H = tf.Variable(initial_value=tf.concat([[1.], tf.zeros(order - 1)], axis=0), trainable=False)
          phat = tf.random.uniform([order, order]) * 1.
          C = tf.eye(order)
          ohats = []
          ohats_posterior = []
          MSE_seq = 0.
          if not use_top_down:
            states_l1 = np.zeros_like(states_l1)
        for n in range(data.shape[0] - 1):
          y = data[n, 1]
          xhat_TD = tf.cast(states_l1[n], dtype=tf.float32)
          ohats.append(tf_dot(H, tf_dot(F, xhat) + tf_dot(G, xhat_TD)).numpy())  # prediction from prior
          for update_x in range(updates_xhat):
              with tapeKF:
                #xhat_both = tf_dot(F, xhat) + tf_dot(G, xhat_TD) #+ tf_dot(B, u)
                xhat_both = tf_dot(F, xhat) + tf_dot(G, xhat_TD)  # + tf_dot(B, u)
                ex = tf.reduce_mean(tf.math.square(tf_dot(phat, (xhat_both - (tf_dot(A, xhat_both))))))
                ey = tf.reduce_mean(tf.math.square(y - tf_dot(H, xhat_both)))
                MSE = ex+ey
                if use_top_down:
                  loss_hierarchical.append(MSE.numpy())
                else:
                  loss_single.append(MSE.numpy())
              optimizer_F.apply_gradients(zip(tapeKF.gradient(MSE, [F]), [F]))
              optimizer_G.apply_gradients(zip(tapeKF.gradient(MSE, [G]), [G]))
              optimizer_x.apply_gradients(zip(tapeKF.gradient(MSE, [xhat]), [xhat])) # x not MSE
              optimizer_A.apply_gradients(zip(tapeKF.gradient(MSE, [A]), [A]))
          with tapeKF:
            MSE_seq = MSE_seq+MSE
          ohats_posterior.append(tf_dot(H, tf_dot(F, xhat) + tf_dot(G, xhat_TD)).numpy())  # prediction from posterior
          states_l1[n] = xhat.numpy()
        xhat_TD = tf.identity(xhat)

      if update_A_TD % plot_interval == 0:
        if not update_A_TD == 0:
          plt_row += 1
          info = "Update A_TD:"+str(update_A_TD)+" Update A:"+str(update_A)+" MSE:"+str(MSE.numpy())
          print(info)
          if update_A_TD == updates_A_TD-1:
            axs[plt_row, plt_col].plot(data[:, 1], label="Target signal", color="black")
            axs[plt_row, plt_col].plot(ohats, label="Prior prediction", color="blue")
            axs[plt_row, plt_col].plot(ohats_posterior, label="Posterior prediction", color="green")
          else:
            axs[plt_row, plt_col].plot(data[:, 1], color="black")
            axs[plt_row, plt_col].plot(ohats, color="blue")
            axs[plt_row, plt_col].plot(ohats_posterior, label="Posterior prediction", color="green")
          axs[plt_row, plt_col].grid()
          #axs[plt_row, plt_col].scatter(x=[0], y=[ohats[0]], c='b')
          #axs[plt_row, plt_col].scatter(x=[0], y=[ohats_posterior[0]], c='g')
          axs[plt_row, plt_col].set_ylim([-1, 2.])
          #axs[plt_row, plt_col].set_xlabel("Time (samples)")
          #axs[plt_row, plt_col].set_ylabel("Amplitude")
          axs[plt_row, plt_col].title.set_text('Update ' + str(update_A_TD))
    axs[-1, 0].set_xlabel("Time (samples)")
    axs[-1, 1].set_xlabel("Time (samples)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[1, 0].set_ylabel("Amplitude")
    axs[2, 0].set_ylabel("Amplitude")

    """
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend(loc='lower left')
    plt.ylim((-0.1, 1.2))
    if use_top_down:
      #plt.title("Hierarchical predictive coding model")
      plt.tight_layout()
      f.savefig("./hierarchical.pdf")
      plt.show()
    else:
      #plt.title("Single layer predictive coding model")
      plt.tight_layout()
      f.savefig("./singlelayer.pdf")
      plt.show()
    """
  f.savefig("./both_"+str(updates_xhat)+".pdf")
  plt.show()

  f = plt.figure()
  plt.plot(loss_hierarchical)
  plt.xlabel("Update")
  plt.ylabel("Sensory prediction error")
  #plt.title("Hierarchical predictive coding model")
  plt.grid()
  plt.tight_layout()
  f.savefig("./hierarchical_posterior.pdf")
  plt.show()

  f = plt.figure()
  plt.plot(loss_single)
  plt.xlabel("Update")
  plt.ylabel("Sensory prediction error")
  #plt.title("Single layer predictive coding model")
  plt.grid()
  plt.tight_layout()
  f.savefig("./singlelayer_posterior.pdf")
  plt.show()


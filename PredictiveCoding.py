""" Hierarchical gradient based predictive coding network"""
import sys
sys.path.append('APC')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PLP_model.PDDSP_spectral_ops import *
import tensorflow.experimental.numpy as tnp
import simpleaudio as sa
import pysptk
from scipy.io import wavfile
import soundfile as sf
from tqdm import tqdm
import librosa
import librosa.display
from pysptk.synthesis import AllPoleDF
from pysptk.synthesis import MLSADF, Synthesizer

from IIR import *

def play(audio, fs):
    audio = audio * (2 ** 15 - 1) / np.max(np.abs(audio))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def FFT(data_in, win_length=1024, hop_length=128, center=True):
    return stft(np.squeeze(data_in), win_length, frame_step=hop_length,
                                            fft_length=win_length, pad_end=False,
                                            center=center, window_fn=tf.signal.hann_window)

def IFFT(FFT_in, win_length=1024, hop_length=128, center=False):
    return inverse_stft(FFT_in, win_length, hop_length, fft_length=win_length, center=center,
       window_fn=tf.signal.inverse_stft_window_fn(hop_length, forward_window_fn=tf.signal.hann_window))

class predictive_coding_layer(tf.Module):
    """ Predictive coding layer with local optimization and top down input
    Sizes:
    m: int - state size
    n : int - observation size
    c : int - control input size

    Inputs:
    prediction_in: top-down state prediction
    control_in: bottom-up control input
    observation_in: bottom-up observation

    Computes:
    self.x_hat : Inferred latent state mean
    self.p_hat : Inferred latent state variance
    self.A : transition weights
    self.B : control input weights
    self.C : observation weights
    self.ex : Latent prediction error
    self.ey : Outgoing prediction error
    self.prediction_error : Total prediction error

    Remarks:
    For prediction or autoencoding tasks, simply feed the same
    value for both control and observation
    """
    def __init__(self, batch_size=1, m=3, n=1, c=1,  learning_rate=0.001, tape=None):
        bs = batch_size
        # layer variables
        self.A = tf.Variable(self.init_weights(bs, m, m), dtype=tf.float32, name="A") # transition weights
        self.B = tf.Variable(self.init_weights(bs, m, c), dtype=tf.float32, name="B") # control input weights
        self.C = tf.Variable(self.init_weights(bs, n, m), dtype=tf.float32, name="C") # observation weights
        self.x_hat = tf.Variable(self.init_weights(bs, m, 1), dtype=tf.float32, name="x_hat") # inferred state # todo sizes
        self.p_hat = tf.Variable([tf.experimental.numpy.identity(m, dtype=tf.float32)  for b in range(bs)], dtype=tf.float32, name="p_hat") # inferred state covariance
        self.tape = tape

        # optimizers for trainable variables
        self.optimizer_A = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_B = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_C = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer_x = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # prediction errors
        self.ex = 0.
        self.ey = 0.
        self.grads_x = 0.
        self.grads_A = 0.
        self.grads_B = 0.
        self.grads_C = 0.
        self.prediction_error = 0.

    def init_weights(self, batch_size, size1, size2):
        bound = tf.sqrt(1. / (size1 * size2))
        init = tf.random.uniform([size1, size2], minval=-bound, maxval=bound)
        return [init for b in range(batch_size)]

    def forward(self, residual_in, layer_updates=10):
        self.x_hat = (tf.matmul(self.A, self.x_hat))
        self.o_hat = (tf.matmul(self.C, self.x_hat)) + residual_in

    def update(self, control_in, observation_in, layer_updates=10):
        """ Indendently update layer weights and the inferred latent state """
        for i in range(layer_updates):
            # compute prediction errors
            ex = tf.math.square(self.x_hat - tf.matmul(self.A, self.x_hat) - tf.matmul(self.B, control_in))
            ey = tf.math.square(observation_in - tf.matmul(self.C, self.x_hat))
            prediction_error = tf.reduce_sum(ey, axis=1, keepdims=True) + tf.reduce_sum(ex, axis=1, keepdims=True)
            # Optimise state
            dldmu = tf.reduce_sum(ex, axis=2, keepdims=True)-tf.reduce_sum(ey, axis=2, keepdims=True)
            self.x_hat = self.x_hat - (0.05 * dldmu)

            self.grads_A = self.tape.gradient(ex, [self.A])
            self.grads_B = self.tape.gradient(ex, [self.B])
            self.grads_C = self.tape.gradient(ey, [self.C])
            self.optimizer_A.apply_gradients(zip(self.grads_A, [self.A]))
            self.optimizer_B.apply_gradients(zip(self.grads_B, [self.B]))
            self.optimizer_C.apply_gradients(zip(self.grads_C, [self.C]))

        self.ex = ex
        self.ey = ey
        self.prediction_error = prediction_error
        self.o_hat = tf.matmul(self.C, self.x_hat)


def to_FFT(data_in, win_length=128, hop_length=1, center=True):
    data_in_FFT = FFT(data_in, win_length=win_length, hop_length=hop_length, center=center)
    data_in_FFT_split = tf.concat([tnp.complex64(data_in_FFT).real, tnp.complex64(data_in_FFT).imag], axis=2)
    return tf.transpose(data_in_FFT_split, [0, 2, 1])

def to_FFT_complex(data_in, win_length=128, hop_length=1, center=False):
    data_in_FFT = FFT(data_in, win_length=win_length, hop_length=hop_length, center=center)
    return data_in_FFT

def to_FFT_magnitude(data_in, win_length=128, hop_length=1, center=False):
    data_in_FFT = FFT(data_in, win_length=win_length, hop_length=hop_length, center=center)
    return tf.transpose(tf.abs(data_in_FFT), [0, 2, 1])

def to_IFFT(FFT_in, bins, win_length=128, hop_length=1, plot=True):
    data = tf.dtypes.complex(FFT_in[:, :bins], FFT_in[:, bins:])
    data = IFFT(data, win_length=win_length, hop_length=hop_length)
    if plot:
        plt.plot(data)
        plt.title("IFFT")
        plt.show()
    return data

def predictive_coding_filter(batch,
                             n_sequences = 1,
                             use_FFT = False,
                             win_length = 128,
                             hop_length = 64,
                             ndim = 1,
                             layer_updates = 10,
                             log_details=True):
    """ Filter batched sequences in provided dataset with a predictive coding network.
    Optionall applies FFT transform to inputs first. """
    #todo select amount of units per FFT bin

    coeffs_list_A = []
    coeffs_list_B = []
    coeffs_list_C = []
    residual_x = []
    residual_y = []
    predaudio = None

    with tf.GradientTape(persistent=True) as tape:
        PCL_1 = None
        data = tf.cast(batch, dtype=tf.float32)

        if use_FFT:
            data_in = to_FFT_magnitude(data, win_length=win_length, hop_length=hop_length, center=False) # center True if not novelty
            tf.print("data in", data_in.shape)
            data_target = data_in
            if PCL_1 is None:
                PCL_1 = predictive_coding_layer(batch_size=data_in.shape[0], m=data_in.shape[1], n=data_in.shape[1],
                                            c=data_target.shape[1], tape=tape)
        else:
            data_in = data
            data_target = data_in
            print("AUDIO shape: ", data_in.shape)
            if PCL_1 is None:
                PCL_1 = predictive_coding_layer(batch_size=data_in.shape[0], m=data_in.shape[1] * ndim, n=data_in.shape[1],
                                            c=data_target.shape[1], tape=tape)

        x_hats = []
        predicted_obs = []
        tf.print("data_in", data_in.shape)
        for i in tqdm(range(data_in.shape[2])):
            # first layer predicts changes between observed sensory states
            if i == 0:
                PCL_1.update(layer_updates=layer_updates, control_in=np.zeros_like(data_in[:,:,i-1:i]), observation_in=data_target[:,:, i:i+1])
            else:
                PCL_1.update(layer_updates=layer_updates, control_in=data_in[:,:,i-1:i], observation_in=data_target[:,:, i:i+1])

            # Collect layer 1 predictions
            if log_details: x_hats.append(PCL_1.x_hat.numpy())
            if log_details: predicted_obs.append(PCL_1.o_hat.numpy())

            # collect residual signal = prediction error
            if i > 0:
                if log_details: residual_x.append(PCL_1.ex.numpy())
                residual_y.append(PCL_1.ey.numpy())

        # transform back from FFT bins to continuous audio
        if log_details:
            predicted_o = np.squeeze(predicted_obs)
            if False: # todo fix IFFT
                predicted_o = to_IFFT(tf.transpose(predicted_o, [0,2,1]), bins=int(predicted_o.shape[2] / 2),
                                 win_length=win_length, hop_length=hop_length, plot=False)
                predaudio = predicted_o.numpy()
            else:
                predaudio = predicted_o

            coeffs_list_A.append(PCL_1.A)
            coeffs_list_B.append(PCL_1.B)
            coeffs_list_C.append(PCL_1.C)

    return coeffs_list_A, coeffs_list_B, coeffs_list_C, residual_x, residual_y, predaudio

if False:
    n_dim = 25
    layer_updates = 10
    frame_length = 256
    hop_length = 128
    d_batch_size = 2000

    # generate input and compute frames
    sr_, x = wavfile.read(pysptk.util.example_audio_file())
    x = x.astype(np.float32)
    sr = 16000
    x = librosa.resample(x, sr_, sr)
    #x /= np.max(x)

    # F0 estimation and source excitation generation
    f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
    pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
    source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]

    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames = np.expand_dims(frames, axis=-1)
    frames = np.transpose(frames, [0,2,1])
    dataset = tf.data.Dataset.from_tensor_slices(frames)
    dataset = dataset.batch(d_batch_size)
    dataset_len = len(list(dataset)) # get number of batches
    dataset = dataset.as_numpy_iterator()

    CA, CB, CC, EX, EY, pred = predictive_coding_filter(dataset, n_sequences=dataset_len,
                             use_FFT=False, win_length=frame_length, hop_length=hop_length, ndim=n_dim,
                                                        layer_updates=layer_updates)
    #pred /= np.max(pred)

    # play input
    plt.plot(x)
    plt.title("Input audio")
    plt.show()
    play(x, fs=sr)

    # play prediction
    pred_concat = tf.signal.overlap_and_add(pred.T, frame_step=hop_length).numpy()
    #pred_concat /= np.max(pred_concat)
    plt.plot(pred_concat)
    plt.title("Predicted audio")
    plt.show()
    play(pred_concat, fs=sr)

    # play residual
    EY_concat = tf.signal.overlap_and_add(np.squeeze(EY).T, frame_step=hop_length).numpy()
    #EY_concat /= np.max(EY_concat)
    plt.plot(EY_concat)
    plt.title("Residual")
    plt.show()
    play(EY_concat, fs=sr)


    # Create dataset from input excitation
    EYframes = librosa.util.frame(EY_concat, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    EYframes = np.asarray(EYframes, dtype=np.float32)
    EYframes /= np.max(EYframes)
    zeros = np.zeros([1, frame_length])
    EYframes = np.concatenate([EYframes, zeros], axis=0)
    EYframes = np.expand_dims(EYframes, axis =-1)
    EYframes = np.transpose(EYframes, [0,2,1])
    dataset = tf.data.Dataset.from_tensor_slices(EYframes)
    dataset = dataset.batch(d_batch_size)
    dataset_len = len(list(dataset)) # get number of batches
    dataset = dataset.as_numpy_iterator()

    def reconstruct(dataset,n_sequences = 1,  ndim = 1, layer_updates = 20, CA=None, CB=None, CC=None):
        with tf.GradientTape(persistent=True) as tape:
            for sequence in range(n_sequences):
                data_in = tf.cast(dataset.next(), dtype=tf.float32)
                data_target = data_in
                print("AUDIO shape: ", data_in.shape)
                PCL_1 = predictive_coding_layer(batch_size=data_in.shape[0], m=data_in.shape[1] * ndim, n=data_in.shape[1],
                                                c=data_target.shape[1], tape=tape)
                PCL_1.A = CA[0]
                PCL_1.B = CB[0]
                PCL_1.C = CC[0]
                predicted_obs = []
                for i in tqdm(range(data_in.shape[2])):
                    PCL_1.forward(residual_in=data_in[:, :, i:i+1])
                    # Collect layer 1 predictions
                    predicted_obs.append(PCL_1.o_hat)
                predaudio = np.squeeze(predicted_obs)
        return predaudio

    reconstruction = reconstruct(dataset=dataset,
                                 n_sequences=dataset_len,
                                 ndim=n_dim,
                                 layer_updates=layer_updates,
                                 CA=CA, CB=CB, CC=CC)

    rec_concat = tf.signal.overlap_and_add(reconstruction.T, frame_step=hop_length).numpy()
    rec_concat /= np.max(rec_concat)

    plt.plot(rec_concat)
    plt.show()

    play(rec_concat, fs=sr)
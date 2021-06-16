""" Differentiable IIR filter as gradient based predictive coding network"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from tqdm import tqdm
import copy

def concatenate_prediction(data_target, prediction=None, hop_length=None):
    if prediction is not None:
        prediction = prediction[:, :, :, 0]
        residual = data_target - prediction
        residual_concat = tf.signal.overlap_and_add(np.squeeze(residual), frame_step=hop_length).numpy()
        pred_concat = tf.signal.overlap_and_add(np.squeeze(prediction), frame_step=hop_length).numpy()
        return pred_concat, residual_concat
    else:
        return tf.signal.overlap_and_add(np.squeeze(data_target), frame_step=hop_length).numpy()

def plot_waves(target_concat, pred_concat, residual_concat):
    plt.plot(target_concat, label="Target")
    plt.plot(pred_concat, label="Prediction")
    plt.plot(residual_concat, label="Residual")
    plt.legend()
    plt.show()

def plot_spectrogram(train_input, train_target, pred_concat, residual_concat):
    def spec_plot(audio, title, ax=None):
        D = np.abs(librosa.stft(audio))**2
        S = librosa.feature.melspectrogram(S=D, sr=fs)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time',
                                 y_axis='mel', sr=fs,
                                 fmax=8000, ax=ax)
        ax.set_title(title)
    fig, axs = plt.subplots(2, 2)
    spec_plot(train_input, "Input", axs[0,0])
    spec_plot(train_target, "Target", axs[1,0])
    spec_plot(pred_concat, "Prediction", axs[0, 1])
    spec_plot(residual_concat, "Residual", axs[1, 1])
    plt.tight_layout()
    plt.show()

def play(audio, fs):
    audio = audio * (2 ** 15 - 1) / np.max(np.abs(audio))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

class dense_layer(tf.Module):
    """ Dense layer without weights sharing"""
    def __init__(self, batch_size=1, hidden_dim=256):
        bs = batch_size
        self.in_layer = tf.Variable(self.init_weights(bs, 1, hidden_dim), dtype=tf.float32)
        self.hidden1 = tf.Variable(self.init_weights(bs, hidden_dim, hidden_dim), dtype=tf.float32)
        self.hidden2 = tf.Variable(self.init_weights(bs, hidden_dim, hidden_dim), dtype=tf.float32)
        self.out_layer = tf.Variable(self.init_weights(bs, hidden_dim, 1), dtype=tf.float32)
    def init_weights(self, batch_size, size1, size2):
        bound = tf.sqrt(1. / (size1 * size2))
        init = tf.random.uniform([size1, size2], minval=-bound, maxval=bound)
        return [init for b in range(batch_size)]
    def forward(self, input):
        input = input[...,0]
        input = tf.matmul(input, self.in_layer)
        input = tf.matmul(input, self.hidden1)
        input = tf.matmul(input, self.hidden2)
        output = tf.matmul(input, self.out_layer)
        return output

class WH_filter(tf.Module):
    def __init__(self, num_states=2, batch_size=512):
        super(WH_filter, self).__init__()
        self.ssm_in = LinearStateSpaceModel(num_states=num_states, bs=batch_size)
        self.model = dense_layer(batch_size=batch_size, hidden_dim=64)
        self.ssm_out = LinearStateSpaceModel(num_states=num_states, bs=batch_size)
    def forward(self, input, initial_states=None):
        encoded_input = self.ssm_in.forward(input)
        nonlinear_state = self.model.forward(encoded_input)
        decoded_input = self.ssm_out.forward(nonlinear_state)
        return decoded_input
    def synthesize(self, input):
        return self.ssm_out.synthesize(self.model.forward(self.ssm_in.synthesize(input)))

class LinearStateSpaceModel(tf.Module):
    def __init__(self, num_states=2, bs=512):
        super(LinearStateSpaceModel, self).__init__()
        bound = 1.0/(num_states+1)
        self.num_states = num_states
        self.pre_gain = tf.Variable(initial_value=tf.random.uniform([bs, 1, 1, 1],
                        minval=1., maxval=1.), name='pre_gain', trainable=False) #todo
        self.state_and_input_to_output_layer = tf.Variable(
            tf.random.uniform([bs, 1, num_states+1, 1], minval=-bound, maxval=bound), name='state_and_input_to_output_layer') # todo num_states+1?
        self.cell = LinearStateSpaceCell(num_states, batch_size=bs)
        self.ex = tf.Variable(0., trainable=True)
        self.ey = tf.Variable(0., trainable=True)
        self.prediction_error = tf.Variable(0., trainable=True)
        self.hidden = tf.Variable(tf.zeros((bs, 1, self.num_states)), trainable=True, name="hidden")
        self.prediction = 0.

    def synthesize(self, input, initial_states=None):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        self.hidden = tf.zeros((batch_size, 1, self.num_states))
        prev_pred = tf.zeros_like(input[:,0:1])
        states_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        for i in range(sequence_length - 1):
            prev_obs = (prev_pred + input[:,i:i+1]) * self.pre_gain
            self.hidden = self.cell.forward(prev_obs[:, 0], self.hidden)
            states_sequence = states_sequence.write(i+1, self.hidden[:,:])
        states_sequence = states_sequence.stack()
        tf.print("states_sequence", states_sequence.shape)
        states_sequence = tf.transpose(states_sequence, [1, 0, 2, 3])  # todo check transpose
        concat_out = tf.concat([input, states_sequence], axis=-1)
        predicted_output_sequence = tf.matmul(concat_out, self.state_and_input_to_output_layer)
        return predicted_output_sequence

    def forward_IIR(self, input, initial_states=None):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        self.hidden = tf.zeros((batch_size, 1, self.num_states))
        input = input*self.pre_gain
        states_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        for i in range(sequence_length - 1):
            self.hidden = self.cell.forward(input[:, i], self.hidden)
            states_sequence = states_sequence.write(i+1, self.hidden[:,:])
        states_sequence = states_sequence.stack()
        states_sequence = tf.transpose(states_sequence, [1, 0, 2, 3])  # todo check transpose
        concat_out = tf.concat([input, states_sequence], axis=-1)
        tf.print("concat_out.shape", concat_out.shape)
        predicted_output_sequence = tf.matmul(concat_out, self.state_and_input_to_output_layer)
        return predicted_output_sequence

    def forward(self, input, tape, updates=1):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        input = tf.cast(tf.expand_dims(input, axis=-1), dtype=tf.float32)
        input = input*self.pre_gain
        states_sequence = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        for i in tqdm(range(sequence_length - 1)):
            for update in range(updates):
                self.encoded = tf.concat([input[:, i], self.hidden], axis=-1)
                self.prediction = tf.matmul(self.encoded[:, :, tf.newaxis, :], self.state_and_input_to_output_layer)
                self.ey = tf.math.square(input[:, i] - self.prediction)
                self.ex = tf.math.square(self.hidden - self.cell.forward(input[:, i], self.hidden) - tf.matmul(input[:, i], self.cell.input_to_state_layer))
                self.prediction_error = tf.reduce_sum(self.ey, axis=1, keepdims=True) + tf.reduce_sum(self.ex, axis=2, keepdims=True)
                self.grads_x = tape.gradient(self.prediction_error, [self.hidden])
                self.grads_A = tape.gradient(self.prediction_error, [self.cell.state_to_state_layer])
                self.grads_B = tape.gradient(self.prediction_error, [self.cell.input_to_state_layer])
                self.grads_C = tape.gradient(self.prediction_error, [self.state_and_input_to_output_layer])
                optimizer_x.apply_gradients(zip(self.grads_x, [self.hidden]))
                optimizer_A.apply_gradients(zip(self.grads_A, [self.cell.state_to_state_layer]))
                optimizer_B.apply_gradients(zip(self.grads_B, [self.cell.input_to_state_layer]))
                optimizer_C.apply_gradients(zip(self.grads_C, [self.state_and_input_to_output_layer]))
            states_sequence = states_sequence.write(i+1, self.hidden[:,:])
        states_sequence = states_sequence.stack()
        states_sequence = tf.transpose(states_sequence, [1, 0, 2, 3])  # todo check transpose
        concat_out = tf.concat([input, states_sequence], axis=-1)
        tf.print("concat_out.shape", concat_out.shape)
        predicted_output_sequence = tf.matmul(concat_out, self.state_and_input_to_output_layer)
        return predicted_output_sequence

class LinearStateSpaceCell(tf.Module):
  def __init__(self, num_states=2, batch_size=512, name=None):
    super(LinearStateSpaceCell, self).__init__(name=name)
    bound = 1.0 / (num_states)
    self.num_states = num_states
    self.state_to_state_layer = tf.Variable(
      tf.random.uniform([batch_size, num_states, num_states], minval=-bound, maxval=bound), name='state_to_state_layer')
    self.input_to_state_layer = tf.Variable(
        tf.random.uniform([batch_size, 1, num_states], minval=-1, maxval=1), name='input_to_state_layer')
  def forward(self, input, in_states):
      state_output = tf.matmul(in_states, self.state_to_state_layer) + tf.matmul(input, self.input_to_state_layer)
      return state_output

def loss(model, x, y, tape):
    return tf.math.square(model(x, tape)[...,0]-y)

def grad(model, inputs, targets, tape):
    loss_value = loss(model.forward, inputs, targets, tape)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

if __name__ == '__main__':
    """Testing IIR analysis and synthesis"""
    n_dim = 5
    frame_length = 256
    hop_length = 128
    d_batch_size = 1024
    speech = True # speech learns independent filter coefficients per block / batch element
    wiener_hammerstein = False

    if speech:
        # generate input and compute frames
        sr, x = wavfile.read(pysptk.util.example_audio_file())
        x = x.astype(np.float32)
        x /= np.abs(x).max()
        train_target = x
        train_input = x
        fs = sr
    else:
        fs = 22050
        f0 = 20
        f1 = 20e3
        t = np.linspace(0, 60, int(60*fs))
        sr = fs
        train_input = signal.chirp(t=t, f0=f0, t1=60, f1=f1, method='logarithmic') + np.random.normal(scale=5e-2, size=len(t))
        fc = 2e3
        sos = signal.butter(N=2, Wn=fc/fs, output='sos')
        train_target = signal.sosfilt(sos, train_input)

    if speech: # source excitation as input
        # F0 estimation and source excitation generation
        f0 = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="f0")
        pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=240, otype="pitch")
        source_excitation = pysptk.excite(pitch, hop_length)[:x.shape[0]]
        source_excitation /= np.abs(source_excitation).max()
        # windowed inputs
        frames = librosa.util.frame(source_excitation, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
        frames *= pysptk.blackman(frame_length)
        frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
        frames /= np.abs(frames).max()
        frames = np.expand_dims(frames, axis=-1)
    else:
        # windowed inputs
        frames = librosa.util.frame(train_input, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
        frames *= pysptk.blackman(frame_length)
        frames /= np.abs(frames).max()
        frames = np.expand_dims(frames, axis=-1)

    # windowed outputs
    frames_se = librosa.util.frame(train_target, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames_se *= pysptk.blackman(frame_length)
    frames_se /= np.abs(frames_se).max()
    frames_se = np.expand_dims(frames_se, axis=-1)

    # Dataset from frames
    frames_concat = np.concatenate([frames, frames_se], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(frames_concat)

    if speech:
        dataset = dataset.batch(d_batch_size).as_numpy_iterator()
        batch_size_ = dataset.next().shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(frames_concat) # # todo improve
        dataset = dataset.batch(d_batch_size)
        dataset = dataset.as_numpy_iterator()
    else:
        dataset = dataset.batch(d_batch_size, drop_remainder=True).as_numpy_iterator()
        batch_size_ = dataset.next().shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(frames_concat) # todo improve
        dataset = dataset.batch(d_batch_size, drop_remainder=True)
        dataset = dataset.repeat().shuffle(buffer_size=512).as_numpy_iterator()

    with tf.GradientTape(persistent=True) as tape:
        if wiener_hammerstein:
            ssm = WH_filter(num_states=n_dim, batch_size=batch_size_)
        else:
            ssm = LinearStateSpaceModel(num_states=n_dim, bs=batch_size_)
        tape.watch(ssm.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer_x = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer_A = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer_B = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer_C = tf.keras.optimizers.Adam(learning_rate=0.001)

        loss_list = []
        if speech:
            data = dataset.next()

        if True:
            for update in range(1000):
                if not speech:
                    data = dataset.next()
                    data_in = data[:, :, 0:1]
                    data_target = data[:, :, 1:2]
                else:
                    data_in = data[:, :-1, 1:2]
                    data_target = data[:, 1:, 1:2]
                    # synthesize from external excitation signal
                    #data_in = data[:, :, 0:1]
                    #data_target = data[:, :, 1:2]
                #loss_value, grads = grad(ssm, data_in, data_target, tape)
                #loss_list.append(np.mean(loss_value.numpy()))
                prediction = ssm.forward(data_in, tape, updates=5)
                loss_value = np.mean(ssm.prediction_error.numpy())
                loss_list.append(loss_value)
                #optimizer.apply_gradients(zip(grads, ssm.trainable_variables))
                if update % 1 == 0:
                    MSE = loss_value
                    print("Update: ", update, "Loss: ", MSE)
                if update % 100 == 0:
                    index = np.random.randint(0,batch_size_)
                    residual = np.squeeze(data_target[index]) - np.squeeze(prediction[index])
                    plt.plot(np.squeeze(data_in[index]), label="Input")
                    plt.plot(np.squeeze(prediction[index]), label="Prediction")
                    plt.plot(np.squeeze(data_target[index]), label="Target")
                    plt.plot(np.squeeze(residual), alpha=0.3, label="Residual")
                    plt.title("Update: " + str(update) + " Loss " + str(MSE))
                    plt.legend()
                    plt.show()
                if update % 1 == 0:
                    pred_concat, residual_concat = concatenate_prediction(data_target, prediction, hop_length=hop_length)
                    input_concat = concatenate_prediction(data_in, prediction=None, hop_length=hop_length)
                    target_concat = concatenate_prediction(data_target, prediction=None, hop_length=hop_length)
                    plot_spectrogram(input_concat, target_concat, pred_concat, residual_concat)
                    if False:
                        synth_ssm = copy.deepcopy(ssm) # copy to keep vars trainable
                        synthesized = synth_ssm.synthesize(data[:, :, 0:1])
                        synthesized_concat = concatenate_prediction(synthesized, prediction=None, hop_length=hop_length)
                        play(synthesized_concat, fs=fs)
        if False:
            plt.plot(loss_list)
            plt.title("Loss")
            plt.show()

            play(pred_concat, fs=fs)
            play(residual_concat, fs=fs)

            plt.plot(target_concat)
            plt.show()

            play(residual_concat, fs=fs)
            play(pred_concat, fs=fs)

            synthesized = ssm.synthesize(data[:, :, 0:1])
            synthesized_concat = concatenate_prediction(synthesized, prediction=None, hop_length=hop_length)
            play(synthesized_concat, fs=fs)


""" Tempo tracking with prediction errors from gradient based predictive coding"""

import librosa, librosa.display
import scipy.stats
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

import PLP_model.PDDSP_encoder
from PredictiveCoding import *

def novelty_to_pulse(
        oenv=None,
        sr=22050,
        tempo_min=30,
        tempo_max=200,
        hop_length=1,
        win_length=32,
        clip_value_min=0.0,
        prior=None):
    # get fourier tempogram
    tempogram = tf.transpose(PLP_model.PDDSP_encoder.PDDSP_spectral_ops.stft(oenv, win_length,
                                            frame_step=1, fft_length=win_length, pad_end=False,
                                            center=True, window_fn=tf.signal.hann_window))

    # restrict to tempo range prior
    tempo_frequencies = tf.cast(PLP_model.PDDSP_encoder.fourier_tempo_frequencies(sr=sr,
                                                  hop_length=hop_length,
                                                  win_length=win_length), dtype=tf.float32)
    mask = tempo_frequencies < tempo_max
    mask = tf.tile(mask[:, tf.newaxis], [1, tempogram.shape[1]])
    tempogram = tempogram * tf.cast(mask, dtype=tempogram.dtype)
    mask = tempo_frequencies > tempo_min
    mask = tf.tile(mask[:, tf.newaxis], [1, tempogram.shape[1]])
    tempogram = tempogram * tf.cast(mask, dtype=tempogram.dtype)

    # discard everything below the peak
    ftmag = tf.math.log1p(1e6 * np.abs(tempogram))
    if prior is not None:
        log_prob = tf.squeeze(prior.log_prob(tempo_frequencies))
        log_prob = tf.tile(log_prob[:, tf.newaxis], [1, ftmag.shape[1]])
        ftmag += log_prob
    peak_values = tf.math.reduce_max(ftmag, axis=0, keepdims=True)
    peak_values = tf.tile(peak_values, [ftmag.shape[0], 1])
    tempogram = tf.cast(ftmag >= peak_values, dtype=tempogram.dtype) * tempogram

    tempogram = tempogram.numpy()
    tempogram /= 0.000001 + np.abs(tempogram.max(axis=0, keepdims=True))
    tempogram = tf.cast(tempogram, dtype=tf.complex64)

    # Compute pulse by inverting the tempogram
    pulse = PLP_model.PDDSP_spectral_ops.inverse_stft(
        tf.transpose(tempogram), win_length, 1, fft_length=win_length, center=True,
       window_fn=tf.signal.inverse_stft_window_fn(1, forward_window_fn=tf.signal.hann_window))

    # retain only the positive part and normalize
    pulse /= tf.math.reduce_max(pulse)
    pulse -= tf.math.reduce_mean(pulse)
    pulse -= clip_value_min
    pulse = tf.clip_by_value(pulse, clip_value_min=0.0, clip_value_max=100000)
    return pulse


def compute_prediction_error_PLP(audio, sr, plot=False, layer_updates=1, batch_size=128, win_length=1024):
    """ compute dominant local pulse on prediction error"""
    # use a single update so that we do not update the prior

    # Params for inputs to the network --> 2 PCN timesteps in each frame
    hop_length = 512
    frame_length = win_length+hop_length

    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).astype(np.float32).T
    frames = np.expand_dims(frames, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(frames)
    dataset = dataset.batch(batch_size)
    dataset_len = len(list(dataset))  # get number of batches
    print("Input batches:", dataset_len)
    dataset = dataset.as_numpy_iterator()
    EY_np = []

    for b in range(dataset_len):
        CA, CB, CC, EX, EY, pred = predictive_coding_filter(batch=next(dataset), n_sequences=dataset_len,
                                                            use_FFT=True, win_length=win_length, hop_length=hop_length,
                                                            ndim=1, layer_updates=layer_updates, log_details=False)
        EY_np_ = np.concatenate(EY, axis=0)
        EY_np_ = np.concatenate([np.zeros_like(EY_np_[0:1]), EY_np_], axis=0)
        EY_np.append(EY_np_)
    EY_np = np.concatenate(EY_np, axis=0)
    # aggregate FFT bins over all frequencies. We could use a learned frequency prior here
    EY_np = np.median(EY_np, axis=1)
    EY_np = np.abs(EY_np)
    EY_np /= np.max(EY_np)
    EY_np -= np.mean(EY_np)

    pulse = novelty_to_pulse(oenv=np.squeeze(EY_np),
            sr=sr,
            hop_length=hop_length,
            win_length=384,
            prior=None)

    tf.print("pulse", pulse.shape)

    peaks = np.flatnonzero(librosa.util.localmax(pulse))
    times = librosa.times_like(np.squeeze(EY_np), sr=sr, hop_length=hop_length)
    peaks_in_seconds = times[peaks]

    if plot:
        details = False
        # compare with librosa PLP
        libr_plp_onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        libr_plp_pulse = librosa.beat.plp(onset_envelope=libr_plp_onset_env, sr=sr)
        libr_plp_times = librosa.times_like(libr_plp_onset_env, sr=sr)
        libr_plp_beats = np.flatnonzero(librosa.util.localmax(libr_plp_pulse))
        plt.plot(libr_plp_times, libr_plp_pulse)
        plt.vlines(libr_plp_times[libr_plp_beats], 0, 1, alpha=0.5, color='g',
                   linestyle='--', label='Librosa PLP Beats')
        o_env = librosa.onset.onset_strength_multi(audio, sr=sr, center=False, hop_length=hop_length, n_fft=win_length)[0]
        times_librosa = librosa.times_like(o_env, sr=sr, hop_length=hop_length)
        if details: plt.plot(times, pulse, label="Dominant pulse of prediction error")
        if details: plt.plot(times, EY_np, label="Prediction error (novelty) for full spectrum")
        if details: plt.plot(times_librosa, o_env/(np.max(o_env)), label="Librosa spectral flux novelty")
        plt.vlines(peaks_in_seconds, 0, 1, alpha=0.5, color='r',
                     linestyle='--', label='Beats from prediction error')
        plt.legend()
        plt.show()

    return peaks_in_seconds

if __name__ == '__main__':
    """Testing the code"""

    #load demo audio
    y, sr = librosa.load(librosa.ex('nutcracker'), duration=20)

    # compute PLP from prediction error
    peaks_predictive_coding = compute_prediction_error_PLP(y, sr, layer_updates=1, plot=True)
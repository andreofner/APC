""" Predictive Encoder"""
import sys
sys.path.append('APC')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import ddsp.core as core

import PLP_model.PDDSP_spectral_ops as PDDSP_spectral_ops


def np_diff(a, n=1, axis=-1):
    """Tensorflow implementation of np.diff"""
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
            "order must be non-negative but got " + repr(n))
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return np_diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]


def audio_to_spectralflux_tf(audio, N, H, Fs):
    """ computes novelty via differences in the spectral energy between adjacent frames """
    X = tf.transpose(PDDSP_spectral_ops.stft(audio, N, H,
                                    fft_length=N, pad_end=False, center=True, # todo centering?
                                    window_fn=tf.signal.hann_window))
    gamma = 10 # todo tune compression prior, make adaptive?
    Y = tf.math.log(1 + gamma * np.abs(X))
    Y_diff = np_diff(Y, n=1)
    Y_diff = tf.clip_by_value(Y_diff, clip_value_min=0., clip_value_max=1000000.) # todo

    # todo replace the audio filtering with adaptive weighting of FT bins:
    nov = tf.reduce_mean(Y_diff, axis=0) # todo tune aggregation function
    nov = tf.concat([nov, np.array([0])], axis=0)
    Fs_nov = Fs / H
    nov -= tf.math.reduce_mean(nov) # todo tune output normalization
    nov = tf.clip_by_value(nov, clip_value_min=0., clip_value_max=1000000.)
    nov /= tf.math.reduce_max(nov)  # normalize

    return nov, Fs_nov


def get_slope(prev, cur):
    return tf.cond(prev[0] < cur, lambda: (cur, ascending_or_valley(prev, cur)), lambda: (cur, descending_or_peak(prev, cur)))


def ascending_or_valley(prev, cur):
    return tf.cond(tf.logical_or(tf.equal(prev[1], 'A'), tf.equal(prev[1], 'V')), lambda: np.array('A'), lambda: np.array('V'))


def descending_or_peak(prev, cur):
    return tf.cond(tf.logical_or(tf.equal(prev[1], 'A'), tf.equal(prev[1], 'V')), lambda: np.array('P'), lambda: np.array('D'))


def label_local_extrema(tens):
    """Return a vector of chars indicating ascending, descending, peak, or valley slopes"""
    initializer = (np.array(0, dtype=np.float32), np.array('A'))
    slope = tf.scan(get_slope, tens, initializer)
    return slope[1][1:]


def find_local_maxima(tens):
    """Tensorflow peak picking via local maxima
    Returns the indices of the local maxima of the first dimension of the tensor
    Based on https://stackoverflow.com/questions/48178286/finding-local-maxima-with-tensorflow
    """
    return tf.squeeze(tf.where(tf.equal(label_local_extrema(tens), 'P')))


def fft_frequencies(sr=22050, n_fft=2048):
    """Tensorflow-based implementation of np.fft.fftfreq """
    # TODO endpoint=True
    return tf.linspace(0, tf.cast(sr/2., dtype=tf.int32), tf.cast(1. + n_fft // 2., dtype=tf.int32))


def fourier_tempo_frequencies(sr=22050, win_length=384, hop_length=512):
    """Tensorflow-based implementation of librosa.core.fourier_tempo_frequencies"""
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)


def bandpass_filter_audio(audio, f_low=400, f_high=450):
    """Bandpass filters audio to given frequency range"""
    filtered_audio = core.sinc_filter(audio, f_low, window_size=256, high_pass=True)
    filtered_audio = core.sinc_filter(filtered_audio, f_high, window_size=256, high_pass=False)
    return tf.squeeze(filtered_audio)


def plp_tf(
        y,
        sr=22050,
        tempo_min=30,
        tempo_max=300,
        hop_length=1,
        win_length=512,
        hop_length_novelty=256,
        win_length_novelty=1024,
        loudness_min=0.1,
        loudness_max=1.,
        prior=None):
    """Tensorflow-based implementation of librosa.beat.plp
     Process chain: audio -> spectral flux novelty -> Fourier tempogram -> local pulse """
    y = tf.squeeze(y)

    # get spectral flux novelty
    oenv, sr_ = audio_to_spectralflux_tf(y, win_length_novelty, hop_length_novelty, sr)

    # get fourier tempogram
    tempogram = tf.transpose(PDDSP_spectral_ops.stft(oenv, win_length,
                                            frame_step=hop_length,
                                            fft_length=win_length, pad_end=False,
                                            center=True, window_fn=tf.signal.hann_window))

    # restrict to tempo range prior
    tempo_frequencies = tf.cast(fourier_tempo_frequencies(sr=sr_,
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

    # todo keep only phase
    #ftgram = tempogram.numpy()
    #import librosa
    #ftgram /= librosa.util.tiny(ftgram) ** 0.5 + np.abs(ftgram.max(axis=0, keepdims=True))
    #tempogram = tf.cast(ftgram, dtype=tf.complex64)

    # Compute pulse by inverting the tempogram
    pulse = PDDSP_spectral_ops.inverse_stft(
        tf.transpose(tempogram), win_length, hop_length, fft_length=win_length, center=True,
       window_fn=tf.signal.inverse_stft_window_fn(hop_length, forward_window_fn=tf.signal.hann_window))

    # retain only the positive part and normalize
    pulse /= tf.math.reduce_max(pulse)
    pulse -= tf.math.reduce_mean(pulse)
    pulse = tf.clip_by_value(pulse, clip_value_min=0, clip_value_max=100000)

    # compute mean period and expected next onset position
    F_mean = dominant_freq_from_tempogram(tempogram, tempo_frequencies)
    period_mean, mean_offset, next_onset_shift, peaks = period_from_pulse(pulse, F_mean,
                                                  sr=sr_, loudness_min=loudness_min,
                                                  loudness_max=loudness_max)
    period_mean, next_onset_shift, mean_offset = (period_mean/sr_)*sr, (next_onset_shift/sr_)*sr, (mean_offset/sr_)*sr

    return pulse, tempogram, oenv, sr_, F_mean, period_mean, mean_offset, next_onset_shift


def period_from_pulse(pulse, F_mean_in_Hz, sr, loudness_min=0.1, loudness_max=1.):
    """Compute mean period and the next expected onset position"""

    # Find last peak in the pulse
    peaks = find_local_maxima(tf.clip_by_value(pulse, clip_value_min=loudness_min,
                                               clip_value_max=loudness_max))[1:]
    first_peak = tf.cast(peaks[0], dtype=tf.float32) if peaks.shape[0] > 1 else 0.
    last_peak = tf.cast(peaks[-1], dtype=tf.float32) if peaks.shape[0] > 1 else 0.

    # return average offset for each peak
    mean_offset = tf.math.reduce_mean(tf.cast([tf.math.floormod(tf.cast(peak, dtype=tf.int64), tf.cast(sr, dtype=tf.int64))
                                               for peak in peaks], dtype=tf.float32))

    # Compute mean period
    period_mean = (1/F_mean_in_Hz) * sr

    # Predict the first onset in the next audio input
    next_onset_shift = tf.abs(period_mean - (tf.cast(pulse.shape[0], dtype=tf.float32) - last_peak))
    next_onset_shift = tf.math.floormod(next_onset_shift, period_mean)

    return period_mean, mean_offset, next_onset_shift, peaks


def dominant_freq_from_tempogram(tempogram, tempo_frequencies, return_Hz = True):
    """Calculate dominant frequency from tempogram."""

    tempo_BPM_max = tempo_frequencies \
                            * tf.cast(tf.math.abs(tempogram[:, 0])
                                      == tf.math.reduce_max(tf.math.abs(tempogram[:, 0])),
                                      tempo_frequencies.dtype)
    if return_Hz:
        dominant_tempo = tf.cast(tf.math.reduce_max(tempo_BPM_max)/60, dtype=tf.float32)
    else:
        dominant_tempo = tf.cast(tf.math.reduce_max(tempo_BPM_max), dtype=tf.float32)

    weights = tf.cast(tf.math.abs(tempogram[:, 0]), dtype=tf.float32)
    weighted_mean = tf.nn.weighted_moments(tempo_frequencies, axes=[0], frequency_weights=weights)[0]

    if return_Hz:
        weighted_mean_tempo = tf.expand_dims(tf.cast(weighted_mean/60, dtype=tf.float32), axis = 0)
    else:
        weighted_mean_tempo = tf.expand_dims(tf.cast(weighted_mean, dtype=tf.float32), axis = 0)
    dominant_tempo = tf.expand_dims(dominant_tempo, axis=0)
    out = tf.concat([dominant_tempo, weighted_mean_tempo], axis=0)

    return tf.cast(out, dtype=tf.float32)


def encode_song(y, sr, chunks=8,
                tempo_min=60,
                tempo_max=300,
                f_low=400, f_high=450,
                loudness_min=0.1, loudness_max=1,
                filter=False, plot=True,
                padding_seconds=4,
                frame_step=0.1):
    """Run PLP encoder over all chunks in a song"""
    if chunks != 0:
        y_list = tf.signal.frame(y, sr*chunks, int(sr*frame_step), pad_end=True, pad_value=0, axis=-1) # TODO padding
    else:
        y_list = [tf.cast(y, dtype=tf.float32)]
    tempo_mean_list, period_mean_list, beats_list = None, None, None

    for y, index in zip(y_list, range(len(y_list))):
        # Bandpass filter audio
        if filter:
            y = bandpass_filter_audio(y[tf.newaxis,:], f_low=f_low, f_high=f_high)

        # Compute phase and period
        pulse, tempogram, oenv, sr_, F_mean, period_mean, mean_offset, next_onset_shift = plp_tf(
                y=y, sr=sr,
                tempo_min=tempo_min,
                tempo_max=tempo_max,
                hop_length=1,
                win_length=512,
                hop_length_novelty=256,
                win_length_novelty=1024,
                loudness_min=0.2,
                loudness_max=1.)

        if tempo_mean_list is None:
            tempo_mean_list = [F_mean] # in Hz
            period_mean_list = [mean_offset/sr] # in seconds
        else:
            tempo_mean_list.append(F_mean) # in Hz
            period_mean_list.append(mean_offset/sr) # in seconds

        # Compute beat positions via local maxima
        beats = find_local_maxima(tf.clip_by_value(pulse,
                                                   clip_value_min=loudness_min,
                                                   clip_value_max=loudness_max))[1:]

        # correct timing in each chunk
        beats = tf.cast(beats, dtype=tf.float32) + (tf.cast(index, dtype=tf.float32) * pulse.shape[0])
        beats = beats - padding_seconds*sr_ #remove padding #TODO fix
        if beats_list is None:
            beats_list = beats
        else:
            beats_list = np.concatenate([beats_list, beats], axis=0)

        # Optionally plot tempogram and pulse for each input
        if plot:
            plot_tempogram_and_pulse(tempogram, pulse, oenv, sr_, 1)
            plot_librosa_tempogram(y.numpy(), sr)

    # samples to time
    beats_list = np.asarray(beats_list) / sr_
    return tempo_mean_list, period_mean_list, beats_list, oenv.numpy()


"""Helper functions"""
def plot_librosa_tempogram(y, sr, hop_length = 512):
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
                                                  hop_length=hop_length)
    librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='fourier_tempo', cmap='magma')
    plt.title('Librosa Fourier tempogram')
    plt.show()


def plot_tempogram_and_pulse(tempogram, pulse, oenv, sr_, hop_length, plot_pulse=True):
    """Plots tempogram and local pulse."""
    tempogram = tempogram.numpy()
    librosa.display.specshow(np.abs(tempogram), sr=sr_, hop_length=hop_length,
                             x_axis='time', y_axis='fourier_tempo', cmap='magma')
    plt.show()
    peaks = find_local_maxima(tf.clip_by_value(pulse, clip_value_min=0.1,
                                               clip_value_max=1.))[1:]
    if plot_pulse:
        oenv = oenv.numpy()
        pulse = pulse.numpy()
        plt.plot(oenv, color="black")
        plt.plot(pulse, color="blue")
        plt.plot(peaks, pulse[peaks.numpy()], "ro")
        plt.show()


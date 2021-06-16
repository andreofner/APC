""" Custom TF spectral ops. Includes improved padding for the Short-time Fourier Transform
in tf.signal. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.ops.signal import util_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.ops.signal.spectral_ops import _enclosing_power_of_two
from tensorflow.python.ops.signal.shape_ops import _infer_frame_shape
from tensorflow.python.ops.signal import reconstruction_ops
import tensorflow as tf

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1, center=False,
          name=None):
    """Expands `signal`'s `axis` dimension into frames of `frame_length`.
    Slides a window of size `frame_length` over `signal`'s `axis` dimension
    with a stride of `frame_step`, replacing the `axis` dimension with
    `[frames, frame_length]` frames.
    If `pad_end` is True, window positions that are past the end of the `axis`
    dimension are padded with `pad_value` until the window moves fully past the
    end of the dimension. Otherwise, only window positions that fully overlap the
    `axis` dimension are produced.
    @Andre: add centering
    For example:
    >>> # A batch size 3 tensor of 9152 audio samples.
    >>> audio = tf.random.normal([3, 9152])
    >>>
    >>> # Compute overlapping frames of length 512 with a step of 180 (frames overlap
    >>> # by 332 samples). By default, only 49 frames are generated since a frame
    >>> # with start position j*180 for j > 48 would overhang the end.
    >>> frames = tf.signal.frame(audio, 512, 180)
    >>> frames.shape.assert_is_compatible_with([3, 49, 512])
    >>>
    >>> # When pad_end is enabled, the final two frames are kept (padded with zeros).
    >>> frames = tf.signal.frame(audio, 512, 180, pad_end=True)
    >>> frames.shape.assert_is_compatible_with([3, 51, 512])

    If the dimension along `axis` is N, and `pad_end=False`, the number of frames
    can be computed by:
     ```python
     num_frames = 1 + (N - frame_size) // frame_step
     ```
     If `pad_end=True`, the number of frames can be computed by:
    ```python
    num_frames = -(-N // frame_step) # ceiling division
    ```
    Args:
      signal: A `[..., samples, ...]` `Tensor`. The rank and dimensions
        may be unknown. Rank must be at least 1.
      frame_length: The frame length in samples. An integer or scalar `Tensor`.
      frame_step: The frame hop size in samples. An integer or scalar `Tensor`.
      pad_end: Whether to pad the end of `signal` with `pad_value`.
      pad_value: An optional scalar `Tensor` to use where the input signal
        does not exist when `pad_end` is True.
      axis: A scalar integer `Tensor` indicating the axis to frame. Defaults to
        the last axis. Supports negative values for indexing from the end.
      name: An optional name for the operation.
    Returns:
      A `Tensor` of frames with shape `[..., num_frames, frame_length, ...]`.
    Raises:
      ValueError: If `frame_length`, `frame_step`, `pad_value`, or `axis` are not
        scalar.
    """
    with ops.name_scope(name, "frame", [signal, frame_length, frame_step,
                                        pad_value]):

        def maybe_constant(val):
            val_static = tensor_util.constant_value(val)
            return (val_static, True) if val_static is not None else (val, False)

        signal = ops.convert_to_tensor(signal, name="signal")

        if center: # TODO improve shape checking /remove duplicates
            signal_shape, signal_shape_is_static = maybe_constant(
                array_ops.shape(signal))
            signal_rank = array_ops.rank(signal)
            # Axis can be negative. Convert it to positive.
            axis = math_ops.range(signal_rank)[axis]
            outer_dimensions, length_samples, inner_dimensions = array_ops.split(
                signal_shape, [axis, 1, signal_rank - 1 - axis])
            length_samples = array_ops.reshape(length_samples, [])
            num_outer_dimensions = array_ops.size(outer_dimensions)
            num_inner_dimensions = array_ops.size(inner_dimensions)

            pad_samples = tf.cast(frame_length // 2, dtype=tf.int32)
            paddings = array_ops.concat([
                array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
                ops.convert_to_tensor([[pad_samples, pad_samples]]),
                array_ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype)
            ], 0)
            signal = tf.pad(signal, paddings, mode="REFLECT")

        frame_length = ops.convert_to_tensor(frame_length, name="frame_length")
        frame_step = ops.convert_to_tensor(frame_step, name="frame_step")
        axis = ops.convert_to_tensor(axis, name="axis")

        signal.shape.with_rank_at_least(1)
        frame_length.shape.assert_has_rank(0)
        frame_step.shape.assert_has_rank(0)
        axis.shape.assert_has_rank(0)

        result_shape = _infer_frame_shape(signal, frame_length, frame_step, pad_end,
                                          axis)

        signal_shape, signal_shape_is_static = maybe_constant(
            array_ops.shape(signal))
        axis, axis_is_static = maybe_constant(axis)

        if signal_shape_is_static and axis_is_static:
            # Axis can be negative. Convert it to positive.
            axis = range(len(signal_shape))[axis]
            outer_dimensions, length_samples, inner_dimensions = np.split(
                signal_shape, indices_or_sections=[axis, axis + 1])
            length_samples = length_samples.item()
        else:
            signal_rank = array_ops.rank(signal)
            # Axis can be negative. Convert it to positive.
            axis = math_ops.range(signal_rank)[axis]
            outer_dimensions, length_samples, inner_dimensions = array_ops.split(
                signal_shape, [axis, 1, signal_rank - 1 - axis])
            length_samples = array_ops.reshape(length_samples, [])
        num_outer_dimensions = array_ops.size(outer_dimensions)
        num_inner_dimensions = array_ops.size(inner_dimensions)

        # If padding is requested, pad the input signal tensor with pad_value.
        if pad_end:
            pad_value = ops.convert_to_tensor(pad_value, signal.dtype)
            pad_value.shape.assert_has_rank(0)

            # Calculate number of frames, using double negatives to round up.
            num_frames = -(-length_samples // frame_step)

            # Pad the signal by up to frame_length samples based on how many samples
            # are remaining starting from last_frame_position.
            pad_samples = math_ops.maximum(
                0, frame_length + frame_step * (num_frames - 1) - length_samples)

            # Pad the end of the inner dimension of signal by pad_samples.
            paddings = array_ops.concat([
                array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
                ops.convert_to_tensor([[0, pad_samples]]),
                array_ops.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype)
            ], 0)

            signal = array_ops.pad(signal, paddings, constant_values=pad_value)

            signal_shape = array_ops.shape(signal)
            length_samples = signal_shape[axis]
        else:
            num_frames = math_ops.maximum(
                0, 1 + (length_samples - frame_length) // frame_step)

        subframe_length, _ = maybe_constant(util_ops.gcd(frame_length, frame_step))
        subframes_per_frame = frame_length // subframe_length
        subframes_per_hop = frame_step // subframe_length
        num_subframes = length_samples // subframe_length

        slice_shape = array_ops.concat([outer_dimensions,
                                        [num_subframes * subframe_length],
                                        inner_dimensions], 0)
        subframe_shape = array_ops.concat([outer_dimensions,
                                           [num_subframes, subframe_length],
                                           inner_dimensions], 0)
        subframes = array_ops.reshape(array_ops.strided_slice(
            signal, array_ops.zeros_like(signal_shape),
            slice_shape), subframe_shape)

        # frame_selector is a [num_frames, subframes_per_frame] tensor
        # that indexes into the appropriate frame in subframes. For example:
        # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
        frame_selector = array_ops.reshape(
            math_ops.range(num_frames) * subframes_per_hop, [num_frames, 1])

        # subframe_selector is a [num_frames, subframes_per_frame] tensor
        # that indexes into the appropriate subframe within a frame. For example:
        # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
        subframe_selector = array_ops.reshape(
            math_ops.range(subframes_per_frame), [1, subframes_per_frame])

        # Adding the 2 selector tensors together produces a [num_frames,
        # subframes_per_frame] tensor of indices to use with tf.gather to select
        # subframes from subframes. We then reshape the inner-most
        # subframes_per_frame dimension to stitch the subframes together into
        # frames. For example: [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
        selector = frame_selector + subframe_selector

        frames = array_ops.reshape(
            array_ops.gather(subframes, selector, axis=axis),
            array_ops.concat([outer_dimensions, [num_frames, frame_length],
                              inner_dimensions], 0))

        if result_shape:
            frames.set_shape(result_shape)
        return frames


def stft(signals, frame_length, frame_step, fft_length=None,
         window_fn=window_ops.hann_window, center=True,
         pad_end=False, name=None):
  """Computes the [Short-time Fourier Transform][stft] of `signals`.
  Implemented with TPU/GPU-compatible ops and supports gradients.
  Args:
    signals: A `[..., samples]` `float32`/`float64` `Tensor` of real-valued
      signals.
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT to apply.
      If not provided, uses the smallest power of 2 enclosing `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    pad_end: Whether to pad the end of `signals` with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    name: An optional name for the operation.
  Returns:
    A `[..., frames, fft_unique_bins]` `Tensor` of `complex64`/`complex128`
    STFT values where `fft_unique_bins` is `fft_length // 2 + 1` (the unique
    components of the FFT).
  Raises:
    ValueError: If `signals` is not at least rank 1, `frame_length` is
      not scalar, or `frame_step` is not scalar.
  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  """
  with ops.name_scope(name, 'stft', [signals, frame_length,
                                     frame_step]):
    signals = ops.convert_to_tensor(signals, name='signals')
    signals.shape.with_rank_at_least(1)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)

    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')

    framed_signals = frame(
        signals, frame_length, frame_step, pad_end=pad_end, center=center)

    # Optionally window the framed signals.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=framed_signals.dtype)
      framed_signals *= window

    # fft_ops.rfft produces the (fft_length/2 + 1) unique components of the
    # FFT of the real windowed signals in framed_signals.
    return fft_ops.rfft(framed_signals, [fft_length])



def inverse_stft(stfts,
                 frame_length,
                 frame_step,
                 fft_length=None,
                 window_fn=window_ops.hann_window,
                 center=False,
                 name=None):
  """Computes the inverse [Short-time Fourier Transform][stft] of `stfts`.
  To reconstruct an original waveform, a complementary window function should
  be used with `inverse_stft`. Such a window function can be constructed with
  `tf.signal.inverse_stft_window_fn`.
  Example:
  ```python
  frame_length = 400
  frame_step = 160
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(waveform, frame_length, frame_step)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(frame_step))
  ```
  If a custom `window_fn` is used with `tf.signal.stft`, it must be passed to
  `tf.signal.inverse_stft_window_fn`:
  ```python
  frame_length = 400
  frame_step = 160
  window_fn = tf.signal.hamming_window
  waveform = tf.random.normal(dtype=tf.float32, shape=[1000])
  stft = tf.signal.stft(
      waveform, frame_length, frame_step, window_fn=window_fn)
  inverse_stft = tf.signal.inverse_stft(
      stft, frame_length, frame_step,
      window_fn=tf.signal.inverse_stft_window_fn(
         frame_step, forward_window_fn=window_fn))
  ```
  Implemented with TPU/GPU-compatible ops and supports gradients.
  Args:
    stfts: A `complex64`/`complex128` `[..., frames, fft_unique_bins]`
      `Tensor` of STFT bins representing a batch of `fft_length`-point STFTs
      where `fft_unique_bins` is `fft_length // 2 + 1`
    frame_length: An integer scalar `Tensor`. The window length in samples.
    frame_step: An integer scalar `Tensor`. The number of samples to step.
    fft_length: An integer scalar `Tensor`. The size of the FFT that produced
      `stfts`. If not provided, uses the smallest power of 2 enclosing
      `frame_length`.
    window_fn: A callable that takes a window length and a `dtype` keyword
      argument and returns a `[window_length]` `Tensor` of samples in the
      provided datatype. If set to `None`, no windowing is used.
    name: An optional name for the operation.
  Returns:
    A `[..., samples]` `Tensor` of `float32`/`float64` signals representing
    the inverse STFT for each input STFT in `stfts`.
  Raises:
    ValueError: If `stfts` is not at least rank 2, `frame_length` is not scalar,
      `frame_step` is not scalar, or `fft_length` is not scalar.
  [stft]: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
  """
  with ops.name_scope(name, 'inverse_stft', [stfts]):
    stfts = ops.convert_to_tensor(stfts, name='stfts')
    stfts.shape.with_rank_at_least(2)
    frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
    frame_length.shape.assert_has_rank(0)
    frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
    frame_step.shape.assert_has_rank(0)
    if fft_length is None:
      fft_length = _enclosing_power_of_two(frame_length)
    else:
      fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
      fft_length.shape.assert_has_rank(0)

    real_frames = fft_ops.irfft(stfts, [fft_length])

    # frame_length may be larger or smaller than fft_length, so we pad or
    # truncate real_frames to frame_length.
    frame_length_static = tensor_util.constant_value(frame_length)
    # If we don't know the shape of real_frames's inner dimension, pad and
    # truncate to frame_length.
    if (frame_length_static is None or real_frames.shape.ndims is None or
        real_frames.shape.as_list()[-1] is None):
      real_frames = real_frames[..., :frame_length]
      real_frames_rank = array_ops.rank(real_frames)
      real_frames_shape = array_ops.shape(real_frames)
      paddings = array_ops.concat(
          [array_ops.zeros([real_frames_rank - 1, 2],
                           dtype=frame_length.dtype),
           [[0, math_ops.maximum(0, frame_length - real_frames_shape[-1])]]], 0)
      real_frames = array_ops.pad(real_frames, paddings)
    # We know real_frames's last dimension and frame_length statically. If they
    # are different, then pad or truncate real_frames to frame_length.
    elif real_frames.shape.as_list()[-1] > frame_length_static:
      real_frames = real_frames[..., :frame_length_static]
    elif real_frames.shape.as_list()[-1] < frame_length_static:
      pad_amount = frame_length_static - real_frames.shape.as_list()[-1]
      real_frames = array_ops.pad(real_frames,
                                  [[0, 0]] * (real_frames.shape.ndims - 1) +
                                  [[0, pad_amount]])

    # The above code pads the inner dimension of real_frames to frame_length,
    # but it does so in a way that may not be shape-inference friendly.
    # Restore shape information if we are able to.
    if frame_length_static is not None and real_frames.shape.ndims is not None:
      real_frames.set_shape([None] * (real_frames.shape.ndims - 1) +
                            [frame_length_static])

    # Optionally window and overlap-add the inner 2 dimensions of real_frames
    # into a single [samples] dimension.
    if window_fn is not None:
      window = window_fn(frame_length, dtype=stfts.dtype.real_dtype)
      real_frames *= window
    reconstructed = reconstruction_ops.overlap_and_add(real_frames, frame_step)

    if center:
        pad_len = tf.cast(frame_length // 2, dtype=tf.int32)
        return reconstructed[pad_len:-pad_len]
    else:
        return reconstructed
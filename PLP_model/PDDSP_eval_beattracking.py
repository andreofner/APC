import sys
sys.path.append('APC')

import mir_eval as mir_eval
import librosa
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import madmom

from PLP_model import PDDSP_encoder
from PLP_model.PLP_tests import compute_prediction_error_PLP

"""____________________________________________________
PDDSP online beat tracking evaluation

Datasets: 
1) SMC MIREX (http://smc.inescporto.pt/data/SMC_MIREX.zip) 
2) NMED-T (https://exhibits.stanford.edu/data/catalog/jn859kj8079)

Tested PDDSP models:
    1) encoder_pulse: Pulse Encoder
    2) TODO kalman_pulse: Pulse Kalman Filter
    3) TODO kalman_decoder: Pulse Kalman Filter with Decoder

Baselines and expected SMC F-scores from literature:
    1) madmom [Boeck & Schedl] 0.401
    2) librosa Beat [Ellis] 0.352 
    3) librosa PLP [Grosche & Muller]
    
"____________________________________________________"""

""" Select evaluation """

max_files = 0
datasets = ["SMC", "NMED-T"]
selected_datset = datasets[1]
sample_rate = 22050

""" Select output"""
print_per_song = False
print_beats = False
plot_beats = False  # TODO Check functionality
plot_with_truth = True
save_results = False

""" Select functionality"""
calculate_beats = True  # if false it will skip the beat tracker and just evaluate, dirty workaround, better would be to move the functionalities
get_scores = True

""" Select which model to evaluate """
models = ["madmom-online", "madmom-offline", "librosa", "librosa_plp",
          "encoder_pulse", "kalman_pulse", "kalman_decoder"]  # kalman_pulse, kalman_decoder not implemented
selected_model = models[4]
print("Evaluating model ", selected_model)

""" Directories"""
# Datasets Directories
if selected_datset == "SMC":
    annotation_dir = "/Users/schleiss/datasets/SMC_MIREX/SMC_MIREX_Annotations/"  # "/Users/andre/Desktop/SMC_MIREX/SMC_MIREX_Annotations/"
    audio_dir = "/Users/schleiss/datasets/SMC_MIREX/SMC_MIREX_Audio/"
    result_dir = "/Users/schleiss/datasets/SMC_MIREX/results/"
elif selected_datset == "NMED-T":
    subjects = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "19", "20", "21", "23"]
    subjects_dir = "/Users/schleiss/datasets/NMED-T/annotation/"
    audio_dir = "/Users/schleiss/datasets/NMED-T/NMEDT_audio_mp3_cut/"
    result_dir = "/Users/schleiss/datasets/NMED-T/results/"
    scores_dir = "/Users/schleiss/datasets/NMED-T/results/scores/"
else:
    raise ValueError("Couldn't find dataset")

""" Select priors """
tempo_prior = 120  # BPM
# TODO select our of other priors, e.g. FFT size, hop_size, peak-picking, ...
# TODO samplerate (16000, 22050 has influence on baselines)

# Helper functions -------------------------------------------------------------------------------
def plot_librosa_beats(y, sr, hop_length=512):
    """Plot librosa beat tracking and spectrogram"""
    fig, ax = plt.subplots(nrows=2, sharex=True)
    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    librosa.display.specshow(librosa.power_to_db(M, ref=np.max), y_axis='mel', x_axis='time',
                             hop_length=hop_length, ax=ax[0])
    ax[0].label_outer()
    ax[0].set(title='Mel spectrogram')
    ax[1].plot(times, librosa.util.normalize(onset_env),
               label='Onset strength')
    # ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')  ## TODO: Throws IndexError: arrays used as indices must be of integer (or boolean) type
    ax[1].legend()
    plt.show()


def plot_beats_total(y, sr, model_beats, reference_beats, title, x_min=0, x_max=35):
    """Plot beats and reference beats on one plot."""
    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)

    model_beat_taps = np.asarray(model_beats)  # in seconds
    model_beat_taps = librosa.time_to_frames(model_beat_taps, sr=sr, hop_length=512)  # in frames

    reference_beats_taps = np.asarray(reference_beats)
    reference_beat_taps = librosa.time_to_frames(reference_beats_taps, sr=sr, hop_length=512)  # in frames

    hop_length = 512
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.vlines(times[model_beat_taps], 0, 1, alpha=0.5, color='r', linestyle='--', label='Model Taps')
    plt.vlines(times[reference_beat_taps], 0, 1, alpha=0.5, color='g', linestyle='--', label='Reference Taps')

    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')

    # formatting and showing only part of signal
    plt.title(title)
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
    plt.tight_layout()
    plt.show()


def plot_beats(y, sr, beats, x_min=0, x_max=10):
    """Function to show beats."""
    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)

    beat_taps = np.asarray(beats)  # in seconds
    beat_taps = librosa.time_to_frames(beat_taps, sr=sr, hop_length=512)  # in frames

    hop_length = 512
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.vlines(times[beat_taps], 0, 1, alpha=0.5, color='r', linestyle='--', label='Taps')
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    plt.xlim(x_min, x_max)
    plt.gca().xaxis.set_major_formatter(librosa.display.TimeFormatter())
    plt.tight_layout()
    plt.show()


def plot_librosa_plp(y, sr, onset_env, pulse, beats_plp):
    """Plot beat tracking with librosa's predominant local pulse."""
    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
    times = librosa.times_like(onset_env, sr=sr)
    ax.plot(times, librosa.util.normalize(onset_env),
            label='Onset strength')
    # Limit the plot to a 15-second window
    times = librosa.times_like(pulse, sr=sr)
    ax.plot(times, librosa.util.normalize(pulse),
            label='PLP')
    ax.vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
              linestyle='--', label='PLP Beats')
    ax.legend()
    ax.set(title='librosa.beat.plp', xlim=[5, 20])
    ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    plt.show()


def madmom_beat_tracker(audiofile, online):
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100, online=online)
    act = madmom.features.beats.RNNBeatProcessor()(audiofile)
    beats = proc(act)
    return beats


def encoder_pulse(audio, sr, plot, chunks_sec=5):
    """Send chunked input to PLP encoder. Currently not used."""
    n_chunks = int(len(audio)/(sr*chunks_sec))
    beats = []
    # loop over audio
    for i in range(0, n_chunks):
        # get audio chunk and local beats
        y = audio[chunks_sec*sr*i:chunks_sec*sr*(i+1)]
        local_beats = compute_prediction_error_PLP(y, sr, plot=plot)

        # stitch together with appending beats and adding the chunk_seconds
        beats.append(local_beats+(chunks_sec*i))

    # get desired format
    beats = np.hstack(beats).squeeze()
    return beats


def get_beats_from_model(selected_model, audio_path, plot_beats):
    """Get the beats for the selected model given an audio path"""

    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Madmom beat tracker online
    if selected_model == "madmom-online":
        beats = madmom_beat_tracker(audio_path, online=True)

    # madmom beat tracker offline
    if selected_model == "madmom-offline":
        beats = madmom_beat_tracker(audio_path, online=False)

    # Librosa beat tracker
    if selected_model == "librosa":
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        # if plot_beats: plot_librosa_beats(y=y, sr=sr)
        if plot_beats: plot_beats(y=y, sr=sr, beats=beats)

    # librosa PLP
    if selected_model == "librosa_plp":
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        beats_frames = np.flatnonzero(librosa.util.localmax(pulse))
        beats = librosa.frames_to_time(beats_frames, sr=sr)
        if plot_beats: plot_librosa_plp(y, sr, onset_env, pulse, beats_frames)

    # PLP encoder
    if selected_model == "encoder_pulse":
        beats = compute_prediction_error_PLP(y, sr, plot=False, batch_size=128, layer_updates=2)
        # if you are running out of RAM, you could potentially send chunks of audio
        #beats = encoder_pulse(audio=y, sr=sr,chunk_seconds=5, plot=False)
        if plot_beats: plot_beats(y=y, sr=sr, beats=beats)

    if selected_model == "kalman_pulse":
        raise NotImplementedError

    if selected_model == "kalman_decoder":
        raise NotImplementedError

    return y, sr, beats


def get_scores_for_audio(result_path, annotation_path):
    """Calculate the evaluation scores for one model and one audio."""
    # load model and annotation beats
    model_beats = np.genfromtxt(fname=result_path)
    reference_beats = mir_eval.io.load_events(annotation_path)

    # trim beats for the first 5 seconds
    reference_beats = mir_eval.beat.trim_beats(reference_beats)
    model_beats = mir_eval.beat.trim_beats(model_beats)

    # calculate scores
    scores = mir_eval.beat.evaluate(reference_beats, model_beats)

    if print_beats:
        print("Ground truth: ", reference_beats)
        print("Prediction ", model_beats)

    if print_per_song:
        for k, v in zip(scores.keys(), scores.values()): print(k, " - ", v)

    if plot_with_truth:
        plot_beats_total(y, sr, model_beats=model_beats, reference_beats=reference_beats, title=annotation_path, x_min=5, x_max=15)

    return scores


def get_total_scores(audio_dir, result_dir, annotation_dir):
    """Calculate the total scores over all audios."""
    total_scores = None
    file_count = 0

    for (file, file_index) in zip(os.listdir(audio_dir), range(len(os.listdir(audio_dir)))):
        if file_count < max_files or max_files <= 0:
            filename = os.fsdecode(file)
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                # set result path
                result_path = result_dir + filename.split(".")[0] + "_" + selected_model + ".txt"

                # find matching annotation file
                for a_file in os.listdir(annotation_dir):
                    a_filename = os.fsdecode(a_file)

                    # mapping files for SMC dataset
                    if a_filename.startswith(filename.split(".")[0]) and a_filename.endswith(".txt"):
                        print(file_index, "Audio: ", filename, " annotation ", a_filename)
                        file_count += 1
                        scores = get_scores_for_audio(result_path=result_path,
                                                      annotation_path=annotation_dir + a_filename)

                    # mapping files for NMED-T dataset based on song identifier
                    # annotation files PPP_SS.txt PPP=Participant number SS=song number
                    # audio files SS_title.mp3
                    elif a_filename.split("_")[1].startswith(filename.split("_")[0]) and a_filename.endswith(
                            ".txt"):
                        print(file_index, "Audio: ", filename, " annotation ", a_filename)
                        file_count += 1
                        scores = get_scores_for_audio(result_path=result_path,
                                                      annotation_path=annotation_dir + a_filename)

                # add scores to total score
                if total_scores is None:
                    total_scores = scores
                else:
                    for k in total_scores.keys():
                        total_scores[k] += scores[k]

    # calculate average scores
    for k in total_scores.keys():
        total_scores[k] /= file_count

    print("Evaluated files: ", file_count)

    return total_scores


# Dataset processing for Beat Tracking-------------------------------------------------------
file_count = 0
if calculate_beats:
    # go through all audio files and calculate beats for it
    for (file, file_index) in zip(os.listdir(audio_dir), range(len(os.listdir(audio_dir)))):
        if file_count < max_files or max_files <= 0:
            filename = os.fsdecode(file)
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                # set result path
                result_path = result_dir + filename.split(".")[0] + "_" + selected_model + ".txt"

                # get beats from model and save to result path
                y, sr, beats = get_beats_from_model(selected_model, audio_dir + filename, plot_beats=plot_beats)
                np.savetxt(fname=result_path, X=beats)
                print("Beats saved to {}".format(result_path))
                file_count += 1

# Evaluation of Beat tracking with reference -------------------------------------------------------
if get_scores:
    if selected_datset == "SMC":
        total_scores = get_total_scores(audio_dir=audio_dir, result_dir=result_dir, annotation_dir=annotation_dir)

    elif selected_datset == "NMED-T":
        total_scores = None

        # go over all subjects and get the matching scores
        for subject_count, selected_subject in enumerate(subjects, start=1):
            annotation_dir = subjects_dir + selected_subject + "/"

            subject_scores = get_total_scores(audio_dir=audio_dir, result_dir=result_dir, annotation_dir=annotation_dir)

            # add subject scores to total score
            if total_scores is None:
                total_scores = subject_scores
            else:
                for k in total_scores.keys():
                    total_scores[k] += subject_scores[k]

        # calculate average scores over all subjects
        for k in total_scores.keys():
            total_scores[k] /= subject_count
    else:
        raise ValueError("Unknown Dataset")

    # print results
    print("Dataset: {}. Model: {}".format(selected_datset, selected_model))
    for k, v in zip(total_scores.keys(), total_scores.values()): print(k, " - ", v)

    # save results to npy file
    if save_results:
        save_path = scores_dir + "{}_{}_annot-{}.npy".format(selected_datset, selected_model, selected_subject)
        np.save(save_path, total_scores)
        print("Saved scores to: ", save_path)

    # load and print to check
    #scores = np.load(save_path, allow_pickle='TRUE').item()
    #for k, v in zip(total_scores.keys(), total_scores.values()): print(k, " - ", v)


""" Results SMC:

MADMOM (OFFLINE):
    Evaluated SMC files:  217
    F-measure  -  0.5683128209626371
    Cemgil  -  0.44919744764958575
    Cemgil Best Metric Level  -  0.4866176392132091
    Goto  -  0.22580645161290322
    P-score  -  0.6610660719355975
    Correct Metric Level Continuous  -  0.36065961707131095
    Correct Metric Level Total  -  0.4743360329180148
    Any Metric Level Continuous  -  0.46361459223492324
    Any Metric Level Total  -  0.6225396789903174
    Information gain  -  0.3115967747484779
    
MADMOM (ONLINE):
    Evaluated SMC files:  217
    F-measure  -  0.5207962980599968
    Cemgil  -  0.40459152154729017
    Cemgil Best Metric Level  -  0.4257031579989529
    Goto  -  0.07373271889400922
    P-score  -  0.5693582008599621
    Correct Metric Level Continuous  -  0.22696348901356994
    Correct Metric Level Total  -  0.3624391786755101
    Any Metric Level Continuous  -  0.27483216981925906
    Any Metric Level Total  -  0.43325416461028543
    Information gain  -  0.29440895191272987
    
LIBROSA:
    Evaluated SMC files:  217
    F-measure  -  0.3386513297666259
    Cemgil  -  0.227140711124723
    Cemgil Best Metric Level  -  0.28778144415893875
    Goto  -  0.059907834101382486
    P-score  -  0.4752874018495283
    Correct Metric Level Continuous  -  0.0941789404405269
    Correct Metric Level Total  -  0.16146148133578503
    Any Metric Level Continuous  -  0.154366969333679
    Any Metric Level Total  -  0.314709163565147
    Information gain  -  0.17220141675205886
    
LIBROSA PLP:
    Evaluated SMC files:  217
    F-measure  -  0.36040334389897183
    Cemgil  -  0.25879786947907013
    Cemgil Best Metric Level  -  0.3460833291269496
    Goto  -  0.004608294930875576
    P-score  -  0.4488204416551971
    Correct Metric Level Continuous  -  0.040525625098979993
    Correct Metric Level Total  -  0.07118076522526735
    Any Metric Level Continuous  -  0.10955325500823147
    Any Metric Level Total  -  0.22119886721031556
    Information gain  -  0.1310494842008602
    
Pulse Encoder:
    Evaluated files:  217
    Dataset: SMC. Model: encoder_pulse
    F-measure  -  0.2097717529856688
    Cemgil  -  0.15022224724956224
    Cemgil Best Metric Level  -  0.2225815508381969
    Goto  -  0.0
    P-score  -  0.38501174768784086
    Correct Metric Level Continuous  -  0.025771999847509802
    Correct Metric Level Total  -  0.041591698628542895
    Any Metric Level Continuous  -  0.06847819105070349
    Any Metric Level Total  -  0.1259529556015043
    Information gain  -  0.11455730935567168
"""

"""NMED-T
Dataset: NMED-T. Model: madmom-online
    F-measure  -  0.09207002176428304
    Cemgil  -  0.06358836010717463
    Cemgil Best Metric Level  -  0.149382111182732
    Goto  -  0.0
    P-score  -  0.2878235200381705
    Correct Metric Level Continuous  -  0.04427280904517669
    Correct Metric Level Total  -  0.10482769567358714
    Any Metric Level Continuous  -  0.11455258538947197
    Any Metric Level Total  -  0.27979634556752236
    Information gain  -  0.3055873032511921

Dataset: NMED-T. Model: madmom-offline
    F-measure  -  0.10938345889168484
    Cemgil  -  0.07816051571940201
    Cemgil Best Metric Level  -  0.15757304836041838
    Goto  -  0.01
    P-score  -  0.3156745058748141
    Correct Metric Level Continuous  -  0.04325957110969123
    Correct Metric Level Total  -  0.12781308979241532
    Any Metric Level Continuous  -  0.13219506568655262
    Any Metric Level Total  -  0.3220591370413497
    Information gain  -  0.3002316584535828
    
Dataset: NMED-T. Model: librosa
    F-measure  -  0.2773828264991582
    Cemgil  -  0.14850433373293243
    Cemgil Best Metric Level  -  0.22691168538009837
    Goto  -  0.02
    P-score  -  0.38714185938663304
    Correct Metric Level Continuous  -  0.10626915707266545
    Correct Metric Level Total  -  0.1945438267701179
    Any Metric Level Continuous  -  0.2963072804308731
    Any Metric Level Total  -  0.47334066042976436
    Information gain  -  0.32231180926798597
    
Dataset: NMED-T. Model: librosa_plp
    F-measure  -  0.30545036595431463
    Cemgil  -  0.1590955721079782
    Cemgil Best Metric Level  -  0.23272262899119825
    Goto  -  0.0
    P-score  -  0.38249562067516385
    Correct Metric Level Continuous  -  0.016948432394615395
    Correct Metric Level Total  -  0.037293182807693004
    Any Metric Level Continuous  -  0.0506947051965463
    Any Metric Level Total  -  0.1249686492087155
    Information gain  -  0.22188037428128932
    
Dataset: NMED-T. Model: encoder_pulse
    F-measure  -  0.2580012005735798
    Cemgil  -  0.18470568539678672
    Cemgil Best Metric Level  -  0.32390516712929607
    Goto  -  0.0
    P-score  -  0.3452898333734537
    Correct Metric Level Continuous  -  0.004803684782303306
    Correct Metric Level Total  -  0.0127922287371038
    Any Metric Level Continuous  -  0.09485936179762175
    Any Metric Level Total  -  0.19732960458559443
    Information gain  -  0.18147815375986376  
"""

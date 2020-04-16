import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import mode


def chromagram(filename, sr, hop_length, display):
    """
    read audio files and use CQT algorithm to convert them into chromagram

    :param filename: file path
    :param sr: sampling rate
    :param hop_length: number of samples between consecutive chroma frames (frame size)
    :param display: whether to display the chromagram
    :return: chromagram
    """

    # reads audio file
    y, sr = librosa.load(filename, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # harmonic content extraction
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Constant Q Transform
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
    # STFT
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    # 功率归一化音色谱
    # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

    if display is True:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(chromagram, sr=sr, x_axis="frames",  y_axis="chroma")
        plt.title("Chroma Features")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return chromagram, duration


def chordgram(chromagram, sr, display):
    """

    :param chromagram: chromagram
    :param chromagram: framerate
    :param display: whether to display the chromagram
    :return: chordgram
    """
    frames = chromagram.shape[1]

    # initialize
    template = chord_template()
    chords = list(template.keys())
    chroma_vectors = np.transpose(chromagram)  # matrix transpose

    # chordgram
    chordgram = []

    for n in np.arange(frames):
        cr = chroma_vectors[n]
        sims = []

        for chord in chords:
            t = template[chord]
            # calculate cos sim, add weight
            if chord == "NC":
                sim = cossim(cr, t) * 0.8
            else:
                sim = cossim(cr, t)
            sims += [sim]
        chordgram += [sims]
    chordgram = np.transpose(chordgram)

    if display is True:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(chordgram, sr=sr, x_axis="frames")
        plt.title("Chordgram")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return chordgram


def chord_template():
    """
    Template for 24 chords(contain major and minor) and an empty chord
    :return: a dictionary of chord template
    """
    template = {}
    majors = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    minors = ["Cm", "Dbm", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "Abm", "Am", "Bbm", "Bm"]

    # template for C, Cm and Csus4
    tc = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    tcm = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    shifted = 0

    for chord in majors:
        template[chord] = tc[12 - shifted:] + tc[:12 - shifted]
        shifted += 1

    for chord in minors:
        template[chord] = tcm[12 - shifted:] + tcm[:12 - shifted]
        shifted += 1

    # template for empty chord
    tnc = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    template["NC"] = tnc

    return template


def cossim(u, v):
    """
    :param u: non-negative vector u
    :param v: non-negative vector v
    :return: the cosine similarity between u and v
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def chord_sequence(chordgram):
    """
    :param H: chordgram H
    :return: a sequence of chords
    """
    template = chord_template()

    chords = list(template.keys())

    frames = chordgram.shape[1]
    chordgram = np.transpose(chordgram)
    chordset = []

    for n in np.arange(frames):
        index = np.argmax(chordgram[n])
        if chordgram[n][index] == 0.0:
            chord = "NC"
        else:
            chord = chords[index]

        chordset += [chord]

    return chordset


def to_string(input):
    string = ""
    for r in input:
        if r == "NC":
            string += " -"
        else:
            string += " " + r
    return string


def match_time(chordset, duration):
    gap = duration / len(chordset)
    point = 0
    time = []
    matched = {}
    for i in range(len(chordset)):
        time.append(point)
        matched[point] = chordset[i]
        point += gap
        point = round(point, 1)
    print(time)
    print(matched)
    return time


def smoothing(s):
    """
    :param s: sequence s
    :return: mode filter for sequence s
    """
    w = 2
    news = [0] * len(s)
    for k in np.arange(w, len(s) - w):
        m = mode([s[i] for i in range(k - w // 2, k + w // 2 + 1)])[0][0]
        news[k] = m
    return news


def smoothed_chordgram(chordgram, sr, display):
    """
    :param H: chordgram
    :return: chordgram after filtering
    """
    chords = chordgram.shape[0]
    smoothed = []

    for n in np.arange(chords):
        smoothed += [smoothing(chordgram[n])]

    smoothed = np.array(smoothed)
    if display == True:
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(smoothed, sr=sr, x_axis="frames")
        plt.title("Chordgram")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return smoothed


if __name__ == '__main__':
    filename = 'audiosamples/PianoChordSet.wav'
    sr = 44100
    hop_length = 4096
    display = False

    c, d = chromagram(filename, sr, hop_length, display)
    chordgram = chordgram(c, sr, display)
    chordset = chord_sequence(chordgram)

    # print(chordset)
    print(to_string(chordset))

    print("")
    print("After Smoothing")
    smoothed = smoothed_chordgram(chordgram, sr, display)
    chordset = chord_sequence(smoothed)
    # print(chordset)
    print(len(chordset))
    print(to_string(chordset))


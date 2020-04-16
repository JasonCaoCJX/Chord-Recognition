import os
import filetype
import wave
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


def read_audio(filename):
    """identifying file type and read the file"""

    # get the type of file
    kind = filetype.guess(filename)
    if kind is None:
        print('Cannot guess file type!')
        return

    # convert to wav if needed
    if kind.extension != 'wav':
        filename = convert(filename, kind.extension)

    # get information of the file
    wav = wave.open(filename, "rb")
    params = wav.getparams()
    channels, sampwidth, framerate = params[:3]
    wav.close()

    # get tempo of the file
    y, sr = librosa.core.load(filename, sr=None)
    frames = len(y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    tempo = int(round(tempo))

    if tempo > 140:
        tempo = int(round(tempo / 2))
    elif 30 < tempo <= 60:
        tempo = int(round(tempo * 2))
    elif tempo <= 30:
        tempo = int(round(tempo * 4))

    # save the information to info array
    info = [filename, channels, sampwidth, framerate, frames, tempo]

    # return parameter
    return info


def convert(filename, type):
    """convert mp3 to wav"""
    if type == 'mp3':
        song = AudioSegment.from_mp3(filename)
        song.export('audio/now.wav', format='wav')
    elif type == 'm4a':
        os.system("ffmpeg -i " + filename + " " + 'temp/now.wav')
    filename = 'audio/now.wav'
    return filename


def oscillogram(filename):
    """draw oscillogram"""

    # 打开WAV文档
    f = wave.open(filename, "rb")
    # 读取格式信息
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()
    # 将波形数据转换为数组
    wave_data = np.frombuffer(str_data, dtype=np.short)
    if nchannels == 2:
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        time = np.arange(0, nframes) * (1.0 / framerate)
        # 绘制波形
        plt.subplot(211)
        plt.plot(time, wave_data[0])
        plt.subplot(212)
        plt.plot(time, wave_data[1], c="g")
        plt.xlabel("time (seconds)")
        plt.show()
    elif nchannels == 1:
        wave_data.shape = -1, 1
        wave_data = wave_data.T
        time = np.arange(0, nframes) * (1.0 / framerate)
        # 绘制波形
        plt.subplot(211)
        plt.plot(time, wave_data[0])
        plt.xlabel("time (seconds)")
        plt.show()


def show_info(info):
    """show the informatian of audio"""

    print('文件名: %s\n'
          '通道数: %s（双声道）\n'
          '采样频率: %s Hz\n'
          '总帧数: %s\n'
          '速度（BPM）: %d\n'
          % (info[0], info[2], info[3], info[4], info[5]))
    print("-------------------------------------------------------")


if __name__ == '__main__':
    filename = 'audiosamples/PianoChordSet.wav'
    info = read_audio(filename)
    show_info(info)
    # oscillogram(filename)

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy import signal

# サンプリング周波数
sample_rate = 48000

def main():
    # 音の長さ（秒）
    duration = 10.0

    # サンプル数
    num_samples = int(sample_rate * duration)

    # 時間軸を作成
    time = np.linspace(0, duration, num_samples, endpoint=False)
    #waveform = np.zeros(num_samples)

    w1 = np.empty(0)
    dt=0.01
    dw = np.zeros(int(sample_rate * dt))
    # 音の周波数（Hz）
    f1 = 17000.0
    f2 = 20000.0
    for i in range(30):
        dw=makedw(f1,f2)
        w1=np.append(w1, dw)
    fp=np.array([f1,f2])#通過域端周波数[Hz]※ベクトル
    fs=np.array([f1-1000,f2+1000]) #阻止域端周波数[Hz]※ベクトル
    gpass=3
    gstop=40
    w1 = bandpass(w1, sample_rate, fp, fs, gpass, gstop)
    
    w2 = np.empty(0)
    f1 = 18000.0
    f2 = 19000.0
    for i in range(100):
        dw=makedw(f1,f2)
        w2=np.append(w2, dw)
    fp=np.array([f1,f2])#通過域端周波数[Hz]※ベクトル
    fs=np.array([f1-1000,f2+1000]) #阻止域端周波数[Hz]※ベクトル
    gpass=3
    gstop=40
    w2 = bandpass(w2, sample_rate, fp, fs, gpass, gstop)

    w3 = np.empty(0)
    f1 = 17000.0
    f2 = 20000.0
    for i in range(30):
        dw = makedw(f1, f2)
        w3 = np.append(w3, dw)
    fp = np.array([f1, f2])  # 通過域端周波数[Hz]※ベクトル
    fs = np.array([f1 - 1000, f2 + 1000])  # 阻止域端周波数[Hz]※ベクトル
    gpass = 3 #通過域端最大損失[dB]
    gstop = 40 #阻止域端最小損失[dB]
    w3 = bandpass(w3, sample_rate, fp, fs, gpass, gstop)

    waveform = np.empty(0)
    #waveform = np.append(waveform, w1)
    waveform = np.append(waveform, w2)
    waveform = np.append(waveform, w3)



    #正規化
    waveform /= max(waveform)
    print(waveform[:10])

    #16ビットに変換
    waveform = np.array(waveform * 32767, dtype=np.int16)
    plotWave(waveform)
    print(waveform[:10])

    # WAVファイルを作成
    with wave.open('la_sound.wav', 'w') as file:
        file.setnchannels(1)  # モノラル
        file.setsampwidth(2)  # 16ビット
        file.setframerate(sample_rate)
        file.writeframes(waveform.tobytes())

def makedw(f1,f2):
    dt = 0.014
    time = np.linspace(0, dt, int(sample_rate * dt), endpoint=False)
    #time=np.random.rand(int(sample_rate*dt))
    dw = np.zeros(int(sample_rate * dt))
    for i in range(int(f1), int(f2), 1):
        dw += makeWave(i, time)
    return dw
def makeWave(frequency, time):
    return 0.001 * np.sin(2 * np.pi * frequency * time)
def bandpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す
def plotWave(freq):
    plt.plot(freq)
    plt.show()


if __name__ == "__main__":
    main()

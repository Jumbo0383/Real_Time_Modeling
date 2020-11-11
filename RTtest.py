# pyaudioをインポート
import pyaudio
import numpy as np
import wave
import time
from train_wavenet import WaveNet
from keras import backend as K

"""
buff_sizeを上げると高速化になるがメモリ次第
"""

"""
OSError: [Errno -9985] Device unavailable
上記のエラー出た場合exit()で直るかも 
"""

"""
アプリ化の場合は
inputで使用するデバイスを変更
"""

"""
平均: 0.48
"""

K.clear_session()
yml_filename = "WaveNet1_SD_config.yml"     # パラメータを読み込むymlファイル
yml_filename = "WaveNet_SD_config.yml"     # パラメータを読み込むymlファイル
load_path = "./WaveNet1_SD.h5"         # モデリングで用いるネットワークのデータ
load_path = "./model_000009.h5"         # モデリングで用いるネットワークのデータ
load_path = "./model_000720.h5"         # input 2048

set_float = "float16"
# ネットワークの読み込み
K.set_floatx(set_float)
wavenet = WaveNet(yml_filename, set_float)
wavenet.Model_Loader("WaveNet", "test", load_path)
wavenet.Load_Model(load_path)

" パラメータ "
device_num = 12                     # 使用する入出力デバイス
buff_size = 512                     # バッファサイズ
data_format = pyaudio.paInt16       # ビットレート (16bit)
ch_num = 1                          # チャンネル数 (モノラル)
fs = 44100                          # サンプリング周波数 [Hz]
input_timesteps = 4096              # ネットワークの入力サイズ
input_timesteps = 2048              # ネットワークの入力サイズ
batch_size = 4
threshold = 0.1         # ディストーションのクリッピング閾値


" デバイス情報を表示 "
def Show_Device_Information(p):
    # ホストAPI数
    hostAPICount = p.get_host_api_count()
    print( "Host API Count = " + str(hostAPICount) )
    
    # ホストAPIの情報を列挙
    for cnt in range(0, hostAPICount):
        print()
        print( p.get_host_api_info_by_index(cnt) )
        
    # ASIOデバイスの情報を列挙
    asioInfo = p.get_host_api_info_by_type(pyaudio.paASIO)
    print( "\nASIO Device Count = " + str(asioInfo.get("deviceCount")) )
    for cnt in range(0, asioInfo.get("deviceCount")):
        print( p.get_device_info_by_host_api_device_index(asioInfo.get("index"), cnt) )


# クリッピングによる単純なディストーション
def Simple_Distortion(data_in, threshold=0.1):
    data_out = np.where(data_in < threshold, data_in, threshold)
    data_out = np.where(data_out > -threshold, data_out, -threshold)
    return data_out

times = []
# 信号処理を行う
def Signal_Processing(stream, buff_hold, buff_size):
    # バッファデータの読み込み
    time_start = time.time()
    data_in = stream.read(buff_size, exception_on_overflow=False)

    # バイト列から[-1~1] の実数値に正規化
    data_in = np.fromstring(data_in, dtype = np.int16)
    data_in = (data_in/ 0x7fff).astype(np.float32)
    
    # buff_holdの更新
    buff_hold[:-buff_size] = buff_hold[buff_size:]
    buff_hold[-buff_size:] = data_in

    # 変換処理
    print("=================")
    data_out = Simple_Distortion(data_in)
#    data_out = wavenet.RealTime_Modeling(buff_hold, buff_size, batch_size)
    # [-1~1]の実数値からバイト列に再変換
    data_out = (data_out * 0x7fff).astype(np.int16)
    data_out = data_out.tostring()

    stream.write(data_out) 

    time_end= time.time()
    print("time_all", time_end - time_start)
    times.append(time_end - time_start)
    return buff_hold


# PyAudioクラスのインスタンスを取得
p = pyaudio.PyAudio()
Show_Device_Information(p)

" 処理部分 "
# streamのオープン
stream = p.open(format=data_format,                             # ビットレート
                rate = fs,                                      # サンプリング周波数
                channels = ch_num,                              # チャンネル数
                frames_per_buffer = buff_size,                  # バッファサイズ
                input=True,  input_device_index  = device_num,  # 入力情報
                output=True, output_device_index = device_num)  # 出力情報


# streamの開始
stream.start_stream()

# データを保持するバッファ
buff_hold = np.zeros( input_timesteps, dtype=np.float32 )


print("\n")
print("============")
print("= Running. =")
print("============")

try:
    while True: # 信号処理
        buff_hold = Signal_Processing(stream, buff_hold, buff_size)
except KeyboardInterrupt:   # Ctrl+C が押された場合に処理を終了
    print("done.")

print("mean", np.mean(times))
# streamの終了
stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()    
    
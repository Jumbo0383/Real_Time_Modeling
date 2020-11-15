# pyaudioをインポート
import pyaudio
import numpy as np
import wave
import time
from train_wavenet import WaveNet
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

"""
OSError: [Errno -9985] Device unavailable
=> exit()
"""

"""
train_wavenet
Loss_Functions - o
Network_Model - o
Function_Wavenet - o
cqt_ops_tf - o
config_WaveNet
"""

K.clear_session()

yml_filename = "WaveNet1_SD_config.yml"     # パラメータを読み込むymlファイル
yml_filename = "WaveNet_SD_config.yml"     # パラメータを読み込むymlファイル
load_path = "./WaveNet_SD.h5"         # モデリングで用いるネットワークのデータ
load_path = "./model_000720.h5"         # input 2048
load_path = "./model_000009.h5"         # モデリングで用いるネットワークのデータ
load_path = "./model_000030.h5"         # input 2048
load_path = "./model_000183.h5"         # input 2048

# Load WaveNet
set_float = "float32"
K.set_floatx(set_float)
wavenet = WaveNet(yml_filename)
wavenet.Model_Loader("WaveNet", "test", load_path)
wavenet.Load_Model(load_path)

" Parameters "
device_num = 12                     # audio device num
buff_size = 512                     # buff size
data_format = pyaudio.paInt16       # bir rate (16bit)
ch_num = 1                          # channel num (default: mono)
fs = 44100                          # sampling frequency [Hz]
input_timesteps = 2048              # Network input size
threshold = 0.1                     # distortion clipping value
buff_hold = np.zeros( input_timesteps, dtype=np.float32 )


def Show_Device_Information(p):
    # Host API num
    hostAPICount = p.get_host_api_count()
    print( "Host API Count = " + str(hostAPICount) )
    
    # show host API information
    for cnt in range(0, hostAPICount):
        print()
        print( p.get_host_api_info_by_index(cnt) )
        
    # show ASIO device information
    asioInfo = p.get_host_api_info_by_type(pyaudio.paASIO)
    print( "\nASIO Device Count = " + str(asioInfo.get("deviceCount")) )
    for cnt in range(0, asioInfo.get("deviceCount")):
        print( p.get_device_info_by_host_api_device_index(asioInfo.get("index"), cnt) )


# Simple distortion by clipping
def Simple_Distortion(data_in, threshold=0.1):
    data_out = np.where(data_in < threshold, data_in, threshold)
    data_out = np.where(data_out > -threshold, data_out, -threshold)
    return data_out

def Signal_Processing(stream, buff_hold, buff_size):
    # read buffer data
    time_start = time.time()
    data_in = stream.read(buff_size, exception_on_overflow=False)

    # byte array to real number[-1~1]
    data_in = np.fromstring(data_in, dtype = np.int16)
    data_in = (data_in/ 0x7fff).astype(np.float32)
    
    # update buff_hold
    buff_hold[:-buff_size] = buff_hold[buff_size:]
    buff_hold[-buff_size:] = data_in

    # Translation
#    data_out = Simple_Distortion(data_in)
    data_out = wavenet.RealTime_Modeling(buff_hold, buff_size)
    # real number to byte array[-1~1]
    data_out = (data_out * 0x7fff).astype(np.int16)
    data_out = data_out.tostring()

    stream.write(data_out) 

    time_end = time.time()
    print("time_all", time_end - time_start)
    return buff_hold


p = pyaudio.PyAudio()
Show_Device_Information(p)

# open stream
stream = p.open(format=data_format,                             # bit rate
                rate = fs,                                      # samppling frequency
                channels = ch_num,                              # channel num
                frames_per_buffer = buff_size,                  # buff size
                input=True,  input_device_index  = device_num,  # input information
                output=True, output_device_index = device_num)  # output information


# start stream
stream.start_stream()

print("\n")
print("============")
print("= Running. =")
print("============")

try:
    while True: # Signal processing
        buff_hold = Signal_Processing(stream, buff_hold, buff_size)
except KeyboardInterrupt:   # Ends the process if Ctrl+C is pressed
    print("done.")

# close stream
stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()    
    
# pyaudioをインポート
import pyaudio
import numpy as np
import wave
import time
from train_wavenet import WaveNet
from keras import backend as K





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




# PyAudioクラスのインスタンスを取得
p = pyaudio.PyAudio()
Show_Device_Information(p)

# close PyAudio
p.terminate()    
    
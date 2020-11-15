# pyaudioをインポート
import os
import sys
import pyaudio
import numpy as np
import wave
import time
from train_wavenet import WaveNet
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from Function_WaveNet import parse_args
import yaml
"""
OSError: [Errno -9985] Device unavailable
=> exit()
"""

K.clear_session()


dirs = os.listdir("./")

while(True):
    yml_filename = input("yml file>>>")
    if(yml_filename in dirs) and (yml_filename[-4:] == ".yml"):
        break

while(True):
    h5_path = input("h5 file>>>")
    if(h5_path in dirs) and (h5_path[-3:] == ".h5"):
        break

# yml_filename = "WaveNet1_SD.yml"     # パラメータを読み込むymlファイル
# h5_path = "./WaveNet1_SD.h5"         # input 2048


# Simple distortion by clipping
def Simple_Distortion(data_in, threshold=0.1):
    data_out = np.where(data_in < threshold, data_in, threshold)
    data_out = np.where(data_out > -threshold, data_out, -threshold)
    return data_out


class RT_modeling():
    def __init__(self):
        self.config_filename = yml_filename
        args = parse_args(self.config_filename)
        with open(args.config_file) as fp:
            config = yaml.safe_load(fp)

        " Parameters "
        self.input_timesteps = config["input_timesteps"]      # Network input size
        self.buff_size       = config["output_timesteps"]     # buff size
        self.fs              = config["Fs"]                   # sampling frequency [Hz]
        self.set_float       = config["set_float"]            # betowrk data format
        self.data_format = pyaudio.paInt16                    # bir rate (16bit)
        self.ch_num = 1                                       # channel num (default: mono)
        self.buff_hold = np.zeros( self.input_timesteps, dtype=np.float32 )

        # Load WaveNet
        K.set_floatx(self.set_float)
        self.wavenet = WaveNet(yml_filename)
        self.wavenet.Model_Loader("test", h5_path)
        self.wavenet.Load_Model(h5_path)
        
        self.p = pyaudio.PyAudio()
        self.index_list = []
        self.Show_Device_Information()
        
        device_num = 12
        # get device num
        while(True):
            device_num = int( input("device index>>>") )
            if( device_num in self.index_list ):
                break
            else:
                print("input is wrong.")
        
        # open stream
        self.stream = self.p.open(format = self.data_format, rate = self.fs, channels = self.ch_num,
                                  frames_per_buffer = self.buff_size,                  # buff size
                                 input=True,  input_device_index  = device_num,        # input information
                                 output=True, output_device_index = device_num)        # output information
        # start stream
        self.stream.start_stream()
        




    def Show_Device_Information(self):
        # Host API num
        hostAPICount = self.p.get_host_api_count()
        print( "Host API Count = " + str(hostAPICount) )
        
        # show host API information
        for cnt in range(0, hostAPICount):
            print()
            print( self.p.get_host_api_info_by_index(cnt) )
            self.index_list.append( self.p.get_host_api_info_by_index(cnt)["index"] )
            
        # show ASIO device information
        asioInfo = self.p.get_host_api_info_by_type(pyaudio.paASIO)
        print( "\nASIO Device Count = " + str(asioInfo.get("deviceCount")) )
        for cnt in range(0, asioInfo.get("deviceCount")):
            print( self.p.get_device_info_by_host_api_device_index(asioInfo.get("index"), cnt) )
            self.index_list.append( self.p.get_device_info_by_host_api_device_index(asioInfo.get("index"), cnt)["index"] )


    def Signal_Processing(self):
        # read buffer data
        time_start = time.time()
        # "exception_on_overflow=False" is must
        data_in = self.stream.read(self.buff_size, exception_on_overflow=False)
    
        # byte array to real number[-1~1]
        data_in = np.fromstring(data_in, dtype = np.int16)
        data_in = (data_in/ 0x7fff).astype(np.float32)
        
        # update buff_hold
        self.buff_hold[:-self.buff_size] = self.buff_hold[self.buff_size:]
        self.buff_hold[-self.buff_size:] = data_in
    
        # Translation
    #    data_out = Simple_Distortion(data_in)
        data_out = self.wavenet.RealTime_Modeling(self.buff_hold, self.buff_size)
        # real number to byte array[-1~1]
        data_out = (data_out * 0x7fff).astype(np.int16)
        data_out = data_out.tostring()
    
        self.stream.write(data_out) 
    
        time_end = time.time()
        print("time_all", time_end - time_start)
    
    def close_all(self):
        # close stream
        self.stream.stop_stream()
        self.stream.close()
        # close PyAudio
        self.p.terminate()    

    
    

if __name__ == "__main__":
    RT = RT_modeling()
    
    print("\n")
    print("============")
    print("= Running. =")
    print("============")
    
    
    try:
        while True: # Signal processing
            RT.Signal_Processing()
    except KeyboardInterrupt:   # Ends the process if Ctrl+C is pressed
        print("done.")

    RT.close_all()
    
        
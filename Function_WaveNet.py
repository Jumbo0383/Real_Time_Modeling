import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import yaml
from argparse import ArgumentParser
import os
import shutil
import wave
from inspect import currentframe
import time
from keras.callbacks import Callback

config_filename = "config_WaveNet.yml"


# ymlファイルの読み込み
def parse_args(config_filename):
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default=config_filename,
        help="configuration file (*.yml)")
    return parser.parse_args()

args = parse_args(config_filename)
with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
Fs = config["Fs"]


class MyCallback(Callback) :
    def __init__(self, epochs, p_wave, p_freq, wave_name, freq_name):
        self.epochs = epochs
        self.p_wave = p_wave
        self.p_freq = p_freq
        self.wave_name = wave_name
        self.freq_name = freq_name

    def on_epoch_begin(self, epoch, logs=None):
        # 損失関数の重み調整
        if epoch == self.epochs:
            self.model.compile(loss=self.model.loss,
                               loss_weights = [self.p_wave,self.p_freq],   
                               optimizer=self.model.optimizer)
            self.model.metrics_names[1] = self.wave_name
            self.model.metrics_names[2] = self.freq_name
            print("loss weight update")

# 高域強調
# x:入力波形, timesteps:出力波形のサイズ, a:プリエンファシス係数
def Pre_Emphasis(x, timesteps, batch_size, a):    
    z = K.concatenate( (K.zeros( (batch_size,1) ), x), 1 )
    x_1 = tf.slice(z, [0,0], [batch_size, timesteps])
    return x - a*x_1


# 波形の読み込み
def load_wave(wave_file):
    with wave.open(wave_file, "r") as w:
        buf = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (buf / 0x7fff).astype(np.float32)


# 波形の保存
def save_wave(buf, wave_file):
    _buf = (buf * 0x7fff).astype(np.int16)
    with wave.open(wave_file, "w") as w:
        w.setparams((1, 2, Fs, len(_buf), "NONE", "not compressed"))
        w.writeframes(_buf)
        
# one-hotベクトルの作成
def one_hot_vec(label, num, length):
    a = [label] * length
    return np.identity(num)[a]


# フレームをoutput_timesteps分スライドさせて推論を進めていく (入力音xの場合)
def sliding_window(x, window, slide, num=0):
    n_slide = (len(x) - window) // slide
    remain = (len(x) - window) % slide
    if num == 0:    # 入力音xの場合
        clopped = x[:-remain]
        shape = (n_slide + 1, window)
        strides = (slide * 4, 4)
    else:           # 補助特徴量cの場合
        clopped = np.array(x[:-remain,:], "float32")
        shape = (n_slide + 1, window, num)
        strides = (slide * 4*num, 4*num, 4)        
    return as_strided(clopped, shape=shape, strides=strides)


# 学習データのパディング
def Pad_post(x, input_size):
    pad = input_size - np.mod( np.shape(x)[0], input_size)
    pad = np.zeros( (pad,1) )
    return np.concatenate( ( x, pad ) )


# 両方向のパディング
def Pad_Pre_Post(x, pre=0, post=0, label=0, num=0):
    if num == 0:
        pre  = np.zeros( (pre,1), np.float32 )
        post = np.zeros( (post,1), np.float32 )
    else:
        pre  = one_hot_vec(label, num, pre)
        post = one_hot_vec(label, num, post)

    return np.concatenate( ( pre, x, post ) )


def Pad_Input(x, input_timesteps, output_timesteps, batch_size, label=0, num=0):
    block_size = output_timesteps * batch_size
    prepad = input_timesteps - output_timesteps
    postpad = len(x) % block_size
    return Pad_Pre_Post(x, prepad, postpad, label, num)


def MSE(x1, x2):
    return  ( (x1 - x2)**2 ).mean()

def esr(y_true, y_pred):
    m = np.sum( (y_true-y_pred)**2 )
    n = np.sum( ( y_true**2 ) ) 
    return m/n    


# グラフの表示(2つ)
def plot_graph2(history, metrics_name, filename):    
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

    # 損失関数のグラフ
    title_loss = "Loss(" + str( min( history.history["loss"] ) ) + ")"
    axL.plot(history.history["loss"], linewidth=2)
    axL.plot(history.history["val_loss"], linewidth=2)
    axL.set_title(title_loss)
    axL.set_xlabel('epochs')
    axL.set_ylabel('loss')
    axL.grid(True)
    
    # 評価関数のグラフ  
    title_val  = metrics_name + "(" + str( min( history.history[metrics_name] ) )  + ")"
    axR.plot(history.history[metrics_name], linewidth=2)
    axR.plot(history.history["val_" + metrics_name], linewidth=2)
    axR.set_title( title_val )
    axR.set_xlabel('epochs')
    axR.set_ylabel('MSE')
    axR.grid(True)
    fig.show()  

    fig.savefig(filename)


def plot_2data(reference, modeling, dir_name, title_, range_, save=True, fs=44100):
    L = len(reference)
    t = np.arange(0, (L/fs), 1/fs)

    fig, ax = plt.subplots()
    ax.plot(t, reference, color="b", label="reference")
    ax.plot(t, modeling, color="g", label="modeling")
    plt.xlim([ range_[0], range_[1] ])
    plt.xlabel("time(secs)")
    plt.ylabel("amplitude")
    plt.title(title_)
    if save == True:
        plt.savefig( "./result/"+dir_name+"/fig/"+title_+".png" )
    
def size_match(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[0:len(data1)]
    elif len(data1) > len(data2):
        data1 = data1[0:len(data2)]
    return data1, data2
    
    
# configファイルの中身を表示    
def config_write(config_file, list_name=[], avoid_word="data"):
    with open(config_file, 'r') as yml:
        config = yaml.load(yml)    
    key = config.keys()    
    for v in key:
        if ( (avoid_word in v) == False ):
            list_name += v + ":" + str(config[v]) + "\n"
    return "".join(list_name)
        
        

# ディレクトリの作成
def make_dir(path,path_add="", num=0):
    flag = False
    cnt = 0
    dirs = os.listdir("./"+path_add)
    # 名前がかぶっていればディレクトリ名を変更
    while(1):
        # ディレクトリを探索し同じ名前のものがあれば、
        for i in range( len(dirs) ):
            if path == dirs[i]:
                flag = True
        if cnt == 0 and flag == True:
            path = str(path) + "(1)"
        elif flag == True:
            path = path[0:-3] + "(" + str(cnt) + ")"
        if flag == False:
            break
        flag = False
        cnt += 1
    if cnt == 0 or cnt == 1:
        os.mkdir(path_add + "/" + path)
        return path
    
    new_path =  path[0:-3] + "(" + str(cnt-2) + ")"
    # プログラムが最後まで回っていない場合は作成したディレクトリを削除して新しく作り直す
    dir_flag = "fig" in os.listdir(path_add + "/" +new_path)
    if dir_flag == False:
        shutil.rmtree( path_add + "/" +new_path )
        os.mkdir(path_add + "/" +new_path)
    elif dir_flag == True:
        os.mkdir(path_add + "/" +path)
    return new_path
        

# h5ファイルの探索
def get_h5_file(path):
    dirs = os.listdir(path) # ディレクトリ内のファイル名の取得
    # ディレクトリ内のh5ファイルを探索
    for i in range( len(dirs) ):
        filename = dirs[i]
        if filename[-3:] == ".h5":
            h5_file = filename   
    return h5_file


# ディレクトリ名の設定
def make_dir_name(wave_loss, p_wave, freq_loss, p_freq, type_, FX, num_mel_bins,N): 
    dir_name = ""
    if str(wave_loss) != "None":
        dir_name = dir_name + str(wave_loss) + str(p_wave) + "_"
    if str(freq_loss) != "None":
        dir_name = dir_name + str(freq_loss)+str(p_freq) + "_" + type_ + "_"
    dir_name = dir_name + "WaveNet3_" + FX + "_e1000" + "_N" + str(N) 
    return dir_name


# 実験条件をtxtデータに保存
def text_list(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return "".join(names.get(id(arg),'???')+': '+repr(arg) + "\n" for arg in args)


# 実験条件をtxtデータに保存
def time_calc(state, Time=0, title_="", times="秒"):
    if   state == "start":
        Time = time.time()
    elif state == "end":
        Time = (time.time() - Time)
        if(Time>=60):
            Time /= 60
            times = "分"
            if(Time >= 60):
                Time /= 60
                times = "時間"            
        print( title_ + str(Time) + times )
    return Time


# 損失関数の値をrxtファイルに書き込む
def loss_write(filename, loss_name, losses):
    losses = np.array(losses)
    # テキストデータ
    text = ""
    for i in range( len(loss_name) ):
        text += loss_name[i] + "\n"
        for j in range( len(losses[0]) ):
            text += "test" + str(j+1) + ":" + str(losses[i,j]) + "\n"
    # データの書き込み
    with open(filename, "w", encoding = "utf_8") as f:
        f.write(text)



    
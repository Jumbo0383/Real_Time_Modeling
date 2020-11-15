import yaml
from keras.losses import mean_squared_error, mean_absolute_error
from keras import backend as K
import tensorflow as tf
from Function_WaveNet import Pre_Emphasis, parse_args
import numpy as np

from cqt_ops_tf import CQT

config_filename = "config_WaveNet.yml"

args = parse_args(config_filename)
with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
" ===================================================================== "
Fs               = config["Fs"]                  # サンプリング周波数
set_float        = config["set_float"]           # 内部処理のフォーマット
N                = config["N"]                   # DFT点数
frame_length     = config["frame_length"]        # STFTのフレーム長
frame_step       = config["frame_step"]          # STFTのフレームシフト長
p                = config["pre_emphasis"]        # プリエンファシス係数
power            = config["power"]               # パワースペクトログラムかどうか
mel              = config["mel"]                 # メル周波数スペクトログラムかどうか
#MFS パラメータ
lower_hz_mel     = config["lower_hz_mel"]        # 最小周波数
upper_hz_mel     = config["upper_hz_mel"]        # 最大周波数
num_mel_bins_mel = config["num_mel_bins_mel"]    # melビン数
# CQT パラメータ
fmin_cqt         = config["fmin_cqt"]           # 最小周波数
nhop_cqt         = config["nhop_cqt"]           # フレームシフト長
n_bin_cqt        = config["n_bin_cqt"]          # 周波数ビン数
fratio_cqt       = config["fratio_cqt"]         # 1オクターブの分割数
spectrogram_type = config["spectrogram_type"]   # スペクトログラムのタイプ ["STFT" or "CQT"]

" ===================================================================== "

# 距離尺度
# 一般化KLダイバージェン
def I_divergence(p, q):
    p = p + K.epsilon()
    q = q + K.epsilon()
    return p * K.log(p/q) - p + q

# 板倉斎藤距離
def IS_divergence0(p, q):
    p = p + K.epsilon()
    q = q + K.epsilon()
    return (p/q) - K.log(p/q) - 1
        
def Mylog(y_true, y_pred, threshold, max_dat):
    div = y_true/y_pred
    div = tf.where( div > threshold, div, max_dat )
    return K.log(div)
    
def IS_divergence(y_true, y_pred):
    eps = 0.001
    threshold = K.constant(eps)
    # STFT後の出力サイズ
    batch_size, frame_size, freq_size = y_true.shape

    max_dat = K.constant( np.reshape( [eps] * (batch_size * frame_size * freq_size), (batch_size, frame_size, freq_size) ) )

    y_pred = tf.where( y_pred>threshold, y_pred, max_dat )
    y_true = tf.where( y_true>threshold, y_true, max_dat )

    div = y_true/y_pred

    return (div) - K.log(div) - 1

# Error to Signal Ratio
def esr0(y_true, y_pred):
    m = K.sum( K.square( K.abs(y_true - y_pred) ), 1 ) + K.epsilon()
    n = K.sum( K.square( K.abs( y_true ) ), 1 ) + K.epsilon()               
    return m/n    

def esr(y_true, y_pred):
    # バッチサイズ
    batch_size = y_true.shape[0]
    eps = 0.0001
    threshold = K.constant(eps)
    max_dat = K.constant( np.reshape( [eps] * batch_size, (batch_size) ) )

    m = K.sum( K.square( K.abs(y_true - y_pred) ), 1 ) 
    n = K.sum( K.square( K.abs( y_true ) ), 1 )

    m = tf.where( m>threshold, m, max_dat )
    max_dat = K.constant( np.reshape( [1] * batch_size, (batch_size) ) )
    n = tf.where( n>threshold, n, max_dat )
    return m/n    

    
def dc(y_true, y_pred, timesteps):
    m = K.square( K.sum( (y_true - y_pred), 1 ) / N ) + K.epsilon()
    n = K.square( K.sum( y_true, 1 ) ) / N + K.epsilon()               
    return m/n    


class Loss_wave:
    def __init__(self, name, wave_loss):
        self.__name__ = name
        self.wave_loss  = wave_loss         # 波形の損失関数
    
    def __call__(self, y_true, y_pred):
        # 目的音とモデリング音 (バッチ数, timesteps)
        self.batch_size = tf.shape(y_true)[0]  # バッチサイズを取得
        _, self.timesteps, _ = y_pred.shape    # タイムステップを取得
        y_true_ = K.reshape (y_true, (self.batch_size, self.timesteps) )
        y_pred_ = K.reshape (y_pred, (self.batch_size, self.timesteps) )

        # 波形の損失関数
        if   self.wave_loss == "None":
            wave = K.variable(0)
        elif self.wave_loss == "MAE":
            wave = mean_absolute_error(y_true_, y_pred_)
        elif self.wave_loss == "MSE":
            wave = mean_squared_error(y_true_, y_pred_)
        elif self.wave_loss == "MSE_PE":
            y_true_ = Pre_Emphasis(y_true_, self.timesteps, self.batch_size, p)
            y_pred_ = Pre_Emphasis(y_pred_, self.timesteps, self.batch_size, p)
            wave = mean_squared_error(y_true_, y_pred_)
        elif self.wave_loss == "ESR":
            wave = esr0(y_true_, y_pred_)
        elif self.wave_loss == "ESR_PE":
            y_true_ = Pre_Emphasis(y_true_, self.timesteps, self.batch_size, p)
            y_pred_ = Pre_Emphasis(y_pred_, self.timesteps, self.batch_size, p)
            wave = esr0(y_true_, y_pred_)
        else:
            raise ValueError("wave_loss value Error")

        wave = K.mean(wave) 
                
        return wave
    
    
    
class Loss_freq:
    def __init__(self, name, timesteps, batch_size, freq_loss):
        self.__name__ = name
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.freq_loss  = freq_loss

        if spectrogram_type == "CQT":
            self.cqt = CQT(self.batch_size, self.timesteps, Fs, fmin=fmin_cqt, nhop=nhop_cqt, nfreq=n_bin_cqt, fratio=fratio_cqt)
            _ = self.cqt.calc_kernel_matrix_librosa()

        
    def __call__(self, y_true, y_pred):
        # 目的音とモデリング音 (バッチ数, timesteps)
        y_true_ = K.reshape (y_true, (self.batch_size, self.timesteps) )
        y_pred_ = K.reshape (y_pred, (self.batch_size, self.timesteps) )
        
        if set_float == "float16":
            y_true_ = K.cast(y_true_, tf.float32)
            y_pred_ = K.cast(y_pred_, tf.float32)
            
        " スペクトログラムの計算 "
        if ( self.freq_loss != None ) and ( spectrogram_type == "CQT" ):
            true_freq, _ =  self.cqt.calc_cqt(y_true_) 
            pred_freq, _ =  self.cqt.calc_cqt(y_pred_) 
            # 振幅スペクトログラム
            true_freq = K.abs(true_freq)
            pred_freq = K.abs(pred_freq)
        
        elif self.freq_loss != None and ( spectrogram_type == "STFT" ):
            # 目標音とモデリング音の振幅スペクトログラムの取得
            true_freq = K.abs( tf.signal.stft(signals=y_true_, frame_length=frame_length, frame_step=frame_step, fft_length=N, pad_end=True) )
            pred_freq = K.abs( tf.signal.stft(signals=y_pred_, frame_length=frame_length, frame_step=frame_step, fft_length=N, pad_end=True) )
            # パワースペクトログラム
            if power == True:
                true_freq  = K.square( true_freq)
                pred_freq  = K.square( pred_freq )                
            # メル周波数スペクトログラム(MFS)
            if mel == True:
                # メルフィルタバンク
                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins_mel, int(N/2+1), Fs, lower_hz_mel, upper_hz_mel)
                # 目的音源のMFS            
                true_freq = tf.tensordot(true_freq, linear_to_mel_weight_matrix, 1)
                true_freq.set_shape(true_freq.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
                # モデリング音源のMFS            
                pred_freq = tf.tensordot(pred_freq, linear_to_mel_weight_matrix, 1)
                pred_freq.set_shape(pred_freq.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        else:
            raise ValueError("spectrogram type value Error")

        # 周波数の損失関数
        if   self.freq_loss == "None":
            freq = K.variable(0)
        elif self.freq_loss == "MAE":
            freq = mean_absolute_error(true_freq, pred_freq)
        elif self.freq_loss == "MSE":
            freq = mean_squared_error(true_freq, pred_freq)
        elif self.freq_loss == "KL":
            freq = I_divergence(true_freq, pred_freq) 
        elif self.freq_loss == "IS":
            freq = IS_divergence(true_freq, pred_freq) 
        else:
            raise ValueError("freq_loss value Error")
        freq = K.mean(freq)
        

        if set_float == "float16":
            freq = K.cast( freq, dtype=tf.float16)
        return freq 
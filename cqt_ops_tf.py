# -*- coding: utf-8 -*-
"""
(仮)実装
"""

"""
CQTの高速版
・ 最高の1オクターブのCQT行列を計算
・ サンプリング周波数を1/2にダウンサンプリング
この2点を用いることで計算時間とメモリの使用量を削減

[参考論文]
Schoerkhuber, Christian, and Anssi Klapuri.
"Constant-Q transform toolbox for music processing."
7th Sound and Music Computing Conference, Barcelona, Spain. 2010.
"""

import scipy
import numpy as np
import tensorflow as tf
from keras import backend as K
import librosa


K.clear_session()


# 対数周波数ビン数を計算
def get_num_freq(fmin, fmax, fratio):
    return int(round(scipy.log2(float(fmax) / fmin) / fratio)) + 1

# 各対数周波数ビンに対応する周波数を計算
def get_freqs(fmin, nfreq, fratio):
    return fmin * (2 ** (scipy.arange(nfreq) * fratio))

# 波形のダウンサンプリング[tensorflow]
def downsampling_tf(batch_size, input_timesteps, conversion_rate,x,fs, taps=511, scale=True):
    """
    batch_size: バッチサイズ, input_timesteps: 入力のタイムステップ数
    conversion_rate: 出力後のサンプリング周波数の倍率(fs/conversion_rate)
    x: 入力データ, fs: 入力のサンプリング周波数, taps: lowpassフィルタのタップ数
    """
    " FIRフィルタの設定(low-pass) "
    nyqF = fs/2.0                                                     # 変換前のナイキスト周波数
    cF = (fs/conversion_rate/2.0) / nyqF                              # カットオフ周波数を設定
    lowpass = scipy.signal.firwin(taps, cF)                           # LPFを用意
    lowpass = K.constant(lowpass.reshape(lowpass.shape[0], 1,1))      # Tensorに変換
    print(lowpass)
    " Low-passフィルタを適用 "
    x = K.reshape( x, (batch_size, input_timesteps, 1) )
    x_filtered = tf.nn.conv1d(x, lowpass, stride=[1, 1, 1], padding='SAME') 
    " 余りを削除 "
    cut_num = input_timesteps % conversion_rate    # 削除する数
    if cut_num != 0:
        x_filtered = x_filtered[:,:-cut_num]

    " データの間引き(ダウンサンプリング) "
    x_downsample = K.reshape( x_filtered, (batch_size, -1, conversion_rate,1) )
    x_downsample = x_downsample[:, :,0,:]   
    x_downsample = K.reshape(x_downsample, (batch_size, -1) )
    
    if scale:   # スケール調整
        x_downsample /= np.sqrt(1/conversion_rate)
    # ダウンサンプリング後の波形, 出力のサンプリング周波数
    return x_downsample, int(fs/conversion_rate) 






### パラメータの初期値
two_pi_j = 2*np.pi * 1j
fmin_default = 60                    # 最小周波数
fmax_default = 8000                  # 最大周波数
fratio_default = 12                  # fratio (1オクターブの分割数)
q_rate_def = 1.0                     # qrate
window_fn = scipy.hanning

class CQT(object):
    def __init__(self, batch_size, sig_len, fs, nhop=0.01, q_rate = q_rate_def, fmin = fmin_default, fmax = None, nfreq=None,
                 fratio = fratio_default, win = window_fn, filter_scale=1, scale=True):
        # CQTのパラメータ
        self.batch_size = batch_size      # バッチサイズ
        self.sig_len = sig_len            # 入力信号の長さ
        self.fs = fs                      # サンプリング周波数
        self.fs_org = self.fs             # 元のサンプリング周波数
        self.win = win                    # 窓関数
        self.filter_scale = filter_scale  # フィルタのスケール調整
        self.scale = scale                # 全体のスケール調整
        
        " フレームシフトサイズを決定(移動サイズがサンプル点数か，サンプルの割合のどちらかになる) "
        if   nhop < 1:      # nhopが1未満の場合は，時間窓がfs*nhopになり
            self.nhop = int(round(nhop * fs))
        elif nhop >= 1:     # nhopが1以上の場合は，時間窓がnhopになり
            self.nhop = nhop
        else:               # nhopがマイナスの場合はエラーを返す
            raise ValueError("nhop must be plus value.")
        self.nframe = int( self.sig_len / self.nhop )  # 時間フレーム数        

        " 周波数ビンの数を決定 "
        if   (fmax != None) and (nfreq == None):     # 最大周波数を引数に入力している場合
            self.nfreq = int( get_num_freq( fmin, fmax, (1/fratio) ) ) 
        elif (fmax == None) and (nfreq != None):     # 周波数ビンの数を引数に入力している場合
            self.nfreq = nfreq
        elif ( (fmax != None) and (nfreq != None) ) and ( (fmax == None) and (nfreq == None) ):     # どちらも引数にしている場合はエラーを返す
            raise ValueError("Which is correct, frequency bin or the number?")


        self.Q = int((1. / ((2 ** ( 1/fratio) ) - 1)) * q_rate)    # Q値     
        self.freqs = get_freqs( fmin, self.nfreq, (1/fratio) )     # 各ビンの周波数[Hz]        
        self.bins_per_octave = fratio                              # 1オクターブの分割数
        self.freqs_n = self.freqs[-self.bins_per_octave:]          # n番目の周波数のオクターブビン(初期値は最高オクターブ)
        self.max_octave = int(self.nfreq / self.bins_per_octave)   # 総オクターブ数
        self.n_octave = self.max_octave                            # n番目のオクターブ
        # N  > max(N_k)
        self.fftLen = int(2 ** (np.ceil(scipy.log2(int(float(fs * self.Q) / self.freqs_n[0])))))    # カーネル行列と入力信号のFFTサイズの計算
        self.h_fftLen = int( self.fftLen / 2 )
        
        # 計算する周波数帯の表示
        freqs_min = int( np.ceil( self.freqs[0] ) )
        freqs_max = int( np.floor( self.freqs[-1] ) )
        print("Frequency Range [%d-%d](Hz)" % (freqs_min, freqs_max) )
        # パラメータの初期値
        self.nhop_org = self.nhop
        self.sig_len_org = self.sig_len
        self.freqs_n_org = self.freqs_n


    # カーネル行列のパラメータの更新
    def update_params(self):
        self.nhop = int(self.nhop / 2)          # ダウンサンプル(1/2)に伴いホップサイズを半分にする
        self.sig_len = int(self.sig_len/2 )     # ダウンサンプル(1/2)に伴い信号長を半分にする
        self.n_octave -= 1                      # 1つ下のオクターブ
        self.freqs_n = self.freqs[ int(self.bins_per_octave*(self.n_octave-1)) : int(self.bins_per_octave*(self.n_octave)) ]  # 最大周波数のオクターブビン

    # パラメータを初期値に戻す
    def reset_params(self):
        self.nhop     = self.nhop_org
        self.sig_len  = self.sig_len_org
        self.n_octave = self.max_octave
        self.freqs_n  = self.freqs_n_org
        self.fs       = self.fs_org
                
    # 信号の前後のパディング
    def signal_padding(self, sig):
        sig_pad = K.concatenate( (K.zeros( ( self.batch_size, int(self.h_fftLen)) ), sig), 1)
        sig_pad = K.concatenate( (sig_pad, K.zeros( ( self.batch_size, int(self.h_fftLen)) )) , 1 )
        return sig_pad

    



    " カーネル行列の作成(librosa) "
    def calc_kernel_matrix_librosa(self, spThresh = 0.01):
        self.N_1 = self.freqs_n[0]
        # カーネル行列の取得
        print("N_1", self.N_1)
        kernelMatrix, lengths = librosa.filters.constant_q(sr=self.fs, fmin=self.N_1, n_bins=self.bins_per_octave, bins_per_octave=self.bins_per_octave, filter_scale=1, pad_fft=True)
        # 正規化
        kernelMatrix *= lengths[:, np.newaxis]# / float(self.fftLen) # 全てのカーネルで振幅がそろうようにする                    
        # フーリエ変換
        kernelMatrix = np.fft.fft(kernelMatrix, n=self.fftLen, axis=1)[:, :self.fftLen//2+1]
        # 小さい値をゼロにする                    
        kernelMatrix[abs(kernelMatrix) <= spThresh] = 0
        ### 複素共役にする
        kernelMatrix = (kernelMatrix.conjugate() / self.fftLen).T
        # complex64型のTensorに変換してカーネル行列を返す
        self.kernelMatrix = K.constant( kernelMatrix, tf.complex64 )
        return kernelMatrix


    " カーネル行列と入力信号それぞれにFFTを行ったものの行列積を行いCQT行列を作成 "
    def calc_cqt(self, sig):
        self.reset_params()
        # 1オクターブずつCQT行列を求める
        for i in range(self.max_octave): 
            # 前後をゼロ埋めした信号を作成
            sig_pad = self.signal_padding(sig)
            # 入力信号STFTの計算
            sig_stft = tf.signal.stft(signals=sig_pad, frame_length=self.fftLen, frame_step=self.nhop, window_fn=None, pad_end=False)
            # stft行列とカーネル行列の行列積で1オクターブ分のCQTを計算
            cqt_1oct = tf.matmul( sig_stft, self.kernelMatrix*np.sqrt(2**i) )            
            # 全体の行列に結合
            if i == 0:
                CQT = cqt_1oct
            else:
                CQT = K.concatenate( (cqt_1oct, CQT), 2 )
            # ダウンサンプリング
            sig, self.fs = downsampling_tf(self.batch_size, self.sig_len, 2, sig, self.fs, taps=16-1, scale=self.scale)
            # CQTのパラメータ情報を更新
            self.update_params()
        # スケール調整
        if self.scale:
            lengths = librosa.filters.constant_q_lengths(
                self.fs_org,
                self.freqs[0],
                n_bins=self.nfreq,
                bins_per_octave=self.bins_per_octave,
                window=self.win,
                filter_scale=self.filter_scale )
            CQT /= np.sqrt(lengths[np.newaxis, :])
            
        return CQT, self.freqs
    

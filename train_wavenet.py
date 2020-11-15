# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import shutil
import yaml
import datetime
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from Network_model import WaveNet_model, LSTM_model
from Loss_Functions import Loss_wave, Loss_freq
from keras.optimizers import Adam
from Data_Loader import DataLoader
import tensorflow as tf
from keras import backend as K
from Function_WaveNet import MyCallback, plot_graph2, one_hot_vec, Pad_Input, sliding_window, \
                             plot_2data, size_match, MSE, esr, config_write, make_dir, time_calc, \
                             save_wave, loss_write, parse_args




dir_name = "waveNet1_SD"
yml_filename = "config_WaveNet.yml"

K.clear_session()

class WaveNet():
    def __init__(self, config_filename):

        self.config_filename = config_filename
        args = parse_args(self.config_filename)
        with open(args.config_file) as fp:
            config = yaml.safe_load(fp)

        # パラメータの読み込み
        self.input_timesteps     = config["input_timesteps"]     # 入力タイムステップ
        self.output_timesteps    = config["output_timesteps"]    # 出力タイムステップ
        self.set_float           = config["set_float"]           # 内部処理の設定
        self.Network_type        = config["Network_type"]        # 使用するネットワーク
        self.layer_num           = config["layer_num"]           # Residual Block数
        self.channel             = config["channel"]             # チャンネル数
        self.filter_size         = config["filter_size"]         # CNNフィルタサイズ
        self.gate                = config["gate"]                # ゲートの活性化関数
        self.LSTM_ch             = config["LSTM_ch"]             # チャンネル数(LSTM)
        self.GPU_use             = config["GPU_use"]             # LSTM層とcuDNNLSTM層の切り替え
        self.batch_size          = config["batch_size"]          # バッチサイズ
        self.max_epochs          = config["max_epochs"]          # エポック数
        self.patience            = config["patience"]            # 打ち止めエポック数
        self.p_wave              = config["p_wave"]              # ハイパーパラメータ(波形)
        self.p_freq              = config["p_freq"]              # ハイパーパラメータ(周波数)
        self.wave_loss           = config["wave_loss"]           # 損失関数(波形領域)
        self.freq_loss           = config["freq_loss"]           # 損失関数(周波数領域)
        self.FX                  = config["FX"]                  # モデリングするエフェクタ
        self.param_save          = config["param_save"]          # ネットワークパラメータの保存
        self.loss_shift          = config["loss_shift"]          # 学習途中でlossの重みを変更
        self.shift_epoch         = config["shift_epoch"]
        self.single_conv         = config["single_conv"]
        K.set_floatx( self.set_float )
        self.plot_x_range        = config["plot_x_range"]
        # 学習の初期は位相の反転を防ぐために波形の損失のみ用いる
        self.lambda_wave_start = 1
        if self.loss_shift == True:
            self.lambda_freq_start = 0
        else:    
            self.lambda_freq_start =  self.p_freq
        self.wave_name = "Loss_wave"
        self.freq_name = "Loss_freq"
        if self.freq_loss != "None":
            self.time_freq_loss = True
        else:
            self.time_freq_loss = False
        # 補助特徴量を使用するか判断する
        self.FX_num = len( self.FX )   # モデリングを行うエフェクタの数
        if self.FX_num != 1:
            self.Condition = True
        else:
            self.Condition = False
        
        dir_flag = "result" in os.listdir("./")
        if dir_flag == False:
            os.mkdir("./result")
            
    

        
    # 訓練の中で一番精度のいいモデルの保存
    def save_best_model(self, cp_dir):    
        # 最良のモデルの読み込み    
        path = os.listdir(cp_dir)
        best_model = path[len(path)-1]        
        # best_modelのパス
        self.best_path = cp_dir + "/" + best_model
        shutil.copy(self.best_path, "./result/"+ self.dir_name)      # 最良のモデルのコピー


    # Networkモデルの作成
    def Model_Loader(self, types=None, load_path=None):
        # 損失関数
        self.Loss_wave = Loss_wave(self.wave_name, self.wave_loss)
        self.Loss_freq = Loss_freq(self.freq_name, self.input_timesteps, self.batch_size, self.freq_loss)

        if type != "test":
            # 最適化法
            self.optimizer = Adam()
            # ネットワークの構築
            if self.Network_type == "wavenet":
                models = WaveNet_model(self.input_timesteps, self.FX_num, self.Condition, self.layer_num, self.channel, self.filter_size, self.gate, self.single_conv, self.time_freq_loss)
                self.Network_model = models.build_WaveNet()
            elif self.Network_type == "LSTM":
                models = LSTM_model(self.input_timesteps, self.LSTM_ch, self.GPU_use)
                self.Network_model = models.build_LSTM()
            else:
                raise ValueError("network_model value Error")
    
            if self.time_freq_loss == True:
                self.Network_model.compile(loss=[self.Loss_wave, self.Loss_freq],
                                           loss_weights = [self.lambda_wave_start, self.lambda_freq_start],
                                           optimizer = self.optimizer)
                # 表示する損失関数名の変更        
                self.Network_model.metrics_names[1] = self.wave_name
                self.Network_model.metrics_names[2] = self.freq_name
            else:
                self.Network_model.compile(loss=self.Loss_wave,
                                           optimizer = self.optimizer)
            # ディレクトリの作成
    
            self.timestamp = str( "_({:%m-%d})".format( datetime.datetime.now()  ) )
            
            self.dir_name = make_dir(dir_name+self.timestamp, "result")
    
            # パラメータの保存
            if self.param_save == True:
                param_file = "./result/" + self.dir_name +  "/Network_Parameter.txt"
                with open(param_file, "w") as fp:
                    self.Network_model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
        # テストデータのモデリングの場合ネットワークのファイルを読み込む
        if types == "test":
            self.best_path = load_path

        
    def train(self):
        self.data_loader = DataLoader(self.config_filename, self.FX, self.FX_num, self.input_timesteps, self.batch_size, self.Condition, self.time_freq_loss)
        " データセットの読み込み "
        train_dataflow, iter_num_t = self.data_loader.load_batch("clean_data", "reference_data")
        val_dataflow, iter_num_v   = self.data_loader.load_batch("clean_data_val", "reference_data_val")
        print("Data Loded")
        
        if self.time_freq_loss == True:
            monitor = self.Network_model.metrics_names[1]
        else:
            monitor = self.Network_model.metrics_names[0]
        #  訓練
        train_start = time_calc("start")           # 訓練の計算時間    
        timestamp = datetime.datetime.now()        # 時間のデータを取得
        # 結果が良い時のみ, パラメータを保存する.
        cp_dir = "./checkpoint/" + self.dir_name + "_({:%m-%d})".format(timestamp)
        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir)
        cp_filepath = os.path.join(cp_dir, "model_{epoch:06d}.h5")
        # コールバックの設定
        cb_mc = ModelCheckpoint( filepath=cp_filepath, monitor="val_"+monitor, period=1, save_best_only=True )  # 精度の良いモデルを保存
        cb_es = EarlyStopping( monitor="val_"+monitor, patience=self.patience )                                 # 早期打ち止め
        cb_tb = TensorBoard( log_dir="./tensorboard/{:%Y%m%d_%H%M%S}".format(timestamp) )                       # Tensorboardに保存
        cb_lw = MyCallback(self.shift_epoch, self.p_wave, self.p_freq, self.wave_name, self.freq_name)          # 学習の途中で損失関数の重みの変更
        
        if self.loss_shift == True:
            callbacks=[cb_mc, cb_es, cb_tb, cb_lw]
        else:
            callbacks=[cb_mc, cb_es, cb_tb]
            
        history = self.Network_model.fit_generator(generator=train_dataflow,                    # 学習データ
                                                   steps_per_epoch=iter_num_t,                  # 学習データのイタレーション数
                                                   validation_data=val_dataflow,                # 検証データ
                                                   validation_steps=iter_num_v,                 # 検証データイタレーション数
                                                   epochs=self.max_epochs,                      # 最大エポック数
                                                   callbacks=callbacks)                         # コールバック
        time_calc("end",train_start)  # 訓練時間の計算

        # 最良のモデルを保存        
        self.save_best_model(cp_dir)
            
        # 学習曲線のグラフの表示と保存
        img_name = "./result/" + self.dir_name + "/Learning_Curve.jpg"
        plot_graph2(history, monitor, img_name)
    
    
    def Modeling(self, clean, reference):
        # ディレクトリdirがなければディレクトリを作成
        test_num = np.shape(clean)[1]
        dir_flag = "fig" in os.listdir("./result/"+self.dir_name)
        if dir_flag == False:
            os.mkdir("./result/"+self.dir_name+"/fig/")
            
        loss_mse = np.zeros( int(test_num) )
        loss_esr = np.zeros( int(test_num) )
            
        for label in range(self.FX_num):
            # label番目のエフェクタを保存するディレクトリの作成
            dir_ = self.dir_name + "/" + self.FX[label]
            dir_= make_dir( dir_, "result" ) 
            for i in range(test_num):
                # 出力するファイル名        
                filename_out = "./result/" + self.dir_name + "/" + self.FX[label] + "/" + self.dir_name + "_test" + str(i+1) + self.FX[label] + ".wav"
                # ファイルのパディング
                clean_in = np.reshape(clean[label][i], (-1,1) )
                padded = Pad_Input( clean_in, self.input_timesteps, self.output_timesteps, self.batch_size)     # Condition 
                # x: Clean音源
                x = sliding_window(padded, self.input_timesteps, self.output_timesteps)
                x = x[:, :, np.newaxis] 
    
                # 補助特徴量をone-hot labelで作成            
                if self.Condition == True:
                    condition_ = one_hot_vec( label, self.FX_num, np.shape(clean[label][i])[0] )
                    # c: 補助特徴量  Condition
                    padded_c = Pad_Input( condition_, self.input_timesteps, self.output_timesteps, self.batch_size, label, self.FX_num)
                    c = sliding_window(padded_c, self.input_timesteps, self.output_timesteps, self.FX_num)
                    # predict関数の入力をxとcのセットにする
                    x = [x,c]

                pred_start = time_calc("start")           # 訓練の計算時間    
                modeling = self.Network_model.predict(x, batch_size=self.batch_size)
                if self.time_freq_loss == True:
                    modeling = modeling[0][:, -self.output_timesteps:, :].reshape(-1)[:len(clean_in)]
                else:
                    modeling = modeling[:, -self.output_timesteps:, :].reshape(-1)[:len(clean_in)]
                # 推論時間の計算
                print("\nTest%d[%s]" % (i+1, self.FX[label]) )
                time_calc("end",pred_start, "推論時間:")

                # 目的音とモデリング音波形の表示
                plot_2data(reference[label][i], modeling, self.dir_name, "test"+str(i+1), self.plot_x_range)
                    
                # 誤差の計算        
                reference_ = reference[label][i].reshape(-1) # reference_: 目的音
                reference_, modeling = size_match( reference_, modeling )
                loss_mse[i] = MSE(reference_, modeling)
                loss_esr[i] = esr(reference_, modeling)
                print( "Loss(MSE): %f" %( loss_mse[i] ) )
                print( "Loss(ESR): %f" %( loss_esr[i] ) )
        
                save_wave( modeling, filename_out)  # 波形の書き込み
            # 誤差をtxtファイルに書き込む
            filename = "./result/" + self.dir_name + "/loss.txt"
            loss_write(filename, ["MSE", "ESR"], [loss_mse, loss_esr])



        
    " テストデータのモデリング "
    def Predict(self):
        # 最良のネットワークパラメータの読み込み
        self.Network_model = load_model(self.best_path, custom_objects={self.wave_name: self.Loss_wave,
                                                                        self.freq_name: self.Loss_freq})
#       # テストデータの読み込み
        clean, reference = self.data_loader.load_test_data()
        self.Modeling(clean, reference)

        # 実験条件をtxtファイルに書き込む 
        filename_text = "./result/" + self.dir_name + "/実験条件.txt"
        f = open(filename_text, 'w')    
        # 書き込むファイルの内容
        config_list = config_write(self.config_filename)
        f.writelines(config_list)

    # リアルタイムモデリング用の変換処理
    def Load_Model(self, model_path):
        self.Network_model = load_model(model_path, custom_objects={self.wave_name: self.Loss_wave,
                                                                    self.freq_name: self.Loss_freq})         
        if self.time_freq_loss == True:
            self.pred = K.function( [self.Network_model.input], [self.Network_model.output[0]] )
        else:
            self.pred = K.function( [self.Network_model.input], [self.Network_model.output] )

    
    
    def RealTime_Modeling(self, data_in, buff_size):
        # ファイルのパディング
        data_in = np.reshape(data_in, (1,-1,1) )
        modeling = self.pred([data_in])
        if self.time_freq_loss == True:
            return modeling[0].reshape(-1)[-buff_size:]
        else:
            return modeling.reshape(-1)[-buff_size:]


if __name__ == "__main__":
    network = WaveNet(yml_filename)
    # Networkの読み込み
    network.Model_Loader()
    network.train()
    network.Predict()


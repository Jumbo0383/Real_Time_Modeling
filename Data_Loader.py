# -*- coding: utf-8 -*-
import numpy as np
import yaml
from Function_WaveNet import one_hot_vec, Pad_post, parse_args, load_wave
                            

class DataLoader():
    def __init__(self, config_filename, FX, FX_num, input_timesteps, batch_size, Condition, time_freq_loss):
        self.config_filename = config_filename
        self.FX = FX
        self.FX_num = FX_num
        self.input_timesteps = input_timesteps
        self.batch_size = batch_size
        self.Condition = Condition
        self.time_freq_loss = time_freq_loss
        

    " 学習データのファイル名の読み込み "
    def read_filename(self, filename):
        args = parse_args(self.config_filename)
        with open(args.config_file) as fp:
            config = yaml.safe_load(fp)
        for i in range( self.FX_num ):
            if i == 0:
                data = config[ filename + self.FX[i] ]
            else:
                data = np.concatenate( ( data, config[ filename + self.FX[i] ] ), axis=0)
        return data


    " データセットの作成 "
    def make_dataset(self, clean_data, reference_data, number=None):    
        # 学習データの読み込み
        clean_dataset     = [ ( load_wave(_[0]).reshape(-1, 1) ) for _ in clean_data ]
        reference_dataset = [ ( load_wave(_[0]).reshape(-1, 1) ) for _ in reference_data ]
    
        for i in range( len(clean_data) ):
            # データ数をinput_sizeの倍数にパディング
            clean_dataset[i]     = Pad_post(clean_dataset[i], self.input_timesteps)
            reference_dataset[i] = Pad_post(reference_dataset[i], self.input_timesteps)
            if self.time_freq_loss == True:
                z = [ clean_dataset[i], reference_dataset[i], reference_dataset[i] ]
            else:
                z = [ clean_dataset[i], reference_dataset[i] ]

            if self.Condition == True:  # 補助特徴量のラベルを作成
                conditions = one_hot_vec( i, self.FX_num, np.shape(clean_dataset[i])[0] )
                
            if i == 0:
                dataset = z
                if self.Condition == True:
                    condition_label = conditions
            else:   # 複数のエフェクタをモデリングする場合のみ(i<=1)
                dataset = np.concatenate( [dataset, z],1 )
                condition_label = np.concatenate( [condition_label, conditions] )
        dataset = np.reshape( dataset, (1,len(z),-1,1) )
        
        if self.Condition == True:  # データセットの組とラベルを出力
            return dataset, condition_label
        else:
            return dataset
    
    
    " サンプルをランダムに取得する "
    def random_clop(self, x, y1, y2=None, c=None):
        max_offset = len(x) - self.input_timesteps
        offsets = np.random.randint(max_offset, size=self.batch_size)
        batch_x = np.stack((x[offset:offset+self.input_timesteps] for offset in offsets))
        batch_y1 = np.stack((y1[offset:offset+self.input_timesteps] for offset in offsets))
        if self.time_freq_loss == True:
            batch_y2 = np.stack((y2[offset:offset+self.input_timesteps] for offset in offsets))
            batchs = [batch_y1, batch_y2]
        else:
            batchs = batch_y1
        if self.Condition == True:
            batch_c = np.stack((c[offset:offset+self.input_timesteps,:] for offset in offsets))
            return [batch_x,batch_c], batchs
        else:
            return batch_x, batchs

        
    " バッチサイズ分のフレームデータの確保 "
    def flow(self, dataset):
        if self.Condition == True:
            c = dataset[1]
            dataset = dataset[0]
        n_data = len(dataset)
        while True:
            i = np.random.randint(n_data)
            if self.time_freq_loss == True:
                x, y1, y2 = dataset[i]
            else:
                x, y1 = dataset[i]
            if self.Condition == True:
                if self.time_freq_loss == True:
                    yield self.random_clop(x, y1, y2, c)
                else:
                    yield self.random_clop(x, y1, c)
            else:
                if self.time_freq_loss == True:
                    yield self.random_clop(x, y1, y2)
                else:
                    yield self.random_clop(x, y1)

    
    " 学習データの読み込み "
    def load_batch(self, filename_clean, filename_reference):
        clean_data          = self.read_filename( filename_clean + "_" )
        reference_data      = self.read_filename( filename_reference + "_" )
        # データセットの組を作成
        dataset = self.make_dataset(clean_data, reference_data)   
        dataflow = self.flow(dataset)
        # 1epoch当たりのデータ数
        if self.Condition == True:
            data_num = np.shape(dataset[0])[2]
        else:
            data_num = np.shape(dataset)[2]
        # 1 epochに必要なイタレーション数
        iter_num = np.ceil( data_num / (self.batch_size * self.input_timesteps) )

        return dataflow, iter_num
    
    
    "　テストデータを読み込む　"
    def load_test_data(self):
        # テストデータの読み込み
        test_data_x = self.read_filename("test_data_x_")
        test_data_y = self.read_filename("test_data_y_")
        clean     = [ ( load_wave(_[0]) ) for _ in test_data_x ]
        reference = [ ( load_wave(_[0]) ) for _ in test_data_y ]
                
        test_num = len(test_data_x) / self.FX_num
        clean_data, reference_data = [], []
        for i in range(self.FX_num):
            idx_start = int(test_num*i)
            idx_end = int(test_num*(i+1))
            clean_data.append(clean[ idx_start:idx_end ])
            reference_data.append(reference[ idx_start:idx_end ])
                
        return clean_data, reference_data
        
        
            
            
            
        
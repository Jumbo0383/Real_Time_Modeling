# -*- coding: utf-8 -*-
from keras.layers import Conv1D, Input, Add, Multiply, Activation, Concatenate, BatchNormalization
from keras.models import Model

from keras.models import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense

class WaveNet_model():
    def __init__(self, input_timesteps, FX_num, Condition, layer_num, ch, filter_size, gate, single_conv, time_freq_loss):
        self.input_timesteps = input_timesteps
        self.FX_num = FX_num
        self.Condition = Condition
        self.layer_num = layer_num
        self.ch = ch
        self.filter_size = filter_size
        self.gate = gate
        self.single_conv = single_conv
        self.time_freq_loss = time_freq_loss

        # ゲートの活性化関数
        if   self.gate == "gated":
            self.activation1 = "tanh"
            self.activation2 = "sigmoid"
        elif self.gate == "sigmoid":
            self.activation1 = "linear"
            self.activation2 = "sigmoid"
        elif self.gate == "softgated":
            self.activation1 = "softsign"
            self.activation2 = "softsign"
        elif self.gate == "gated2":
            self.activation1 = "relu"
            self.activation2 = "sigmoid"
        else:
            raise ValueError("gate value Error")

        # 畳み込み層の数とdilationパターンが一致するようにする
        if self.layer_num == 10:
            self.d_layer = 10
        elif self.layer_num == 18:
            self.d_layer = 9
        else:
            raise ValueError("layer_num value Error")
        
    # WaveNetの構築        
    def build_WaveNet(self):
        def Gated_Activation_Function(num, dilation_rate=1, padding="same"):
            def f(z, condition=None):
                # u_{k,1}の計算
                Conv1 = Conv1D(filters=self.ch,kernel_size=self.filter_size, padding="causal",
                               dilation_rate=dilation_rate, name="".join(["u_", str(num), "1" ]))(z)  
#                Conv1 = BatchNormalization(name="".join(["BN1_", str(num)]))(Conv1)
                # u_{k,2}の計算
                if self.single_conv == False:
                    Conv2 = Conv1D(filters=self.ch,kernel_size=self.filter_size, padding="causal",
                                   dilation_rate=dilation_rate, name="".join( ["u_", str(num), "2" ] ))(z)  
#                Conv2 = BatchNormalization(name="".join(["BN2_", str(num)]))(Conv2)

                if self.Condition == True:                    
                    # 補助特徴量の1x1 conv
                    Conv1_condition = Conv1D(filters=self.ch,kernel_size=1, name="Conv_c1_"+str(num))(condition)  
                    Conv2_condition = Conv1D(filters=self.ch,kernel_size=1, name="Conv_c2_"+str(num))(condition)
                    # 補助特徴量との1x1 ConvとDC convの足し合わせ
                    Conv1 = Add(name="Add_uk1_c1_"+str(num))([Conv1, Conv1_condition])
                    Conv2 = Add(name="Add_uk2_c2_"+str(num))([Conv2, Conv2_condition])

                # v_kの計算
                Conv1 = Activation(self.activation1, name="Gate1_"+str(num))(Conv1)
                if self.single_conv == True:
                    Conv2 = Activation(self.activation2, name="Gate2_"+str(num))(Conv1) 
                else:
                    Conv2 = Activation(self.activation2, name="Gate2_"+str(num))(Conv2)        
        
                return Multiply(name="".join( ["v_", str(num) ] ))( [Conv1, Conv2] )
            return f
        
        
        def Post_Processing_Module():
            def f(z):
                # first layer
                z = Conv1D(1, 1, padding="same")(z)
                tanh_out    = Activation("tanh")(z)  
                sigmoid_out = Activation("sigmoid")(z)  
                z = Multiply()( [tanh_out, sigmoid_out] )
                # Second Layer
                z = Conv1D(1, 1, padding="same", activation="tanh")(z)
                # Linear Layer
                out = Conv1D(1, 1, padding="same", name="output")(z)
                return out
            return f
        
        def Linear_Mixer():
            def f(z):
                # 出力を波形の誤差とスペクトログラムの誤差用の2つ用意する(出力は同じ)
                output1 = Conv1D(1, 1, padding="same", use_bias=False, name="Linear_Mixer_wave")(z)
                if self.time_freq_loss == True:
                    output2 = Activation("linear",name="Linear_Mixer_freq")(output1)
                    return [output1, output2]
                else:
                    return output1
            return f
        
        # 残差ブロック
        def Residual_Block(num):
            def f(xk_1, condition=None):
                dilation_rate = 2**(num%self.d_layer)
                # ショートカット接続
                residual = xk_1 
                # ゲート付き活性化関数[v_k] 
                vk = Gated_Activation_Function(num, dilation_rate=dilation_rate, padding="causal")(xk_1, condition)
                # 畳み込み層の出力(Post-Processing Moduleの入力)[s_k]
                sk = Conv1D(self.ch, 1, padding="same", activation="relu", name= "".join( ["s_", str(num) ] ))(vk)
                # 次の層の入力(xk)
                xk = Conv1D(self.ch,1, padding="same", activation="relu", name="".join( ["x_", str(num), "_pre" ] ))(vk)
                xk = Add(name="".join( ["x_", str(num) ] ))([xk, residual])
        
                return xk, sk
            return f
        
        
        
        # Pre Processing layer
        inputs    = Input( shape=(self.input_timesteps, 1) )         # ネットワークの入力
        Input_Pre = Conv1D(self.ch,1, padding="same", name="x0")(inputs)
        if self.Condition == True:
            condition     = Input( shape=(self.input_timesteps, self.FX_num) )  # 補助特徴量の入力
            condition_Pre = Conv1D(self.ch,1, padding="same", name="c0")(condition)
            inputs=[inputs, condition]      # Modelの入力
        else:
            condition_Pre = None
           
        " Convolutional Part "    
        skip_connections = []  
        A = Input_Pre
        for i in range(0, self.layer_num):
            A, B = Residual_Block( i )( A, condition_Pre )
            skip_connections.append(B)
    
        # スキップコネクション
        skip_connections = Activation("relu")( Add(name="Skip_Connetion")(skip_connections) )
        outputs = Linear_Mixer()( skip_connections )

        model = Model(inputs=inputs, outputs=outputs)
        return model



class LSTM_model():
    def __init__(self, input_timesteps, ch, GPU_use):
        self.input_timesteps = input_timesteps
        self.ch = ch
        self.input_shape = (self.input_timesteps, 1)
        self.GPU_use = GPU_use
        
    # LSTMモデルの構築        
    def build_LSTM(self):
        inputs    = Input( shape=self.input_shape )         # ネットワークの入力
        if self.GPU_use == True:    # GPUの場合、高速なcuDNNLSTMに切り替え
            model = CuDNNLSTM(units=self.ch, input_shape=self.input_shape, return_sequences=True) (inputs)
        else:
            model = LSTM(units=self.ch, input_shape=self.input_shape, return_sequences=True)(inputs)
        model = Dense(units=1)(model)
        model = Add()([model, inputs])

        model = Model(inputs=inputs, outputs=[model, model])

        return model

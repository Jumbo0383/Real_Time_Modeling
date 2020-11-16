# Real_Time_Modeling
WaveNetを用いた歪みエフェクタのリアルタイムモデリング <br>
Intel Core i7 CPU 1.90GHz だとぎりぎりリアルタイムモデリングできました． <br>

Install version list: <br>
python==3.7.7 <br>
numpy==1.16.4 <br>
tensorflow-gpu==1.14.0 <br>
keras==2.2.4 <br>
librosa==0.6.3 <br>
pyaudio==0.2.11 <br>
pip==20.1.1 <br><br>

pyaudioインストールについて(Windows)<br>
Windowsでは標準のサウンドドライバを用いると遅延が大きくなるので，リアルタイムでのモデリングを行おうとすればASIOドライバを用いる必要があります．
しかし，通常のpip等でインストールするpyaudioはASIOに対応していないのでASIOに対応したpyaudioをインストールする必要があります．
以下がインストールの手順になります．<br>
1. pipのバージョンを確認(pip==20.1.1の場合)<br>
```
from pip._internal.utils.compatibility_tags import get_supported
get_supported()
```

2. 下記URLから1で確認したpipに対応するpyaudioの.whlファイルをダウンロード <br>
https://www.lfd.uci.edu/~gohlke/pythonlibs/ <br>

3. whlファイルをインストール <br>
例) >>> ```pip install ./保存したwhlファイルのダウンロード先/PyAudio‑0.2.11‑cpXX‑cpXX‑win_amd64.whl``` <br>

mac, LINUXではその辺最適化されているようで普通のインストールで大丈夫です．


実行方法
```
$ python RT_modeling.py
```
でリアルタイムモデリングのプログラムが動きます．
```
yml file>>> WaveNet1_SD.yml
```
で使用するymlファイルを入力(ここでは WaveNet1_SD.yml)
```
h5 file >>> WaveNet1_SD.h5
```
で使用するh5ファイルを入力(ここでは WaveNet1_SD.h5)

上記2つを入力すると以下のように使用できるオーディオデバイスが表示される．
```
Host API Count = 5

{'index': 0, 'structVersion': 1, 'type': 2, 'name': 'MME', 'deviceCount': 5, 'defaultInputDevice': 1, 'defaultOutputDevice': 3}

{'index': 1, 'structVersion': 1, 'type': 1, 'name': 'Windows DirectSound', 'deviceCount': 5, 'defaultInputDevice': 5, 'defaultOutputDevice': 7}

{'index': 2, 'structVersion': 1, 'type': 3, 'name': 'ASIO', 'deviceCount': 0, 'defaultInputDevice': -1, 'defaultOutputDevice': -1}

{'index': 3, 'structVersion': 1, 'type': 13, 'name': 'Windows WASAPI', 'deviceCount': 3, 'defaultInputDevice': 12, 'defaultOutputDevice': 11}

{'index': 4, 'structVersion': 1, 'type': 11, 'name': 'Windows WDM-KS', 'deviceCount': 16, 'defaultInputDevice': 13, 'defaultOutputDevice': 14}

ASIO Device Count = 0
```

上記のリストの中から使用するデバイスを使用
```
device index>>>
```
で使用するデバイスIDを入力

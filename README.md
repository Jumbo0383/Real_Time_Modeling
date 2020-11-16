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
'''
from pip._internal.utils.compatibility_tags import get_supported <br>
get_supported()<br>
'''

2. 下記URLから1で確認したpipに対応するpyaudioの.whlファイルをダウンロード <br>
https://www.lfd.uci.edu/~gohlke/pythonlibs/ <br>

3. whlファイルをインストール <br>
例) >>> pip install ./保存したwhlファイルのダウンロード先/PyAudio‑0.2.11‑cpXX‑cpXX‑win_amd64.whl <br>

mac, LINUXではその辺最適化されているようで普通のインストールで大丈夫です．

# PANNs AT (Audio Tagging) inference

**panns_AT_inference** provides an easy to use Python interface for audio tagging. The audio tagging models are trained from PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition: https://github.com/qiuqiangkong/audioset_tagging_cnn

This is a forked repository/project w. the Top-3 models (see References)

## Installation
PyTorch>=1.0 is required.
```
$ pip install panns_AT_inference
```

## Usage
```
$ python3 example.py
```

For example:

```
import librosa
import panns_AT_inference
from panns_AT_inference import AudioTagging, labels

audio_path = 'examples/R9_ZSCveAHg_7s.wav'
(audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(model_name=None, device='cuda')
(clipwise_output, embedding) = at.inference(audio)
```


## Results
<pre>
------ Audio tagging ------
Checkpoint path: /root/panns_data/Cnn14_mAP=0.431.pth
GPU number: 1
Speech: 0.893
Telephone bell ringing: 0.754
Inside, small room: 0.235
Telephone: 0.183
Music: 0.092
Ringtone: 0.047
Inside, large room or hall: 0.028
Alarm: 0.014
Animal: 0.009
Vehicle: 0.008
</pre>

## References
[1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

[2] 
```
@article{9229505,
  author={Kong, Qiuqiang and Cao, Yin and Iqbal, Turab and Wang, Yuxuan and Wang, Wenwu and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition}, 
  year={2020},
  volume={28},
  number={},
  pages={2880 -- 2894},
  doi={10.1109/TASLP.2020.3030497}
  }
```

[3] Official GitHub repository: https://github.com/qiuqiangkong/audioset_tagging_cnn , https://github.com/qiuqiangkong/panns_inference

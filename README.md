# Torch VGGish & YAMNet embedding models

**torch_vggish_yamnet** provides a ready-to-use PyTorch porting of AudioSet (Google) audio embedding models. The audio tagging models are trained from Models for AudioSet: A Large Scale Dataset of Audio Events: https://github.com/tensorflow/models/tree/master/research/audioset

This is a re-structured forked repository/project from ```torch_audioset``` (see References)

## Installation
PyTorch>=1.0 is required (dependecies are auto-installed).
```
pip install torch-vggish-yamnet
```

## Usage
```
from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet import vggish
from torch_vggish_yamnet.input_proc import *

# Input signal (x_in) tensor conversion & ad-hoc patching
converter = WaveformToInput()
in_tensor = converter(x_in.float(), in_sr)
in_tensor.shape

# Models init
embedding_yamnet = yamnet.yamnet(pretrained=True)
embedding_vggish = vggish.get_vggish(with_classifier=False, pretrained=True)

# Embedding (forward)
emb_yamnet, _ = embedding_yamnet(in_tensor)  # discard logits
emb_vggish = embedding_vggish(in_tensor)

emb_yamnet.shape, emb_vggish.shape
```

## References
[1] AudioSet Official site: http://g.co/audioset

[2] 
```
@inproceedings{45857,
 title	    = {Audio Set: An ontology and human-labeled dataset for audio events},
 author	    = {Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter},
 year	      = {2017},
 booktitle	= {Proc. IEEE ICASSP 2017},
 address	  = {New Orleans, LA}}
```
[4] 
```
@incollection{45611,
title	      = {CNN Architectures for Large-Scale Audio Classification},
author	    = {Shawn Hershey and Sourish Chaudhuri and Daniel P. W. Ellis and Jort F. Gemmeke and Aren Jansen and Channing Moore and Manoj Plakal and Devin Platt and Rif A. Saurous and Bryan Seybold and Malcolm Slaney and Ron Weiss and Kevin Wilson},
year	      = {2017},
URL	        = {https://arxiv.org/abs/1609.09430},
booktitle	  = {International Conference on Acoustics, Speech and Signal Processing (ICASSP)}}
```

[3] torch_audioset GitHub repository: https://github.com/w-hc/torch_audioset/tree/master

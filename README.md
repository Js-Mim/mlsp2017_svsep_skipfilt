# Singing Voice Separation via Gated Recurrent Units and Skip-Filtering connections

Support material and source code for the model described in : S.I. Mimilakis, K. Drossos, T. Virtanen, G. Schuller, "A Recurrent Encoder-Decoder Approach With Skip-Filtering Connections For Monaural Singing Voice Separation", accepted for poster presentation at the 2017 IEEE International Workshop on Machine Learning for Signal Processing, September 25–28, 2017, Tokyo, Japan.

Please use the above citation if you find any of the code useful.

Listening Examples :  https://js-mim.github.io/mlsp2017_svsep_skipfilt/

### Requirements   :
- numpy            :  numpy==1.11.3
- SciPy            :  scipy==0.18.1
- Theano           :  Theano==0.9.0.dev2
- Keras            :  Keras==1.2.2
- ASP-Toolkit      :  https://github.com/Js-Mim/ASP
- Trained Models   :  https://zenodo.org/record/840388       [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.840388.svg)](https://doi.org/10.5281/zenodo.840388)


### Usage          :
- Grab ASP-Toolkit and place it inside the cloned folder.
- Download the trained models via the above link. Unzip the downloaded file and place its content inside the folder "trainedModels".
- Use testGDAE.py for estimating the waveforms of singing voice and background music: run testGDAE.py 2 *or* run testGDAE.py 3 *or* run testGDAE.py 4

### Acknowledgements :
The major part of the research leading to these results has received funding from the European Union's H2020 Framework Programme (H2020-MSCA-ITN-2014) under grant agreement no 642685 MacSeNet. Konstantinos Drossos was partially funded from the European Union's H2020 Framework Programme through ERC Grant Agreement 637422 EVERYSOUND. Minor part of the computations leading to these results were performed on a TITAN-X GPU donated by NVIDIA.

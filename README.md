# Introduction
This repository builds upon the innovative work of [icefall](https://github.com/k2-fsa/icefall), integrating and extending its features to provide a convenient and controllable framework for model training and testing. The base line code use [Zipformer](https://arxiv.org/abs/2310.11230) with Eden scheduler and ScaledAdam. Feel free to customize the provided [configuration file](D:\DeepLearning\Zipformer_Lightning\egs\librispeech\configs\zipformer.yaml) and [layer](D:\DeepLearning\Zipformer_Lightning\zipformer_lightning\layers) modules to suit your specific model and training requirements.

# How to use
## Installation

Step 1: Create environment
```bash
conda create -n zipformer_lightning python=3.10
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDAToolkit_ROOT_DIR=$CUDA_HOME
export CUDAToolkit_ROOT=$CUDA_HOME

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS
export CUDAToolkit_TARGET_DIR=$CUDA_HOME/targets/x86_64-linux

export CUDA_HOME=/usr/local/cuda # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export PATH=${CUDA_HOME}/bin:${PATH}
export CMAKE_CUDA_ARCHITECTURES=75

pip install -r requirements.txt
```
Step 2: Install Warp RNNT

```bash
bash scripts/install_rnnt.sh
```
Step 3: Install the repo to an editable package
```bash
pip install -e .
```

## Training and Test
Customize your configuration at the [config file](D:\DeepLearning\Zipformer_Lightning\egs\librispeech\configs\zipformer.yaml). Customize your model and modules at [here](D:\DeepLearning\Zipformer_Lightning\zipformer_lightning\layers). 

To train and test:
```bash
cd egs/librispeech
export CUDA_VISIBLE_DEVICES=0,1 python3 run.py --config configs/zipformer.yaml
```


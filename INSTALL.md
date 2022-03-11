# Installation Guide on Embedded Devices
We provide a Docker option to evaluate LiteReconfig without the time-consuming setup process. Check the [installation guide with docker](docker/README.md) and skip the remaining of this guide. This is a proof of concept for the installing steps below. To guarantee the performance of LiteReconfig, you must run LiteReconfig on the original OS without a docker container.

## pip and conda packages
We save the list of pip and conda installed packages on the TX2 and the AGX Xavier at "env/{conda,pip}\_package_list\_{tx2,xv}.txt". These files are references for the exact packages and versions. However, we still recommend you to go through this installation guide if you work on your own device.  

## Fix the power mode and DVFS setting
To help replicate the results, we suggest fixing the power mode and DVFS setting of the board.  
This is only needed if you want to set up the env on a new board. Use the following commands:
```
# We use power mode 0 for both TX2 and AGX Xavier
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Install the virtual environment tool
The following commands will install a special conda for the embedded boards (aarch64).
```
cd ~
wget https://github.com/jjhelmus/conda4aarch64/releases/download/1.0.0/c4aarch64_installer-1.0.0-Linux-aarch64.sh
bash c4aarch64_installer-1.0.0-Linux-aarch64.sh
# press enter and follow the default settings
source ~/.bashrc

# Add the `c4aarch64` and `conda-forge` channels to the conda configuration:
conda config --add channels c4aarch64
conda config --add channels conda-forge
```

## Install the Python packages in a virtual env
```
# Create a new env, follow the instruction and choose the default setting.
conda create -n ae python=3.6 -c conda-forge
conda activate ae

# Install numba 0.46.0
conda install -c numba numba=0.46.0

# Downgrade the pip to 19.0 version to install tensorflow
conda install pip=19.0

# Install some Python packages
conda install tqdm
pip install gdown

# The following parts are different depending on the hardward and Jetpack verison of Jetsons.
# Bellow are two examples for TX2 (e.g. L4T R32.2.1, Jetpack v4.2.2) and Xavier (e.g. L4T R32.4.4, Jetpack v4.4.1)

# TX2:
# Install tensorflow-gpu
pip install numpy==1.16.4 termcolor tensorboard==1.14.0 tensorflow-estimator=1.14.0 sklearn
pip install pip testresources setuptools numpy==1.16.1 future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
conda install h5py==2.10.0
pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.14.0+nv19.10
# Install opencv
conda install opencv=4.5.1
# Install scikit-learn
conda install scikit-learn==0.21.3
# Install torch and torchvision
wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.5.0  # where 0.x.0 is the torchvision version 
python setup.py install

# Xavier:
# Install tensorflow-gpu
pip install pip testresources setuptools numpy==1.16.1 future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
conda install h5py==2.10.0
pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'
# Install opencv
conda install opencv=4.5.1
# Install scikit-learn
conda install scikit-learn==0.21.3
# Install torch and torchvision
wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.7.0-cp36-cp36m-linux_aarch64.whl
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.8.1  # where 0.x.0 is the torchvision version 
python setup.py install
```

## Installation for Baseline Models(REPP, MEGA, SELSA)

- General Installation
```
# Create a new environment for the baselines.
conda create --name baselines python=3.6
conda activate baselines

# Latest pip seems to require python>=3.7, Downgrade pip.
conda install pip=19.0

# Get the jetson board pre-compiled pytorch 1.3.0, and install.
wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl
mv phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl torch-1.3.0-cp36-cp36m-linux_aarch64.whl

pip install torch-1.3.0-cp36-cp36m-linux_aarch64.whl
pip install 'pillow<7.0.0'
pip install pycocotools

# Install torchvision for torch 1.3.0
git clone --branch v0.4.2 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.4.2
python3 setup.py install

# install tensorflow
pip install pip testresources setuptools numpy==1.16.1 future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
conda install h5py==2.10.0
conda install scikit-learn==0.21.3
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'

# install specific version of keras with pip
pip install keras==2.2.4

# the pythonpath should be appended if there is no sudo privilages with the account.
PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages/"
export PYTHONPATH

pip install gdown

# Grab the replacement files from our repo.
git clone https://github.com/StarsThu2016/LiteReconfig_AE.git
cd LiteReconfig_AE
mkdir Files_for_Baselines; cd Files_for_Baselines
gdown https://drive.google.com/uc?id=1-3yenubLcZ5XizG_jgehV0OB-TmiTzPQ
tar -xf Files_for_Baselines.tar.xz
export REPLACE=$PWD
```

- Install MEGA
```
# Install MEGA and its dependencies.
pip install yacs

mkdir MEGA
cd MEGA

git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install
cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
# checkout an older timestamp to work with pytorch 1.3
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
python setup.py install --cuda_ext --cpp_ext
cd ..

git clone https://github.com/Scalsol/mega.pytorch.git
cd mega.pytorch
python setup.py build develop

# Download weights
gdown https://drive.google.com/uc?id=1i39MwpP46x61eHLkRXMzcKhpeKZhkgA6

# Replace files
mv REPLACE/mega.pytorch/demo/demo.py demo/
mv REPLACE/mega.pytorch/demo/predictor.py demo/
mv REPLACE/mega.pytorch/demo/VID_testimg_00106000.txt ./
mv REPLACE/mega.pytorch/mega_core/utils/distributed.py mega_core/utils/
```

- Install SELSA
```
# Install SELSA and its dependencies.
mkdir SELSA
cd SELSA

conda install ninja cmake pandas scipy

# Install mmcv from old timestamp
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu100/torch1.3.0/index.html


git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout c76ab0eb3c637b86c343d8454e07e00cfecc1b78
pip install -r requirements/build.txt
pip install -v -e . 
cd ..

git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking
git checkout b3eedb1ac885821123debad9980a5566cd78c29c
pip install -r requirements/build.txt
pip install -v -e .

# Download the weights
wget https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth

# Modify files
mv REPLACE/mmtracking/demo/demo_vid.py demo/
mv REPLACE/mmtracking/mmtrack/models/vid/base.py mmtrack/models/vid
mv REPLACE/mmtracking/demo/VID_testimg_00106000.txt ./
```

- Install REPP
```
# Install REPP
mkdir REPP
cd REPP

git clone https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection.git

# Download the weights and locate them.
cd Robust-and-efficient-post-processing-for-video-object-detection/demos/YOLOv3/pretrained_models
gdown https://drive.google.com/uc?id=19aRWqMytRRq3ukV5h3lNAghIgs4JCpbz
unzip REPP_models.zip
mv REPP_models/demos/YOLOv3/pretrained_models/ILSVRC ILSVRC

# Replace and Add files
mv REPLACE/Robust-and-efficient-post-processing-for-video-object-detection/demos/YOLOv3/get_repp_predictions.py demos/YOLOv3
mv REPLACE/Robust-and-efficient-post-processing-for-video-object-detection/REPP.py ./
mv REPLACE/Robust-and-efficient-post-processing-for-video-object-detection/data_annotations/annotations_val_ILSVRC_chop.txt data_annotations/
```


The Tensorflow verion on Jetson devices is related to the Jetpack version of the board.
Refer to [TX2](https://forums.developer.nvidia.com/t/tensorflow-for-jetson-tx2/64596) and [AGX Xavier](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-agx-xavier/65523) for more details.
The Pytorch version on Jetson devices is also related to the Jetpack version. Refer to this [link](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) for more details.

## PyTorch 1.6.0a + ROCm 3.3 for AMD RadeonGPU @ Apr 20th, 2020

### Update and upgrade to latest packages.
```
sudo apt update
sudo apt -y dist-upgrade
```

### Install "Non Uniform Memory Access" dev package.
```
sudo apt install -y libnuma-dev
```


### Add the ROCm apt repository

```
wget -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
```

### Install the ROCm driver.
```
sudo apt update
sudo apt install -y rocm-dkms rocm-libs hipcub miopen-hip rccl
sudo reboot
```


### GPU info

| AMDGPU | NVIDIA | Description |
|:---|:---|:---|
| rocm-smi | nvidia-smi | GPU information command. |
| clinfo | clinfo | OpenCL GPU information command. |



### Test for rocm installations
```
# Make sure to recognized GPUs as file descriptor.
ls /dev/dri/

# >> card0  renderD128
```

```
# Make sure to recognized GPUs.
/opt/rocm/bin/rocm-smi

# ========================ROCm System Management Interface========================
# ================================================================================
# GPU  Temp   AvgPwr  SCLK    MCLK    Fan     Perf  PwrCap  VRAM%  GPU%
# 0    35.0c  18.0W   808Mhz  350Mhz  21.96%  auto  250.0W    0%   0%
# ================================================================================
# ==============================End of ROCm SMI Log ==============================
```

```
# Make sure to recognized GPUs using OpenCL.
/opt/rocm/opencl/bin/x86_64/clinfo

# Number of platforms:				 1
#   Platform Profile:				 FULL_PROFILE
#   Platform Version:				 OpenCL 2.1 AMD-APP (3098.0)
#   Platform Name:				 AMD Accelerated Parallel Processing
#   Platform Vendor:				 Advanced Micro Devices, Inc.
#   Platform Extensions:				 cl_khr_icd cl_amd_event_callback cl_amd_offline_devices
# 
# 
#   Platform Name:				 AMD Accelerated Parallel Processing
# Number of devices:				 1
#   Device Type:					 CL_DEVICE_TYPE_GPU
#   Vendor ID:					 1002h
#   Board name:					 Vega 20
#   Device Topology:				 PCI[ B#3, D#0, F#0 ]
#   Max compute units:				 60
#   .
#   .
#   .
```


### Add rocm binary paths
```
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' |
sudo tee -a /etc/profile.d/rocm.sh
sudo reboot
```


### Set permissions. To access the GPU you must be a user in the video group
### To add your user to the video group run

```
sudo usermod -a -G video $LOGNAME
```
<br>

### Preparing for installing pytorch
Install mkl
```
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get install intel-mkl-64bit-2018.2-046

```


### Installing PyTorch
You can choose either to install from [wheels](#Install-from-wheel) or build from [source](#Source-build-for-developers)

### Install from wheel

| Python | ROCm | PyTorch | GPU | S |
|:---:|:---:|:---:|:---:|:---:|
| 3.7 | 3.3 | [1.6.0a](http://install.aieater.com/libs/pytorch/rocm3.3/gfx906/torch-1.6.0a0-cp37-cp37m-linux_x86_64.whl) | GFX906 | ✓ | 
| 3.7 | 3.3 | 1.6.0a | GFX900 |  - |
| 3.5 | 2.9 | [1.3.0a](http://install.aieater.com/libs/pytorch/rocm2.9/gfx900/torch-1.3.0a0+e5d82b5-cp35-cp35m-linux_x86_64.whl) | GFX900 |  ✓ | 
| 3.5 | 2.9 | [1.3.0a](http://install.aieater.com/libs/pytorch/rocm2.9/gfx906/torch-1.3.0a0+e5d82b5-cp35-cp35m-linux_x86_64.whl) | GFX906 |  ✓ |
| 3.7 | 2.9 | 1.3.0a | GFX900 |  - |
| 3.7 | 2.9 | 1.3.0a | GFX906 |  - |


<br>

| GFX Code | Architecture | Products |
|:---:|:---:|:---|
| GFX806 | Polaris Series | RX550/RX560/RX570/RX580/RX590 ... |
| GFX900 | Vega10 Series | Vega64/Vega56/MI25/WX9100/FrontierEdition ... |
| GFX906 | Vega20 Series | RadeonVII/MI50/MI60 ... |


<br>




```
# ROCm3.3 PyTorch1.6.0a
sudo pip3 install http://install.aieater.com/libs/pytorch/rocm3.3/gfx906/torch-1.6.0a0-cp37-cp37m-linux_x86_64.whl torchvision
```
continue to [test](#GPU-visibly-masking-and-multiple-GPUs) the installation. 

<br>

### Source build for developers


#### Install dependencies
```
sudo apt install -y gcc cmake clang ccache llvm ocl-icd-opencl-dev python3-pip
sudo apt install -y rocrand rocblas miopen-hip miopengemm rocfft rocprim rocsparse rocm-cmake rocm-dev rocm-device-libs rocm-libs rccl hipcub rocthrust

export PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/opencl/bin:$PATH
export USE_LLVM=/opt/llvm
export LLVM_DIR=/opt/llvm/lib/cmake/llvm
```


#### Clone PyTorch repository
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
```

#### <font color="red">'Hipify' PyTorch source by executing python tools/amd_build/build_amd.py</font>
```
python3 tools/amd_build/build_amd.py
```

#### Alternative to get a fixed version.
```
wget http://install.aieater.com/libs/pytorch/sources/pytorch1.6.0.tar.gz
```
This pytorch project has already hippified, and cloned sub modules.

#### <font color="red">Required environment variables</font>

| GFX Code | Architecture | Products |
|:---:|:---:|:---|
| GFX806 | Polaris Series | RX550/RX560/RX570/RX580/RX590 ... |
| GFX900 | Vega10 Series | Vega64/Vega56/MI25/WX9100/FrontierEdition ... |
| GFX906 | Vega20 Series | RadeonVII/MI50/MI60 ... |

```
#export HCC_AMDGPU_TARGET=gfx806 #(RX550/RX560/RX570/RX580/RX590 ...)
export HCC_AMDGPU_TARGET=gfx900 #(Vega64/Vega56/MI25/WX9100/FrontierEdition ...)
#export HCC_AMDGPU_TARGET=gfx906 #(RadeonVII/MI50/MI60 ...)

export USE_NINJA=1
export MAX_JOBS=8
export HIP_PLATFORM=hcc

echo $HCC_AMDGPU_TARGET
```

#### Install cmake and requirements to Build and install
```
pip3 install -r requirements.txt
python3 setup.py install --user
```

#### Distribution build for wheel
```
python3 setup.py build
```

#### Cleanup
```
python3 setup.py clean
```
<br>

### GPU visibly masking and multiple GPUs

| AMDGPU | NVIDIA | Description |
|:---|:---|:---|
| export=HIP_VISIBLE_DEVICES= | export=CUDA_VISIBLE_DEVICES= | CPU |
| export=HIP_VISIBLE_DEVICES=0 | export=CUDA_VISIBLE_DEVICES=0 | Single GPU |
| export=HIP_VISIBLE_DEVICES=0,1 | export=CUDA_VISIBLE_DEVICES=0,1 | Multiple GPUs |

### Make sure to recognize GPU device via PyTorch.

```
# Check GPU is available or not.
python3 -c 'import torch;print("GPU:",torch.cuda.is_available())'

# GPU: True
```

```
python3 -c 'import torch;print("DeviceID:",str(torch.cuda.current_device()))'


# DeviceID: 0
```

```
python3 -c 'import torch;print("DeviceName:",str(torch.cuda.get_device_name(torch.cuda.current_device())))'

# DeviceName: Vega 20
```

### Make sure everything is working
```
PYTORCH_TEST_WITH_ROCM=1 python3 test/run_test.py --verbose
```


### Also see, AMDGPU - ROCm Caffe/PyTorch/Tensorflow 1.x installation, official, introduction on docker
- GPUEater ROCM-Tensorflow installation https://www.gpueater.com/help
- GPUEater github ROCm-Tensorflow information https://github.com/aieater/rocm_tensorflow_info
- GPUEater github ROCm-PyTorch information https://github.com/aieater/rocm_pytorch_informations
- GPUEater github ROCm-Caffe information https://github.com/aieater/rocm_caffe_informations
- ROCm+DeepLearning libraries https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html
- ROCm github https://github.com/RadeonOpenCompute/ROCm
- ROCm-TensorFlow on Docker https://hub.docker.com/r/rocm/tensorflow/

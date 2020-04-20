## PyTorch 1.6 + ROCm 3.3 for AMD RadeonGPU

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
sudo apt install rocm-dkms
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
```


### Set permissions. To access the GPU you must be a user in the video group
### To add your user to the video group run

```
sudo usermod -a -G video $LOGNAME
```
<br>
<br>
<br>
<br>


### Preparing for installing pytorch
### Install ROCm PyTorch dependencies (some might already be installed)
```
sudo apt install -y rocrand hiprand rocblas miopen miopengemm rocfft rocsparse rocm-cmake rocm-dev rocm-device-libs rocm-libs hcc hip_base hip_hcc hip-thrust
```


### Install from wheel

| Python | ROCm | PyTorch | GPU | S |
|:---:|:---:|:---:|:---:|:---:|
| 3.5 | 3.3 | [1.6.0a](http://install.aieater.com/libs/pytorch/rocm3.3/gfx900/torch-1.6.0a0+d83509e-cp35-cp35m-linux_x86_64.whl) | GFX900 | ✓ | 
| 3.5 | 3.3 | [1.6.0a](http://install.aieater.com/libs/pytorch/rocm3.3/gfx906/torch-1.6.0a0+d83509e-cp35-cp35m-linux_x86_64.whl) | GFX906 | ✓ | 
| 3.7 | 3.3 | 1.6.0a | GFX900 |  - |
| 3.7 | 3.3 | 1.6.0a | GFX906 |  - |
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
# RadeonVII(GFX906) ROCm3.3 PyTorch1.6.0a
sudo pip3 install http://install.aieater.com/libs/pytorch/rocm3.3/gfx906/torch-1.6.0a0+d83509e-cp35-cp35m-linux_x86_64.whl torchvision
```
```
# Vega64(GFX900) ROCm3.3 PyTorch1.6.0a
sudo pip3 install http://install.aieater.com/libs/pytorch/rocm3.3/gfx900/torch-1.6.0a0+d83509e-cp35-cp35m-linux_x86_64.whl torchvision
```

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


### GPU visibly masking and multiple GPUs

| AMDGPU | NVIDIA | Description |
|:---|:---|:---|
| export=HIP_VISIBLE_DEVICES= | export=HIP_VISIBLE_DEVICES= | CPU |
| export=HIP_VISIBLE_DEVICES=0 | export=HIP_VISIBLE_DEVICES=0 | Single GPU |
| export=HIP_VISIBLE_DEVICES=0,1 | export=HIP_VISIBLE_DEVICES=0,1 | Multiple GPUs |

<br>
<br>
<br>
<br>
<br>
<br>

-----------------------------------

<br>
<br>
<br>


## Source build for developers


### Clone PyTorch repository
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
```

### <font color="red">'Hipify' PyTorch source by executing python tools/amd_build/build_amd.py</font>
```
python3 tools/amd_build/build_amd.py
```


### <font color="red">Required environment variables</font>

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

echo $HCC_AMDGPU_TARGET
```

### Build and install
```
python3 setup.py install
```

### Distribution build for wheel
```
python3 setup.py build
```

### Cleanup
```
python3 setup.py clean
```


### Make sure everything is working
```
PYTORCH_TEST_WITH_ROCM=1 python test/run_test.py --verbose
```


### Also see, AMDGPU - ROCm Caffe/PyTorch/Tensorflow 1.x installation, official, introduction on docker
- GPUEater ROCM-Tensorflow installation https://www.gpueater.com/help
- GPUEater github ROCm-Tensorflow information https://github.com/aieater/rocm_tensorflow_info
- GPUEater github ROCm-PyTorch information https://github.com/aieater/rocm_pytorch_informations
- GPUEater github ROCm-Caffe information https://github.com/aieater/rocm_caffe_informations
- ROCm+DeepLearning libraries https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html
- ROCm github https://github.com/RadeonOpenCompute/ROCm
- ROCm-TensorFlow on Docker https://hub.docker.com/r/rocm/tensorflow/

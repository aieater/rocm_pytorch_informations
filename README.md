# AMD RadeonGPU ROCm-PyTorch information
<br>
<br>
<br>
This README is intended to provide helpful information for Deep Learning developers with AMD ROCm.<br>
<br>
Unfortunately, AMD's official repository for ROCm sometimes includes old or missing information. Therefore, on this readme, we will endeavor to describe accurate information based on the knowledge gained by GPUEater infrastructure development and operation.<br>
<br>
<br>
<br>
<br>

- How to setup ROCm-PyTorch1.1.0a on Ubuntu16.04/18.04
  + ROCm(AMDGPU)-PyTorch1.1.0a Python2.7/Python3.5.2 + UbuntuOS
<br>
<br>
<br>
<br>

### Python3.5.2 + ROCm Latest Driver(2.x) + ROCm-PyTorch1.1.0a easy installer without Docker (Recommend)
```
curl -sL http://install.aieater.com/setup_pytorch_rocm | bash -
```

<br>
<br>

### Install script without Docker
```

if [ "$(uname)" == 'Darwin' ]; then
  OS='MacOSX'
  echo "Your platform ( $OS ) is not supported."
  exit 1
elif [ "$(expr substr $(uname -s) 1 5)" == 'Linux' ]; then
  OS='Linux'
  echo "Detected Linux OS."
elif [ "$(expr substr $(uname -s) 1 10)" == 'MINGW32_NT' ]; then                                                                                           
  OS='Cygwin'
  echo "Your platform ( $OS ) is not supported."
  exit 1
else
  echo "Your platform ($(uname -a)) is not supported."
  exit 1
fi



IS_AVAILABLE_OS="NO"

OSSTR=`cat /etc/os-release | grep "Ubuntu 16.04"`
echo "Check OS ${OSSTR}"
if [ ${#OSSTR} == 0 ]; then
echo "None"
else
IS_AVAILABLE_OS="Ubuntu 16.04"
fi

OSSTR=`cat /etc/os-release | grep "Ubuntu 18.04"`
echo "Check OS ${OSSTR}"
if [ ${#OSSTR} == 0 ]; then
echo "None"
else
IS_AVAILABLE_OS="Ubuntu 18.04"

if cat /etc/apt/sources.list | grep 'bionic main universe' >/dev/null; then
    echo "System already has bionic main universe repository"
else
    #sudo sh -c "echo '' > /etc/apt/sources.list"
    sudo sh -c "echo 'deb http://archive.ubuntu.com/ubuntu bionic main universe' >> /etc/apt/sources.list"
    sudo sh -c "echo 'deb http://archive.ubuntu.com/ubuntu bionic-security main universe' >> /etc/apt/sources.list"
    sudo sh -c "echo 'deb http://archive.ubuntu.com/ubuntu bionic-updates main universe' >> /etc/apt/sources.list"
    sudo apt update
fi


fi

if test "$IS_AVAILABLE_OS" != "NO" ; then




apt-get update && apt-get install -y --no-install-recommends curl && \
  curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' \
  
  
apt-get update &&  apt-get install -y --no-install-recommends \
  libelf1 \
  build-essential \
  bzip2 \
  ca-certificates \
  cmake \
  ssh \
  apt-utils \
  pkg-config \
  g++-multilib \
  gdb \
  git \
  less \
  libunwind-dev \
  libfftw3-dev \
  libelf-dev \
  libncurses5-dev \
  libomp-dev \
  libpthread-stubs0-dev \
  make \
  miopen-hip \
  miopengemm \
  python3-dev \
  python3-future \
  python3-yaml \
  python3-pip \
  vim \
  libssl-dev \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  libopenblas-dev \
  rpm \
  wget \
  net-tools \
  iputils-ping \
  libnuma-dev \
  rocm-dev \
  rocrand \
  rocblas \
  rocfft \
  hipsparse \
  hip-thrust \
  rccl \
  
curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
sh -c 'echo deb [arch=amd64] http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main > /etc/apt/sources.list.d/llvm7.list' && \
sh -c 'echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main >> /etc/apt/sources.list.d/llvm7.list'\

apt-get update && apt-get install -y --no-install-recommends clang-7

apt-get clean && \
rm -rf /var/lib/apt/lists/*

sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocsparse/lib/cmake/rocsparse/rocsparse-config.cmake
sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocfft/lib/cmake/rocfft/rocfft-config.cmake
sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/miopen/lib/cmake/miopen/miopen-config.cmake
sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocblas/lib/cmake/rocblas/rocblas-config.cmake


prf=`cat <<'EOF'
export HIP_VISIBLE_DEVICES=0
export HCC_HOME=/opt/rocm/hcc
export ROCM_PATH=/opt/rocm
export ROCM_HOME=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=/usr/local/bin:$HCC_HOME/bin:$HIP_PATH/bin:$ROCM_PATH/bin:/opt/rocm/opencl/bin/x86_64:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/opencl/lib/x86_64
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export HIP_PLATFORM="hcc"
export KMTHINLTO="1"
export CUPY_INSTALL_USE_HIP=1
export MAKEFLAGS=-j8
export __HIP_PLATFORM_HCC__
export HIP_PLATFORM=hcc
export PLATFORM=hcc
export USE_ROCM=1
export MAX_JOBS=2
EOF
`

GFX=gfx900
echo "Select a GPU type."
select INS in RX500Series\(RX550/RX560/RX570/RX580/RX590\) Vega10Series\(Vega56/64/WX9100/FE/MI25\) Vega20Series\(RadeonVII/MI50/MI60\) Default
do
case $INS in
Patch)
PATCH;
break;;
RX500Series\(RX550/RX560/RX570/RX580/RX590\))
GFX=gfx806
break;;
Vega10Series\(Vega56/64/WX9100/FE/MI25\))
GFX=gfx900
break;;
Vega20Series\(RadeonVII/MI50/MI60\))
GFX=gfx906
break;;
Default)
break;;
*) echo "ERROR: Invalid selection"
;;
esac
done
export HCC_AMDGPU_TARGET=$GFX


echo "$prf" >> ~/.profile
source ~/.profile

pip3 install cython pillow h5py numpy scipy requests sklearn matplotlib editdistance pandas portpicker jupyter setuptools pyyaml typing enum34 hypothesis


update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-7 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-7 50

# git clone https://github.com/pytorch/pytorch.git
git clone https://github.com/ROCmSoftwarePlatform/pytorch.git pytorch-rocm
cd pytorch-rocm
git checkout e6991ed29fec9a7b7ffb09b6ec58fb9d3fec3d22 # 1.1.0a0+e6991ed
git submodule init
git submodule update

#python3 tools/amd_build/build_pytorch_amd.py
#python3 tools/amd_build/build_caffe2_amd.py
python3 tools/amd_build/build_amd.py

python3 setup.py install
pip3 install torchvision

cd ~/
clinfo | grep '  Name:'
python3 -c "import torch;print('CUDA(hip) is available',torch.cuda.is_available());print('cuda(hip)_device_num:',torch.cuda.device_count());print('Radeon device:',torch.cuda.get_device_name(torch.cuda.current_device()))"


else
        echo "System must be Ubuntu16.04 or Ubuntu18.04"
fi
```

<br>
<br>
<br>
<br>


### GFX806/GFX900/GFX906

Latest ROCm-PyTorch needs to specify GPU architecture like gfx900.

<br>

#### gfx806(Polaris Series)
RX550/RX560/RX570/RX580/RX590...

#### gfx900(Vega10 Series)
Vega56/Vega64/WX9100/MI25...

#### gfx906(Vega20 Series)
RadeonVII/MI50/MI60...

<br>
<br>
<br>
<br>
<br>

### Also see, AMDGPU - ROCm Caffe/PyTorch/Tensorflow 1.x installation, official, introduction on docker
- GPUEater ROCM-Tensorflow installation https://www.gpueater.com/help
- GPUEater github ROCm-Tensorflow information https://github.com/aieater/rocm_tensorflow_info
- GPUEater github ROCm-PyTorch information https://github.com/aieater/rocm_pytorch_informations
- GPUEater github ROCm-Caffe information https://github.com/aieater/rocm_caffe_informations
- ROCm+DeepLearning libraries https://rocm-documentation.readthedocs.io/en/latest/Deep_learning/Deep-learning.html
- ROCm github https://github.com/RadeonOpenCompute/ROCm
- ROCm-TensorFlow on Docker https://hub.docker.com/r/rocm/tensorflow/

# CUDA Installation

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Configure repo

git clone --single-branch -b dev_sageattn https://github.com/stdcall0/WinT3R
cd WinT3R
sh ./configure.sh

git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 python setup.py install

cd ..
# compare baseline: run
# SAGEATTENTION_DISABLED=1 python recon_fp16.py ...
# and
# SAGEATTENTION_DISABLED= python recon_fp16.py ...

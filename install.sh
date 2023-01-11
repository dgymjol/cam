conda create --name cam python=3.8 -y 
# # source ~/anaconda3/etc/profile.d/conda.sh
conda activate cam

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch -y

pip3 install pandas gpustat matplotlib numpy -y

conda install git wget -y

sudo apt-get update
sudo apt-get install tmux

conda install pyyaml scit-skelarn -y
conda install scikit-learn tensorboardX -y
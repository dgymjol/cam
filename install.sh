conda create --name cam python=3.8 -y 
# # source ~/anaconda3/etc/profile.d/conda.sh
conda activate cam

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch -y

pip3 install pandas gpustat matplotlib numpy -y

conda install git wget -y

cd cam/data
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

#####
sudo apt-get update
sudo apt-get install tmux

git config --global user.name "sjpark"
git config --global user.email dgymjol@yonsei.ac.kr

cd ../../
git clone https://github.com/dgymjol/cam.git

cd cam/data/CUB_200_2011
python preprocessing.py

conda install pyyaml scit-skelarn -y
conda install scikit-learn tensorboardX -y
pip install git+https://github.com/ildoonet/pytorch-randaugment

conda create -n RF-training python=3.6
export PATH="/home/duchstf/miniconda3/bin:$PATH"
source activate RF-training
pip uninstall -y tensorflow
pip uninstall -y tf-nightly
# for gpu support:
#pip install -q -U tf-nightly-gpu
pip install -q -U tf-nightly
pip install -q tensorflow-model-optimization
pip install matplotlib
pip install guildai
pip uninstall -y enum34
pip install tensorflow
pip install keras

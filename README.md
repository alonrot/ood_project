pip install -e .

conda create -n tf-metal python=3.8

pip install tensorflow
pip install tensorflow_probability
pip install numpy scipy matplotlib
pip install hydra-core --upgrade
pip install control
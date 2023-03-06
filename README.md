pip install -e .

conda create -n tf-metal python=3.8

pip install tensorflow
pip install tensorflow_probability
pip install numpy scipy matplotlib
pip install hydra-core --upgrade
pip install control


perl ./install-tl --no-interaction --scheme=small --no-doc-install --no-src-install --texdir=/home/amarco/code_projects/latex_installation --texuserdir=/home/amarco/code_projects/latex_installation

export PATH="/home/amarco/code_projects/latex_installation/bin/x86_64-linux:$PATH"


scp -r -P 4444 amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/dubins_car_reconstruction/\* /Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_reconstruction/from_dawkins/
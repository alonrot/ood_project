

### Install

```bash
conda create -n ood python=3.7

pip install tensorflow
pip install tensorflow_probability
pip install numpy scipy matplotlib==3.2.2
pip install hydra-core --upgrade
pip install control
pip install -e .
cd <path/to/LQRker>
pip install -e .
```

### Install Latex
```bash
perl ./install-tl --no-interaction --scheme=small --no-doc-install --no-src-install --texdir=/home/amarco/code_projects/latex_installation --texuserdir=/home/amarco/code_projects/latex_installation
export PATH="/home/amarco/code_projects/latex_installation/bin/x86_64-linux:$PATH"
```


### Copy data
```bash
scp -r -P 4444 amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/dubins_car_reconstruction/\* /Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_reconstruction/from_dawkins/

scp -r -P 4444 amarco@hybridrobotics.hopto.org:/home/amarco/code_projects/ood_project/ood/experiments/dubins_car_receding_gpflow/model_13_coregionalization_True /Users/alonrot/work/code_projects_WIP/ood_project/ood/experiments/dubins_car_receding_gpflow/from_hybridrobotics/
```
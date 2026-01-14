# Muller-Brown Experiments
This directory contains the code to reproduce results on the 2D Muller-Brown (MB) potential

## Download Data
Run these commands from the ```muller-brown``` directory
```bash
wget https://storage.googleapis.com/om-tps/data/data.zip
unzip data.zip
```
This will download the training data for the Muller-Brown potential, as well as samples from the trained diffusion model used for training the committor model.

The code is organized in a series of Jupyter notebooks (all other files contain utilities called by these notebooks):

- ```mb_analytical.ipynb```: Uses the analytical MB force field with OM optimization to find transition paths. Also used to generate data for training generative models. 
- ```mb_diffusion.ipynb```: Trains a denoising diffusion model on samples from 2DMB and performs various flavors of OM optimization. Also used generated data for estimating committor function and rates.
- ```mb_flowmatch.ipynb```: Trains a flow matching model on samples from 2DMB and performs various flavors of OM optimization. 
- ```mb_committor_rates.ipynb```: Trains a MLP to estimate the committor function, from which reaction rates are computed using the Backward Kolmogorov Equation (BKE) formalism.

If you would like to re-compute the ground truth transition rate using the Backward Kolmogorov Equation, create a dedicated conda envirnoment with fenics/mshr, and then run ```python mb_fem.py```:

```bash
# Create mamba environment with Python 3.12
mamba create -n mb_fem python=3.12 -y
# Activate the environment
conda activate mb_fem
# Install conda-forge packages
mamba install -c conda-forge numpy=1.26.4 matplotlib mshr=2019.1.0 -y
# Install PyTorch 2.3.1 with CUDA 12.1 support from PyPI
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install remaining PyPI packages
pip install fenics==2019.1.0 ase==3.23.0
```




## Citation
If you use this code in your research, please cite our paper.

```bibtex
@inproceedings{raja2025action,
  title={Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional},
  author={Raja, Sanjeev and {\v{S}}{\'\i}pka, Martin and Psenka, Michael and Kreiman, Tobias and Pavelka, Michal and Krishnapriyan, Aditi S},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
  organization={PMLR}
}


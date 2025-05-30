# Fast Folding Proteins and Tetrapeptides

## Requirements
Make sure the ```om-tps``` and ```openmm-env```conda environments are set up as described [here](../README.md)

### Data
#### Fast Folding Proteins
We provide torch tensors containing coordinates of the coarse-grained fast-folding protein trajectories from D.E.Shaw, from which transition path endpoints are extracted:
```bash
cd datasets
wget https://storage.googleapis.com/om-tps/data/torch_trajectories.zip
unzip torch_trajectories.zip
```

If you would like to train models from scratch, you can request the original trajectory data directly to D.E.Shaw at the following email: Christine.Ueda@deshawresearch.com.

#### Tetrapeptides
To download and process the tetrapeptide data, run these commands from this directory:
```bash
mkdir -p datasets/4AA_sim
gsutil -m rsync -r gs://mdgen-public/4AA_sims datasets/4AA_sim
cd mdgen
python -m scripts.prep_sims --splits splits/4AA.csv --sim_dir ../datasets/4AA_sim --outdir ../datasets/4AA_sim --num_workers [N] --suffix _i100 --stride 100
python -m scripts.prep_sims --splits splits/4AA_test.csv --sim_dir ../datasets/4AA_sim --outdir ../datasets/4AA_sim --num_workers [N]
wget https://storage.googleapis.com/om-tps/data/metadata.zip
unzip metadata.zip
```
***

## Transition path sampling
### Pretrained models 
Pretrained models per protein can be found in the [saved_models](./saved_models/) folder. The diffusion models trained on fast-folding proteins are taken directly from [Two for One](https://github.com/microsoft/two-for-one-diffusion), while the flow matching models and all-atom tetrapeptide models are trained ourselves using ```main_train.py```.


For each protein, we provide a ```model-best.pt``` and```model-best-flow.pt``` file with the checkpoint corresponding to the best validation loss for diffusion and flow matching respectively, as well as an ```args.pickle``` and ```args-flow.pickle``` file containing the arguments used for that run.

### Sampling
Transition path sampling for each protein can be done using the [sample script](./sample.py). Here is an example of how to sample 32 transition paths of the BBA protein using OM optimization:

```bash
python sample.py \
    --model_path saved_models/bba \
    --gen_mode om_interpolate \
    --atom_selection c-alpha \
    --num_samples_eval 32 \
    --batch_size_gen 2 \
    --latent_time 20 \
    --initial_guess_level 250\
    --no_encode_and_decode \
    --action hutch \
    --optimizer sgd \
    --lr 1e-5 \
    --append_exp_name test_initial_latent_time_250_hutch_minus_SGD_32paths_physical_params_FINAL \
    --om_dt 0.001 \
    --om_gamma 1 \
    --om_d 1 \
    --path_length 200 \
    --path_batch_size 200 \
    --steps 5000
```

Refer to ```python sample.py --help``` for more sampling options and an explanation of all individual arguments. Exact commands to reproduce results in the paper can be found [here](./evaluate/sampling_commands.md).

### Evaluation
Running the above sampling script will produce a folder in ```saved_models/``` with all evaluation metrics reported in the paper. Evaluation can also be run in a standalone fashion for [fast-folding proteins](./evaluate/evaluate_fastfolders.py) and [tetrapeptides](./evaluate/evaluate_tetrapeptides.py).

***

## External code-sources used to create this codebase
1. This codebase builds heavily on [Two for One Diffusion](https://github.com/microsoft/two-for-one-diffusion) (PyTorch) by Microsoft. We use many of the same pretrained models, sampling and evaluation code, and infrastructure. 
2. We also build on [MDGen](https://github.com/bjing2016/mdgen) for Markov State Model (MSM) analysis and tetrapeptide infrastructure.

***
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
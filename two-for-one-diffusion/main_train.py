# Importing
import argparse
import torch
import sys
import os
import warnings
import mdtraj as md

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ddpm import GaussianDiffusion
from models.flow_matching import FlowMatching
from trainer import Trainer
from models import get_model
from datasets.dataset_utils_empty import get_dataset, Molecules, AtomSelection
from evaluate.evaluators import TicEvaluator


all_molecules = ["alanine_dipeptide"] + [mol.name.lower() for mol in Molecules]

# Arguments
parser = argparse.ArgumentParser(description="coarse-graining")
parser.add_argument(
    "--mol",
    type=str,
    default="alanine_dipeptide",
    help=f"Select molecule, choose from (case insensitive): {all_molecules}",
)
parser.add_argument(
    "--fold",
    type=int,
    default=1,
    help="Fold from [1,2,3,4] for four-fold cross validation. Only for alanine_dipeptide",
)
parser.add_argument(
    "--atom_selection",
    type=str,
    default="c-alpha",
    help="atom selection for fast folding proteins (c-alpha, protein, all)",
)
parser.add_argument(
    "--data_folder",
    type=str,
    default="./data",
    help="directory root to save simulation data",
)
parser.add_argument(
    "--results_folder",
    type=str,
    default="./results",
    help="directory root to save model checkpoints and samples",
)
parser.add_argument(
    "--tensorboard_folder",
    type=str,
    default="./runs",
    help="directory root to save tensorboard log file",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="debug",
    help="experiment name to save run within ./runs/, also allows subdirectory, timestamp will be added",
)
parser.add_argument(
    "--traindata_subset",
    type=int,
    default=None,
    help="Take a randomly sampled subset from the training data to train on. In flow-matching paper: [750000,500000,200000,100000,50000,20000,10000]. Only for alanine_dipeptide",
)

parser.add_argument(
    "--committor_remove_range",
    type=float,
    nargs="+",
    help="Two floats (e.g., 0.4 0.6) giving the bounds of the committor function to remove from the dataset - only for fast folder proteins.",
)

parser.add_argument(
    "--remove_freq",
    type=float,
    help="Float corresponding to the frequency of removing transition region from dataset - only for fast folder proteins.",
)


parser.add_argument(
    "--mean0",
    type=eval,
    default=True,
    help="center molecules from train and validation set to zero",
)
parser.add_argument(
    "--data_aug",
    type=eval,
    default=True,
    help="use data augmentation (rotation) for training",
)

parser.add_argument(
    "--flow_matching",
    action="store_true",
    help="use flow matching instead of diffusion",
)

parser.add_argument(
    "--hidden_features_gnn",
    type=int,
    default=64,
    help="number of hidden features used in graph transformer",
)
parser.add_argument(
    "--num_layers_gnn",
    type=int,
    default=3,
    help="number of layers used in graph transformer",
)
parser.add_argument(
    "--heads", type=int, default=8, help="number of heads used in graph transformer"
)
parser.add_argument(
    "--dim_head",
    type=int,
    default=64,
    help="dimension of each head used in graph transformer",
)

parser.add_argument(
    "--use_layernorm",
    type=eval,
    default=True,
    help="whether using layer norm or not in the GNN",
)
parser.add_argument(
    "--conservative",
    type=eval,
    default=True,
    help="set True to learn a conservative Force Field",
)
parser.add_argument(
    "--diffusion_steps",
    type=int,
    default=1000,
    help="number of time steps used in diffusion",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="batch size used in training and validation",
)
parser.add_argument(
    "--gradient_accumulate_every",
    type=int,
    default=1,
    help="gradient accumulation frequency (effective batch size = batch_size * gradient_accumulate_every)",
)


parser.add_argument(
    "--learning_rate", type=float, default=4e-4, help="learning rate for Adam"
)

parser.add_argument(
    "--weight_decay", type=float, default=1e-12, help="weight decay in the optimizer"
)

parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.05,
    help="fraction of steps to warm up learning rate",
)

parser.add_argument(
    "--gradient_norm_threshold",
    type=int,
    default=1000,
    help="maximum allowed gradient norm before skipping the update",
)
parser.add_argument(
    "--train_iter",
    type=int,
    default=1000000,
    help="number of iterations to train the model",
)
parser.add_argument(
    "--ema_decay",
    type=float,
    default=0.995,
    help="ema decay, 0 is no smoothing, 1 is completely smooth",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=100000,
    help="interval at which to sample and calculate evaluation metrics",
)
parser.add_argument(
    "--log_tensorboard_interval",
    type=int,
    default=1,
    help="Interval at which to log the training loss in tensorboard. We recommend to set this to 100 if running in Amulet to avoid stalling the log.",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=5000,
    help="number of samples to draw from the model",
)
parser.add_argument(
    "--num_samples_final_eval",
    type=int,
    default=400000,
    help="number of samples to draw from the model in the final evaluation after training",
)
parser.add_argument(
    "--use_intrinsic_coords",
    type=eval,
    default=False,
    help="input coordinates as edge attributes in the form of pairwise distances and normalized vectors instead of absolute coordinates in the nodes",
)
parser.add_argument(
    "--use_abs_coords",
    type=eval,
    default=True,
    help="input the absolute coordinates as node embeddings",
)
parser.add_argument(
    "--use_distances",
    type=eval,
    default=True,
    help="input distances in the edges",
)
parser.add_argument(
    "--use_rbf",
    type=eval,
    default=False,
    help="when use_distances=True, embeds the pairwise distances as radial basis functions if use_rbf is set to true",
)
parser.add_argument(
    "--r_max",
    type=float,
    default=None,
    help="choose a maximum radius in (Angstrom) to propagate messages among neighbors. Not coded yet",
)
parser.add_argument(
    "--residual_edge",
    type=eval,
    default=True,
    help="wether using a residual connection in the edges",
)
parser.add_argument(
    "--graph_mlp_decoder",
    type=eval,
    default=False,
    help="use an MLP when mapping to energies in the conservative field",
)
parser.add_argument(
    "--gnn_efficient",
    type=eval,
    default=False,
    help="use a more efficient architecture in the gnn",
)
parser.add_argument(
    "--min_lr_cosine_anneal",
    type=float,
    default=1e-5,
    help="if not None, uses cosine annealing scheduler with the provided value as the minimum lr",
)

# Langevin eval arguments
parser.add_argument(
    "--eval_langevin",
    type=eval,
    default=False,
    help="set True to evaluate Langevin Dynamics during training",
)
parser.add_argument(
    "--langevin_timesteps",
    type=int,
    default=1000000,
    help="number of timesteps per langevin simulation 1M for Alanine, 25M for fast folders",
)
parser.add_argument(
    "--langevin_stepsize",
    type=float,
    default=2e-3,
    help="stepsize resolution for Langevin Simulation in picoseconds",
)
parser.add_argument(
    "--langevin_t_diff",
    type=int,
    nargs="+",
    default=[12],
    help="stepsize resolution for Langevin Simulation in picoseconds",
)
parser.add_argument(
    "--scale_data",
    type=eval,
    default=True,
    help="set True to scale data points by dividing by the dataset's std.",
)
parser.add_argument(
    "--pick_checkpoint",
    type=str,
    default="best",
    help="last to evaluate on the last saved model. Best to evaluate on the best crossvalidated model (which can be noisy sometimes)",
)
parser.add_argument(
    "--start_from_last_saved",
    type=eval,
    default=False,
    help="Load last saved checkpoint and start from there...",
)
parser.add_argument(
    "--iterations_on_val",
    type=float,
    default=5,
    help="how many iterations on the validation partiton",
)

parser.add_argument(
    "--sum_energies",
    type=eval,
    default=True,
    help="this argument is temporal and should be removed",  # TODO: So... Should we remove it?
)
parser.add_argument(
    "--t_diff_interval",
    type=eval,
    default=None,
    help="[0,100], None",
)
parser.add_argument(
    "--loss_weights",
    type=str,
    default="ones",
    help="ones, score_matching, higheruntil_30, higheruntil_100, lower_bound_1000",
)
parser.add_argument(
    "--save_all_checkpoints",
    type=eval,
    default=False,
    help="set True to do save all checkpoints not only the best crossvalidated one",
)

args = parser.parse_args()
args.backbone_network = "graph-transformer"

if "alanine_dipeptide" in args.mol.lower():
    args.shuffle_data_before_splitting = False
else:
    args.shuffle_data_before_splitting = True


if __name__ == "__main__":

    if args.atom_selection == "c-alpha":
        args.atom_selection = AtomSelection.A_CARBON
    elif args.atom_selection == "protein":
        args.atom_selection = AtomSelection.PROTEIN
    elif args.atom_selection == "all":
        args.atom_selection = AtomSelection.ALL
    else:
        raise ValueError("Unknown atom selection")

    tic_evaluator = None
    if args.mol.lower() in all_molecules:
        tic_evaluator = TicEvaluator(
            val_data=None,
            mol_name=args.mol,
            eval_folder=f"./saved_models/{args.mol}",
            data_folder="./datasets/Reference_MD_Sims",
            folded_pdb_folder="./datasets/folded_pdbs",
            bins=101,
            evalset="testset",
            atom_selection=args.atom_selection,
        )

    trainset, valset, testset = get_dataset(
        args.mol,
        args.mean0,
        args.data_folder,
        args.fold,
        atom_selection=args.atom_selection,
        traindata_subset=args.traindata_subset,
        shuffle_before_splitting=args.shuffle_data_before_splitting,
        tic_evaluator=tic_evaluator,
        committor_remove_range=args.committor_remove_range,  # range of committor values to remove
        remove_freq=args.remove_freq,  # frequency of removing clusters from data,
        tetra_atom_selection=args.atom_selection,
    )

    norm_factor = trainset.std if args.scale_data else 1.0

    # Set device
    # Note: Code does not work for cpu in current form
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GNN model
    # For in_node_nf, the features are:
    model = get_model(args, trainset, device)
    print(
        f"Initialized {type(model)} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    # Diffusion model
    if args.flow_matching:
        DDPM_model = FlowMatching(
            model=model,
            features=trainset.bead_onehot,
            num_atoms=trainset.num_beads,
            timesteps=args.diffusion_steps,
            norm_factor=norm_factor,
            loss_weights=args.loss_weights,
            objective="pred_velocity",
        )
    else:

        DDPM_model = GaussianDiffusion(
            model=model,
            features=trainset.bead_onehot,
            num_atoms=trainset.num_beads,
            timesteps=args.diffusion_steps,
            norm_factor=norm_factor,
            loss_weights=args.loss_weights,
        )

    if "tetrapeptides" in args.mol:
        topology = trainset.topoloy if hasattr(trainset, "topology") else None
    else:
        topology = md.load_topology(
            f"./datasets/folded_pdbs/{Molecules[args.mol.upper()].value}-0-{args.atom_selection.value}.pdb"
        )

    # Trainer
    trainer = Trainer(
        DDPM_model.to(device),
        (trainset, valset, testset),
        args.mol,
        args.atom_selection,
        args,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_iter,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        save_and_sample_every=args.eval_interval,
        num_saved_samples=args.num_samples,
        topology=topology,
        results_folder=args.results_folder,
        data_aug=args.data_aug,
        tb_folder=args.tensorboard_folder,
        experiment_name=args.experiment_name,
        weight_decay=args.weight_decay,
        log_tensorboard_interval=args.log_tensorboard_interval,
        num_samples_final_eval=args.num_samples_final_eval,
        min_lr_cosine_anneal=args.min_lr_cosine_anneal,
        warmup_proportion=args.warmup_proportion,
        eval_langevin=args.eval_langevin,
        langevin_timesteps=args.langevin_timesteps,
        langevin_stepsize=args.langevin_stepsize,
        langevin_t_diffs=args.langevin_t_diff,
        start_from_last_saved=args.start_from_last_saved,
        pick_checkpoint=args.pick_checkpoint,
        iterations_on_val=args.iterations_on_val,
        t_diff_interval=args.t_diff_interval,
        save_all_checkpoints=args.save_all_checkpoints,
    )

    # Training
    trainer.train()

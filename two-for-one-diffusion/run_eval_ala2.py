"""Standalone script to run ala2 evaluation on an existing saved run."""
import argparse
from evaluate.evaluate_ala2 import evaluate_ala2
from datasets.dataset_utils_empty import AtomSelection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_mode", type=str, default="om_interpolate")
    parser.add_argument("--append_exp_name", type=str, default=None,
                        help="Suffix after main_eval_output_{gen_mode}. e.g. '' for the base folder.")
    parser.add_argument("--checkpoint_folder", type=str, default="./saved_models")
    parser.add_argument("--reference_folder", type=str, default="./evaluate/saved_references")
    parser.add_argument("--pdb_folder", type=str, default="./datasets")
    parser.add_argument("--num_paths", type=int, default=8)
    parser.add_argument("--no_gif", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    evaluate_ala2(
        gen_mode=args.gen_mode,
        append_exp_name=args.append_exp_name,
        checkpoint_folder=args.checkpoint_folder,
        reference_folder=args.reference_folder,
        pdb_folder=args.pdb_folder,
        atom_selection=AtomSelection.PROTEIN,
        fold=args.fold,
        gif=not args.no_gif,
        num_paths=args.num_paths,
    )

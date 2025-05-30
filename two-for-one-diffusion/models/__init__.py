from models.graph_transformer import GraphTransformer
from models.committor import CommittorNN
from datasets.dataset_utils_empty import AtomSelection


def get_model(args, trainset, device):

    if args.backbone_network == "graph-transformer":
        model = GraphTransformer(
            trainset.num_beads,
            hidden_nf=args.hidden_features_gnn,
            device=device,
            n_layers=args.num_layers_gnn,
            use_intrinsic_coords=args.use_intrinsic_coords,
            use_abs_coords=args.use_abs_coords,
            use_distances=args.use_distances,
            conservative=args.conservative,
            use_bead_identities=args.mol == "tetrapeptides"
            or args.atom_selection == AtomSelection.PROTEIN,
            heads=args.heads if hasattr(args, "heads") else 8,
            dim_head=args.dim_head if hasattr(args, "dim_head") else 64,
        )
    else:
        raise Exception(f"Network { args.backbone_network} not implemented")
    return model

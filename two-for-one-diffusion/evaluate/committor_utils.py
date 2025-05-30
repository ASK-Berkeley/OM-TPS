import os
import numpy as np
import torch
from tqdm import tqdm


def get_gt_committor_probs(
    protein_name, tic_samples, start_cluster, end_cluster, clusters, tic
):
    """Compute the committor probabilities for the ground truth trajectory samples."""
    n_samples = tic_samples.shape[0]

    # Digitize samples into bins
    bins_x = np.digitize(tic_samples[:, 0], tic.bin_edges_x) - 1
    bins_y = np.digitize(tic_samples[:, 1], tic.bin_edges_y) - 1

    # Clip to ensure all indices are within bounds
    bins_x = np.clip(bins_x, 0, tic.bins - 1)
    bins_y = np.clip(bins_y, 0, tic.bins - 1)

    # Create bin index based on (x, y) bin pairs
    bin_idx = bins_x * tic.bins + bins_y
    if os.path.exists(
        f"./evaluate/saved_references/{protein_name}_committor_probs_{start_cluster}_{end_cluster}.npy"
    ):
        committor_probs = np.load(
            f"./evaluate/saved_references/{protein_name}_committor_probs_{start_cluster}_{end_cluster}.npy"
        )
        return committor_probs[bin_idx]

    # Initialize committor probabilities tensor
    print(
        f"Computing empirical committor probabilities from reference simulations for {protein_name}..."
    )
    committor_probs = torch.zeros(tic.bins**2)
    bin_counts = torch.zeros(tic.bins**2)

    # Boolean masks for start and end cluster transitions
    is_start = clusters == start_cluster
    is_end = clusters == end_cluster

    last_committor = None
    last_transition_idx = None

    # Iterate over samples to assign committor probabilities
    for i in tqdm(range(n_samples)):
        bin_counts[bin_idx[i]] += 1
        if last_transition_idx is not None and i <= last_transition_idx:
            # If we already determined the committor for the intermediate steps, assign and skip
            committor_probs[bin_idx[i]] += last_committor
        else:
            if is_start[i]:
                committor_probs[bin_idx[i]] += 0
                last_committor = 0
                last_transition_idx = i
            elif is_end[i]:
                committor_probs[bin_idx[i]] += 1
                last_committor = 1
                last_transition_idx = i
            else:
                # Search forward in the trajectory to find the next transition
                subsequent_clusters = clusters[i:]
                first_start = np.argmax(subsequent_clusters == start_cluster)
                first_end = np.argmax(subsequent_clusters == end_cluster)

                if first_end == 0 or first_start > 0 and first_start < first_end:
                    last_committor = 0
                    last_transition_idx = i + first_start
                elif first_end > 0 and (first_start == 0 or first_end < first_start):
                    last_committor = 1
                    last_transition_idx = i + first_end
                else:
                    raise RuntimeError(f"No transition found for sample {i}")

                # Assign committor for current step
                committor_probs[bin_idx[i]] += last_committor

    # Normalize committor probabilities by bin counts
    committor_probs = committor_probs / torch.clamp(bin_counts, min=1)

    # Save bin-wise committor probabilities
    np.save(
        f"./evaluate/saved_references/{protein_name}_committor_probs_{start_cluster}_{end_cluster}.npy",
        committor_probs,
    )

    return committor_probs[bin_idx]

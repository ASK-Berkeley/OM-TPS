import numpy as np
import torch
from collections import Counter


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    """
    Sample discrete trajectories from a Markov state model transition matrix.
    Adapted from MDGen (https://github.com/bjing2016/mdgen/blob/master/mdgen/analysis.py)
    """
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    """
    Compute the likelihood of a discrete trajectory given a reference MSM transition matrix.
    Adapted from MDGen (https://github.com/bjing2016/mdgen/blob/master/mdgen/analysis.py)
    """
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[
        0, -1
    ]  # final state (this assumes that all trajectories end in the same state)
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_tp_log_likelihood(tp, trans):
    """
    Compute the log-likelihood of a discrete trajectory given a reference
    MSM transition matrix. Adapted from MDGen
    (https://github.com/bjing2016/mdgen/blob/master/mdgen/analysis.py).
    Now returns log probabilities to avoid underflow.
    """
    N = tp.shape[1]
    n_samples = tp.shape[0]

    s_N = tp[0, -1]  # final state (this assumes all trajectories end in the same state)
    log_trans_probs = []

    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]

        # The original function's 'numerator' and 'probs' steps stay the same,
        # but we move to log-space before appending to 'log_trans_probs'.
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        denom = np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        probs = numerator / denom

        # Avoid log of zero or negative by clipping
        probs = np.clip(np.nan_to_num(probs), a_min=1e-15, a_max=None)

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]

        # Convert to log
        log_trans_prob = np.log(trans_prob)
        log_trans_probs.append(log_trans_prob)

    # Stack along the time dimension and sum
    log_probs = np.stack(log_trans_probs, axis=1)

    return log_probs


def discretize_trajectory(xyz, tic_evaluator, cluster_centers, transform=True):
    """
    Trajectory discretization based on nearest cluster centers.
    Args:
        xyz (torch.tensor): Coordinates (in Angstroms) of shape (N_frames, N_residues, 3).
        tic_evaluator (TicEvaluator): TicEvaluator object.
        cluster_centers (torch.tensor): TICA cluster centers of shape (N_clusters, 2).
    Returns:
        assignments (np.ndarray): Cluster assignments of shape (N_frames,).
    """
    # Compute the TIC features for the trajectory
    if transform:
        sample_tic_features = tic_evaluator.get_tic_features(xyz, tic_evaluator.folded)
        transformed_samples = tic_evaluator.tica(sample_tic_features)
    else:
        transformed_samples = xyz

    # Compute the distance between each point in the trajectory and each cluster center
    distances = np.linalg.norm(
        transformed_samples[:, np.newaxis] - cluster_centers, axis=2
    )

    # Assign each point in the trajectory to the nearest cluster center
    assignments = np.argmin(distances, axis=1)

    return assignments, transformed_samples


def compute_stationary_distribution(P):
    # Solve the system of linear equations: pi * P = pi
    # and ensure that sum(pi) = 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    stationary = stationary / np.sum(stationary)  # Normalize to sum to 1
    return stationary.flatten()


# Function to make the transition matrix reversible
def make_reversible(P):
    pi = compute_stationary_distribution(P)  # Stationary distribution
    n = P.shape[0]  # Number of states

    # Create a new reversible transition matrix
    P_rev = np.zeros_like(P)

    for i in range(n):
        for j in range(n):
            # Symmetrize the transition probabilities to enforce detailed balance
            P_rev[i, j] = (pi[i] * P[i, j] + pi[j] * P[j, i]) / (2 * pi[i])

    return P_rev


# Main function to compute the start and end state based on smallest non-zero flux
def find_min_flux_states(T):
    # Step 1: Compute the stationary distribution π
    pi = compute_stationary_distribution(T)

    # Step 2: Compute the flux matrix F
    F = compute_flux(T, pi)

    # Step 3: Find the smallest non-zero flux and return the corresponding state pair
    masked_F = np.ma.masked_equal(F, 0)
    start_state, end_state = np.unravel_index(np.argmin(masked_F), F.shape)

    return start_state, end_state


# Function to compute the flux matrix F = T ⊙ P_i, where P_i is a matrix with stationary distribution π in each column
def compute_flux(T, pi):
    # P_i is a matrix with pi in each column
    P_i = np.outer(pi, np.ones(T.shape[1]))

    # Element-wise multiplication (Hadamard product)
    F = T * P_i
    return F


def remove_consecutive_repeats(path):
    """Helper function to remove consecutive repeats from a path."""
    if len(path) == 0:
        return path
    # Keep only the elements that are different from the previous one
    new_path = [path[0]]  # Start with the first element
    for i in range(1, len(path)):
        if path[i] != path[i - 1]:
            new_path.append(path[i])
    return new_path


def compute_shannon_entropy(paths):
    # Remove consecutive repeats from each path
    cleaned_paths = [remove_consecutive_repeats(path) for path in paths]

    # Count the frequency of each unique path
    path_counts = Counter(
        map(tuple, cleaned_paths)
    )  # Treat each cleaned path as a tuple
    total_paths = len(cleaned_paths)

    # Compute the probability of each path
    probabilities = np.array([count / total_paths for count in path_counts.values()])

    # Compute the Shannon entropy
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

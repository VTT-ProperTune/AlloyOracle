"""Filtering utilities for preselection of candidate compositions.

Contains clustering- and diversity-based preselection used before
Thermo‑Calc validation.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

def LimitNumberOfCompositions(eval_results, config):
    """Limit the number of feasible compositions using kmeans + diversity.

    Picks one lowest-loss sample per cluster and fills remaining slots
    with incremental farthest-point selections to reach the target count.

    Args:
        eval_results (dict): Evaluation dictionary produced by
            `candidate_evaluation.SurrogateEvaluation` containing
            `'x_feasible'`, `'y_feasible'`, and `'losses_feasible'`.
        config (dict): Screening configuration.

    Returns:
        dict: `eval_results` reduced to selected indices.
    """

    n0 = eval_results['x_feasible'].shape[0]
    print('Number of compositions', n0, flush=True)
    
    x = eval_results['x_feasible']
    y = eval_results['y_feasible']
    losses_feasible = eval_results['losses_feasible']['Total loss']

    n_clusters = config['n_compositions_validate_max']
    scaler = StandardScaler()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024*16, random_state=1234)
    x_unscaled = y[config['phase_composition_evaluation_temperature']][config['outputs_a2_composition']]
    x_norm = scaler.fit_transform(x_unscaled)
    kmeans.fit(x_norm)
    labels = kmeans.predict(x_norm)
    clustering_space = x_norm

    selected_indices = []
    for cluster_idx in range(n_clusters):
        cluster_points = np.where(labels == cluster_idx)[0]
        if len(cluster_points) == 0:
            continue
        cluster_losses = losses_feasible.iloc[cluster_points]
        min_loss = np.min(cluster_losses)
        min_loss_indices = cluster_points[np.where(cluster_losses == min_loss)[0]]
        # Pick the first occurrence
        for idx in min_loss_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                break

    print(f"Selected {len(selected_indices)} points with clustering", flush=True)
    # If not enough, use incremental farthest search for diversity
    if len(selected_indices) < n_clusters:
        print(f"Selecting {n_clusters-len(selected_indices)} more points with farthest search", flush=True)
        
        remaining_indices = np.setdiff1d(np.arange(x.shape[0]), selected_indices)
        selected_points = clustering_space[selected_indices]

        # Precompute distances between all remaining points and selected points
        remaining_points = clustering_space[remaining_indices]
        dists = distance.cdist(remaining_points, selected_points)

        # Track minimum distances to selected points
        min_dists = np.min(dists, axis=1)

        while len(selected_indices) < n_clusters and len(remaining_indices) > 0:
            farthest_idx_in_remaining = np.argmax(min_dists)
            farthest_idx = remaining_indices[farthest_idx_in_remaining]
            selected_indices.append(farthest_idx)

            # Update selected points and distances
            new_point = clustering_space[farthest_idx].reshape(1, -1)
            new_dists = distance.cdist(remaining_points, new_point).reshape(-1)
            min_dists = np.minimum(min_dists, new_dists)

            # Remove selected index from remaining
            remaining_indices = np.delete(remaining_indices, farthest_idx_in_remaining)
            remaining_points = np.delete(remaining_points, farthest_idx_in_remaining, axis=0)
            min_dists = np.delete(min_dists, farthest_idx_in_remaining)

    eval_results['x_feasible'] = x.iloc[selected_indices]
    eval_results['y_feasible'] = {T: y[T].iloc[selected_indices, :] for T in y}
    eval_results['losses_feasible'] = eval_results['losses_feasible'].iloc[selected_indices]
    
    print('Number of compositions after preselection', eval_results['x_feasible'].shape[0], flush=True)
    
    return eval_results
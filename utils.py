import numpy as np


def energy_distance_points(X, Y):
    """
    Energy distance between two point clouds X and Y.

    X: array of shape (n, dim)
    Y: array of shape (m, dim)

    Returns a non-negative scalar. 0 iff the two distributions coincide
    in the limit of infinite samples.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n, m = X.shape[0], Y.shape[0]
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        # One process has no points at all, the other has some
        return np.inf

    # pairwise distances
    diff_xy = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)  # (n, m)
    diff_xx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)  # (n, n)
    diff_yy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)  # (m, m)

    return 2.0 * diff_xy.mean() - diff_xx.mean() - diff_yy.mean()


def spatial_poisson_metric(points_model, Ns_model, points_test, Ns_test):
    """
    Metric to compare two spatial Poisson processes:

    - points_model: array of shape (sum_i N_i^model, dim)
    - Ns_model:     array of shape (R_model,), counts per realization for the model
    - points_test:   array of shape (sum_j N_j^test, dim)  (e.g. X_test pooled)
    - Ns_test:       array of shape (R_test,), counts per realization for the reference

    Returns:
        metrics: dict with
            - 'energy_locations': energy distance between pooled locations
            - 'mean_count_diff':  |E[N_model] - E[N_test]|
            - 'var_count_diff':   |Var[N_model] - Var[N_test]|
    """
    points_model = np.asarray(points_model, dtype=float)
    points_ref = np.asarray(points_test, dtype=float)
    Ns_model = np.asarray(Ns_model, dtype=float)
    Ns_test = np.asarray(Ns_test, dtype=float)

    # 1) location distribution metric
    energy_loc = energy_distance_points(points_model, points_ref)

    # 2) count distribution metrics
    mean_model = Ns_model.mean() if Ns_model.size > 0 else 0.0
    mean_ref = Ns_test.mean() if Ns_test.size > 0 else 0.0
    var_model = Ns_model.var(ddof=0) if Ns_model.size > 0 else 0.0
    var_ref = Ns_test.var(ddof=0) if Ns_test.size > 0 else 0.0

    mean_count_diff = abs(mean_model - mean_ref)
    var_count_diff = abs(var_model - var_ref)

    return {
        "energy_locations": float(energy_loc),
        "mean_count_diff": float(mean_count_diff),
        "var_count_diff": float(var_count_diff),
    }

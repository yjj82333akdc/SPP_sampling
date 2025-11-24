import numpy as np
from scipy.stats import multivariate_normal, bernoulli


class Poisson_Gaussian_mixture:
    """
    Inhomogeneous Poisson point process with intensity
        λ(x) = lam * f_GM(x),
    where f_GM is a 2-component isotropic Gaussian mixture in R^dim
    with equal mixture weights 0.5 / 0.5.

    - dim: integer, dimension of the space
    - mean: list or tuple of length 2, means for the two components (scalar per coord)
    - cov: list or tuple of length 2, diagonal cov scalars for the two components
    - lam: scalar, expected number of points per realization (i.e. Poisson mean)
    """

    def __init__(self, dim, mean, cov, lam):
        self.dim = dim
        self.mean_1 = np.full(dim, mean[0], dtype=float)
        self.mean_2 = np.full(dim, mean[1], dtype=float)
        self.cov_1 = np.diag(np.ones(dim) * cov[0])
        self.cov_2 = np.diag(np.ones(dim) * cov[1])
        self.prob = 0.5  # mixture weight for component 1
        self.lam = lam   # Poisson(λ) parameter = expected number of points

    # ----- density (normalized, integrates to 1) -----
    def density_value(self, x_input):
        """
        f_GM(x): mixture density (NOT the Poisson intensity).
        x_input: array-like of shape (dim,) or (N, dim)
        """
        x = np.atleast_2d(x_input)
        gm1 = multivariate_normal.pdf(x, mean=self.mean_1, cov=self.cov_1)
        gm2 = multivariate_normal.pdf(x, mean=self.mean_2, cov=self.cov_2)
        return self.prob * gm1 + (1 - self.prob) * gm2

    # ----- intensity λ(x) = λ * f_GM(x) -----
    def intensity_value(self, x_input):
        """
        λ(x) = lam * f_GM(x)
        """
        return self.lam * self.density_value(x_input)

    # ----- Poisson × Gaussian mixture sampling (single realization) -----
    def sample_poisson(self, random_state=None):
        """
        Draw one realization of the Poisson point process:

        1. N ~ Poisson(lam)
        2. X_1,...,X_N i.i.d. ~ Gaussian mixture f_GM

        Returns:
            points: array of shape (N, dim). If N=0, shape is (0, dim).
            N: integer, number of points.
        """
        rng = np.random.default_rng(random_state)

        # 1. sample N from Poisson(λ)
        N = rng.poisson(self.lam)

        if N == 0:
            return np.empty((0, self.dim)), 0

        # 2. for each point, decide which component (Bernoulli with prob self.prob)
        comp = bernoulli.rvs(self.prob, size=N, random_state=rng).astype(bool)

        # 3. sample from each Gaussian separately and combine
        x1 = rng.multivariate_normal(self.mean_1, self.cov_1, size=N)
        x2 = rng.multivariate_normal(self.mean_2, self.cov_2, size=N)

        points = np.where(comp[:, None], x1, x2)
        return points, N

    # ----- Batched Poisson × Gaussian mixture sampling (flattened) -----
    def sample_poisson_batch(self, N_train, random_state=None):
        """
        Draw N_train independent realizations of the Poisson point process.

        For i = 1..N_train:
            N_i ~ Poisson(lam)
            X^{(i)}_1,...,X^{(i)}_{N_i} i.i.d. ~ Gaussian mixture f_GM

        Returns:
            points_all: array of shape (sum_i N_i, dim),
                        the concatenation of all points from all realizations.
                        If sum_i N_i == 0, shape is (0, dim).
            Ns:         array of shape (N_train,), the Poisson counts N_i.
                        You can recover slice boundaries via Ns.cumsum().
        """
        rng = np.random.default_rng(random_state)

        # Sample all Poisson counts at once
        Ns = rng.poisson(self.lam, size=N_train)
        total_N = int(Ns.sum())

        # If there are no points at all, return an empty array with correct shape
        if total_N == 0:
            return np.empty((0, self.dim)), Ns

        # Allocate flat array for all points
        points_all = np.zeros((total_N, self.dim), dtype=float)

        # Fill in segments corresponding to each realization
        idx = 0
        for N in Ns:
            if N == 0:
                continue

            comp = bernoulli.rvs(self.prob, size=N, random_state=rng).astype(bool)
            x1 = rng.multivariate_normal(self.mean_1, self.cov_1, size=N)
            x2 = rng.multivariate_normal(self.mean_2, self.cov_2, size=N)
            pts = np.where(comp[:, None], x1, x2)

            points_all[idx:idx + N, :] = pts
            idx += N

        return points_all, Ns



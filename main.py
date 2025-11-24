import numpy as np
from Poisson_Gaussian_mixture import Poisson_Gaussian_mixture
from tensor_estimate import vrs_prediction
from utils import spatial_poisson_metric

dim = 3
N_train = 4000
N_test = 4000
N_samples = 4000
MM = 10
lr_rec = 0
kde_rec = 0
lam = 2
distribution = Poisson_Gaussian_mixture(dim, [5, -5], [0.1, 0.1], lam)

if N_train < 2 ** dim * MM:
    print('insufficient data')
    LL = 1
else:
    LL = 2

print(MM, LL)
tensor_shape = [LL for _ in range(dim)]

tensor_shape[0] = MM

#########################################



X_train, Ns_train = distribution.sample_poisson_batch(N_train)
X_test, Ns_test = distribution.sample_poisson_batch(N_test)

#############density transform
vrs_model = vrs_prediction(tensor_shape, dim, MM, X_train, N_train)
y_lr = vrs_model.predict(X_test)
y_true = np.array([distribution.density_value(xx) for xx in X_test])
lr_rec += np.linalg.norm(y_lr - y_true, 2) ** 2 / np.linalg.norm(y_true, 2) ** 2
print('lr transform prediction error = ', lr_rec )



#vrs_Sampling:
samples, Ns_samples, time_elapsed = vrs_model.sample_poisson_spp_batch(N_samples)
print("------------------below is VRS sampling result -------------------")
metrics = spatial_poisson_metric(samples, Ns_samples, X_test, Ns_test)
print(metrics)
print("SPP sampling time cost:", time_elapsed)
print("------------------above is VRS sampling result -------------------")



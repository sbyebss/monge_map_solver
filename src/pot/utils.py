# pylint: disable=invalid-name,line-too-long,no-else-raise
import numpy as np
from einops import repeat
from ot.backend import get_backend
from ot.utils import euclidean_distances
from scipy.spatial.distance import cdist


def sphere_distance(x1, x2):
    r"""Compute the distance between two points on the sphere
    Assume that x1 and x2 are with spherical coordinates (latitude, longitude)
    and they are np.ndarray of shape (n,2), (m,2)
    return: np.ndarray of shape (n,m)
    ([0,2pi], [0, pi])
    """
    x1 = repeat(x1, "n d -> n c d", c=1)
    x2 = repeat(x2, "m d -> c m d", c=1)
    return np.arccos(
        np.sin(x1[:, :, 1]) * np.sin(x2[:, :, 1])
        + np.cos(x1[:, :, 1]) * np.cos(x2[:, :, 1]) * np.cos(x1[:, :, 0] - x2[:, :, 0])
    )


def mask_mse(x1, x2, mask):
    x1 = x1 * (1 - mask)
    x2 = x2 * (1 - mask)
    x1 = repeat(x1, "n d -> n c d", c=1)
    x2 = repeat(x2, "m d -> c m d", c=1)
    return 10000 * np.square(np.linalg.norm(x1 - x2, axis=-1))


def dist(x1, x2=None, metric="sqeuclidean", p=2, w=None, mask=None):
    r"""Compute distance between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------

    x1 : array-like, shape (n1,d)
        matrix with `n1` samples of size `d`
    x2 : array-like, shape (n2,d), optional
        matrix with `n2` samples of size `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)
    metric : str | callable, optional
        'sqeuclidean' or 'euclidean' on all backends. On numpy the function also
        accepts  from the scipy.spatial.distance.cdist function : 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    p : float, optional
        p-norm for the Minkowski and the Weighted Minkowski metrics. Default value is 2.
    w : array-like, rank 1
        Weights for the weighted metrics.


    Returns
    -------

    M : array-like, shape (`n1`, `n2`)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    elif metric == "sphere":
        return sphere_distance(x1, x2)
    elif metric == "mask_mse":
        return mask_mse(x1, x2, mask)
    else:
        if not get_backend(x1, x2).__name__ == "numpy":
            raise NotImplementedError()
        else:
            if isinstance(metric, str) and metric.endswith("minkowski"):
                return cdist(x1, x2, metric=metric, p=p, w=w)
            return cdist(x1, x2, metric=metric, w=w)

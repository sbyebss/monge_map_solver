# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches,too-many-instance-attributes,attribute-defined-outside-init,redefined-outer-name,unbalanced-tuple-unpacking,inconsistent-return-statements

from ot.backend import get_backend
from ot.lp import emd
from ot.optim import cg
from ot.utils import BaseEstimator, check_params, dots, kernel, list_to_array, unif

from src.pot.utils import dist


def joint_OT_mapping_linear(
    xs,
    xt,
    mu=1,
    eta=0.001,
    bias=False,
    verbose=False,
    metric="sqeuclidean",
    mask=None,
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
):
    r"""Joint OT and linear mapping estimation as proposed in
    :ref:`[8] <references-joint-OT-mapping-linear>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma,L}\quad \|L(\mathbf{X_s}) - n_s\gamma \mathbf{X_t} \|^2_F +
          \mu \langle \gamma, \mathbf{M} \rangle_F + \eta \|L - \mathbf{I}\|^2_F

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`d\times d` linear operator that approximates the barycentric
      mapping
    - :math:`\mathbf{I}` is the identity matrix (neutral linear mapping)
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and a linear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-linear>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditionnal gradient)
    and the update of :math:`\mathbf{L}` using a classical least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    bias : bool,optional
        Estimate linear mapping with constant bias
    numItermax : int, optional
        Max number of BCD iterations
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (d, d) array-like
        Linear mapping matrix ((:math:`d+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-linear:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
        "Mapping estimation for discrete optimal transport",
        Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt, d = xs.shape[0], xt.shape[0], xt.shape[1]

    if bias:
        xs1 = nx.concatenate((xs, nx.ones((ns, 1), type_as=xs)), axis=1)
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d + 1, type_as=xs)
        Id[-1] = 0
        I0 = Id[:, :-1]

        def sel(x):
            return x[:-1, :]

    else:
        xs1 = xs
        xstxs = nx.dot(xs1.T, xs1)
        Id = nx.eye(d, type_as=xs)
        I0 = Id

        def sel(x):
            return x

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    # This is where to define the cost
    M = dist(xs, xt, metric=metric, mask=mask) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(xs1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.sum(sel(L - I0) ** 2)
        )

    def solve_L(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(xstxs + eta * Id, nx.dot(xs1.T, xst) + eta * I0)

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(xs1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

        G = cg(
            a,
            b,
            M,
            1.0 / mu,
            f,
            df,
            G0=G0,
            numItermax=numInnerItermax,
            stopThr=stopInnerThr,
        )
        return G

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:

        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print(
                "{:5d}|{:8e}|{:8e}".format(
                    it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])
                )
            )
    if log:
        log["loss"] = vloss
        return G, L, log
    else:
        return G, L


def joint_OT_mapping_kernel(
    xs,
    xt,
    mu=1,
    eta=0.001,
    kerneltype="gaussian",
    sigma=1,
    bias=False,
    verbose=False,
    mask=None,
    metric="sqeuclidean",
    numItermax=100,
    numInnerItermax=10,
    stopInnerThr=1e-6,
    stopThr=1e-5,
    log=False,
):
    r"""Joint OT and nonlinear mapping estimation with kernels as proposed in
    :ref:`[8] <references-joint-OT-mapping-kernel>`.

    The function solves the following optimization problem:

    .. math::
        \min_{\gamma, L\in\mathcal{H}}\quad \|L(\mathbf{X_s}) -
        n_s\gamma \mathbf{X_t}\|^2_F + \mu \langle \gamma, \mathbf{M} \rangle_F +
        \eta \|L\|^2_\mathcal{H}

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) squared euclidean cost matrix between samples in
      :math:`\mathbf{X_s}` and :math:`\mathbf{X_t}` (scaled by :math:`n_s`)
    - :math:`L` is a :math:`n_s \times d` linear operator on a kernel matrix that
      approximates the barycentric mapping
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are uniform source and target weights

    The problem consist in solving jointly an optimal transport matrix
    :math:`\gamma` and the nonlinear mapping that fits the barycentric mapping
    :math:`n_s\gamma \mathbf{X_t}`.

    One can also estimate a mapping with constant bias (see supplementary
    material of :ref:`[8] <references-joint-OT-mapping-kernel>`) using the bias optional argument.

    The algorithm used for solving the problem is the block coordinate
    descent that alternates between updates of :math:`\mathbf{G}` (using conditionnal gradient)
    and the update of :math:`\mathbf{L}` using a classical kernel least square solver.


    Parameters
    ----------
    xs : array-like (ns,d)
        samples in the source domain
    xt : array-like (nt,d)
        samples in the target domain
    mu : float,optional
        Weight for the linear OT loss (>0)
    eta : float, optional
        Regularization term  for the linear mapping L (>0)
    kerneltype : str,optional
        kernel used by calling function :py:func:`ot.utils.kernel` (gaussian by default)
    sigma : float, optional
        Gaussian kernel bandwidth.
    bias : bool,optional
        Estimate linear mapping with constant bias
    verbose : bool, optional
        Print information along iterations
    verbose2 : bool, optional
        Print information along iterations
    numItermax : int, optional
        Max number of BCD iterations
    numInnerItermax : int, optional
        Max number of iterations (inner CG solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner CG solver) (>0)
    stopThr : float, optional
        Stop threshold on relative loss decrease (>0)
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    L : (ns, d) array-like
        Nonlinear mapping matrix ((:math:`n_s+1`, `d`) if bias)
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-joint-OT-mapping-kernel:
    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
       "Mapping estimation for discrete optimal transport",
       Neural Information Processing Systems (NIPS), 2016.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    xs, xt = list_to_array(xs, xt)
    nx = get_backend(xs, xt)

    ns, nt = xs.shape[0], xt.shape[0]

    K = kernel(xs, xs, method=kerneltype, sigma=sigma)
    if bias:
        K1 = nx.concatenate((K, nx.ones((ns, 1), type_as=xs)), axis=1)
        Id = nx.eye(ns + 1, type_as=xs)
        Id[-1] = 0
        Kp = nx.eye(ns + 1, type_as=xs)
        Kp[:ns, :ns] = K

        # ls regu
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # RKHS regul
        K0 = nx.dot(K1.T, K1) + eta * Kp
        Kreg = Kp

    else:
        K1 = K
        Id = nx.eye(ns, type_as=xs)

        # ls regul
        # K0 = K1.T.dot(K1)+eta*I
        # Kreg=I

        # proper kernel ridge
        K0 = K + eta * Id
        Kreg = K

    if log:
        log = {"err": []}

    a = unif(ns, type_as=xs)
    b = unif(nt, type_as=xt)
    M = dist(xs, xt, metric=metric, mask=mask) * ns
    G = emd(a, b, M)

    vloss = []

    def loss(L, G):
        """Compute full loss"""
        return (
            nx.sum((nx.dot(K1, L) - ns * nx.dot(G, xt)) ** 2)
            + mu * nx.sum(G * M)
            + eta * nx.trace(dots(L.T, Kreg, L))
        )

    def solve_L_nobias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, xst)

    def solve_L_bias(G):
        """solve L problem with fixed G (least square)"""
        xst = ns * nx.dot(G, xt)
        return nx.solve(K0, nx.dot(K1.T, xst))

    def solve_G(L, G0):
        """Update G with CG algorithm"""
        xsi = nx.dot(K1, L)

        def f(G):
            return nx.sum((xsi - ns * nx.dot(G, xt)) ** 2)

        def df(G):
            return -2 * ns * nx.dot(xsi - ns * nx.dot(G, xt), xt.T)

        G = cg(
            a,
            b,
            M,
            1.0 / mu,
            f,
            df,
            G0=G0,
            numItermax=numInnerItermax,
            stopThr=stopInnerThr,
        )
        return G

    if bias:
        solve_L = solve_L_bias
    else:
        solve_L = solve_L_nobias

    L = solve_L(G)

    vloss.append(loss(L, G))

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(0, vloss[-1], 0))

    # init loop
    if numItermax > 0:
        loop = 1
    else:
        loop = 0
    it = 0

    while loop:

        it += 1

        # update G
        G = solve_G(L, G)

        # update L
        L = solve_L(G)

        vloss.append(loss(L, G))

        if it >= numItermax:
            loop = 0

        if abs(vloss[-1] - vloss[-2]) / abs(vloss[-2]) < stopThr:
            loop = 0

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print(
                "{:5d}|{:8e}|{:8e}".format(
                    it, vloss[-1], (vloss[-1] - vloss[-2]) / abs(vloss[-2])
                )
            )
    if log:
        log["loss"] = vloss
        return G, L, log
    else:
        return G, L


class MappingTransport(BaseEstimator):

    """MappingTransport: DA methods that aims at jointly estimating a optimal
    transport coupling and the associated mapping

    Parameters
    ----------
    mu : float, optional (default=1)
        Weight for the linear OT loss (>0)
    eta : float, optional (default=0.001)
        Regularization term for the linear mapping `L` (>0)
    bias : bool, optional (default=False)
        Estimate linear mapping with constant bias
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    kernel : string, optional (default="linear")
        The kernel to use either linear or gaussian
    sigma : float, optional (default=1)
        The gaussian kernel parameter
    max_iter : int, optional (default=100)
        Max number of BCD iterations
    tol : float, optional (default=1e-5)
        Stop threshold on relative loss decrease (>0)
    max_inner_iter : int, optional (default=10)
        Max number of iterations (inner CG solver)
    inner_tol : float, optional (default=1e-6)
        Stop threshold on error (inner CG solver) (>0)
    log : bool, optional (default=False)
        record log if True
    verbose : bool, optional (default=False)
        Print information along iterations
    verbose2 : bool, optional (default=False)
        Print information along iterations

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    mapping_ :
        The associated mapping

        - array-like, shape (`n_features` (+ 1), `n_features`),
          (if bias) for kernel == linear

        - array-like, shape (`n_source_samples` (+ 1), `n_features`),
          (if bias) for kernel == gaussian
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
            "Mapping estimation for discrete optimal transport",
            Neural Information Processing Systems (NIPS), 2016.

    """

    def __init__(
        self,
        mu=1,
        eta=0.001,
        bias=False,
        metric="sqeuclidean",
        norm=None,
        kernel="linear",
        sigma=1,
        max_iter=100,
        tol=1e-5,
        max_inner_iter=10,
        inner_tol=1e-6,
        log=False,
        mask=None,
        verbose=False,
        verbose2=False,
    ):
        self.metric = metric
        self.norm = norm
        self.mu = mu
        self.eta = eta
        self.bias = bias
        self.mask = mask
        self.kernel = kernel
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.verbose2 = verbose2

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        r"""Builds an optimal coupling and estimates the associated mapping
        from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self
        """
        self._get_backend(Xs, ys, Xt, yt)

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            self.xs_ = Xs
            self.xt_ = Xt

            if self.kernel == "linear":
                returned_ = joint_OT_mapping_linear(
                    Xs,
                    Xt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    verbose=self.verbose,
                    metric=self.metric,
                    mask=self.mask,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopThr=self.tol,
                    stopInnerThr=self.inner_tol,
                    log=self.log,
                )

            elif self.kernel == "gaussian":
                returned_ = joint_OT_mapping_kernel(
                    Xs,
                    Xt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    sigma=self.sigma,
                    verbose=self.verbose,
                    metric=self.metric,
                    mask=self.mask,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopInnerThr=self.inner_tol,
                    stopThr=self.tol,
                    log=self.log,
                )

            # deal with the value of log
            if self.log:
                self.coupling_, self.mapping_, self.log_ = returned_
            else:
                self.coupling_, self.mapping_ = returned_
                self.log_ = {}

        return self

    def transform(self, Xs):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):
                # perform standard barycentric mapping
                transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                if self.kernel == "gaussian":
                    K = kernel(Xs, self.xs_, method=self.kernel, sigma=self.sigma)
                elif self.kernel == "linear":
                    K = Xs
                if self.bias:
                    K = nx.concatenate(
                        [K, nx.ones((Xs.shape[0], 1), type_as=K)], axis=1
                    )
                transp_Xs = nx.dot(K, self.mapping_)

            return transp_Xs

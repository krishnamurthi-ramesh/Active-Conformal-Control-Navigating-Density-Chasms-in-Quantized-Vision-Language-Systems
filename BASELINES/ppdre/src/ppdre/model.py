"""Projection Pursuit Density Ratio Estimation(PPDRE) model."""

import time
from functools import partial
from typing import NamedTuple, Tuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array, random
from jax.typing import ArrayLike
from tqdm import trange

from ppdre._basis_functions import _expand_gaussian_basis_function
from ppdre._utils import _truncate_to_min_positive

Data = tuple[ArrayLike, ArrayLike]


class _SieveModelTrainableParams(NamedTuple):
    projection: ArrayLike
    bf_args: ArrayLike


def _init_sieve_model_trainable_params(
    df: int,
    p: int,
    random_key: ArrayLike,
) -> _SieveModelTrainableParams:
    [random_key1, random_key2] = random.split(random_key, 2)
    projection = random.normal(random_key1, (p,))
    bf_args = random.normal(
        random_key2,
        (df, 1),
    )
    return _SieveModelTrainableParams(projection, bf_args)


class _SieveModelState(NamedTuple):
    params: _SieveModelTrainableParams
    coefficient: ArrayLike


@jax.jit
def _sieve_model_compute_basis(
    params: _SieveModelTrainableParams,
    x: ArrayLike,
) -> Array:
    x = jnp.asarray(x)
    px = jnp.matmul(x, params.projection)
    return jnp.hstack(
        [
            jnp.ones((x.shape[0], 1)),
            _expand_gaussian_basis_function(px, params.bf_args),
        ]
    )


@jax.jit
def _sieve_model(
    state: _SieveModelState,
    x: ArrayLike,
) -> Array:
    phis = _sieve_model_compute_basis(state.params, x)
    return jnp.matmul(phis, state.coefficient)


@jax.jit
def _sq_loss(nu_r: ArrayLike, de_r: ArrayLike, *, penalty: float = 0) -> Array:
    nu_mean_r = jnp.mean(nu_r)
    de_mean_r2 = jnp.mean(de_r**2)
    return de_mean_r2 - 2 * nu_mean_r + penalty


class _TrainState:
    networks: list[_SieveModelState]
    k: int | None = None

    def __init__(self):
        self.networks = []

    def _get_k(self, k: int | None) -> int:
        if k is not None:
            return k
        if self.k is not None:
            return self.k
        return len(self.networks)


@partial(
    jax.jit,
    static_argnames=(
        "state",
        "k",
    ),
)
def _ppe_predict(state: _TrainState, x: ArrayLike, k=None) -> Array:
    x = jnp.asarray(x)

    k = state._get_k(k)

    res = jnp.ones((x.shape[0], 1))
    for network in state.networks[:k]:
        res *= _truncate_to_min_positive(_sieve_model(network, x)).reshape(-1, 1)
    return res


@partial(
    jax.jit,
    static_argnames=(
        "state",
        "k",
    ),
)
def _ppe_eval(
    state: _TrainState,
    data: Data,
    k: int | None = None,
) -> Array:
    x, y = data
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    k = state._get_k(k)

    nu_r = _ppe_predict(state, x=x, k=k)
    de_r = _ppe_predict(state, x=y, k=k)

    return _sq_loss(nu_r, de_r)


@jax.jit
def _ppe_loss(
    params: _SieveModelTrainableParams,
    x: ArrayLike,
    y: ArrayLike,
    nu_pre_r: ArrayLike,
    de_pre_r: ArrayLike,
    ridge_lam: float,
) -> Tuple[Array, Array]:
    nu_pre_r = jnp.asarray(nu_pre_r)
    de_pre_r = jnp.asarray(de_pre_r)

    nu_phis = _sieve_model_compute_basis(params, x)
    de_phis = _sieve_model_compute_basis(params, y)
    z = de_phis * de_pre_r.reshape(-1, 1)
    w = (nu_phis * nu_pre_r.reshape(-1, 1)).mean(axis=0)
    ztz = jnp.matmul(z.T, z) / z.shape[0]

    reg_term = ridge_lam * jnp.eye(z.shape[1])
    inv_part = ztz + reg_term
    lower = jax.scipy.linalg.cholesky(inv_part, lower=True)
    coefficient = jax.scipy.linalg.cho_solve((lower, True), w.reshape(-1, 1))

    nu_cur_r = _truncate_to_min_positive(
        jnp.matmul(nu_phis, coefficient).reshape(-1, 1)
    )
    de_cur_r = _truncate_to_min_positive(
        jnp.matmul(de_phis, coefficient).reshape(-1, 1)
    )

    nu_r = nu_pre_r * nu_cur_r
    de_r = de_pre_r * de_cur_r

    penalty = ridge_lam * jnp.sum(coefficient**2)

    return _sq_loss(nu_r, de_r, penalty=penalty), coefficient


class ProjectionPursuitEstimator:
    """Projection Pursuit Estimator."""

    _state: _TrainState | None = None

    def predict(self, x: ArrayLike, k=None) -> Array:
        """
        Predict the estimated density ratio.

        Args:
            x: the data points on which the density ratio is estimated.
            k: the number of layers used to estimate the density ratio (default: None).

        Returns:
            the estimated density ratios.
        """
        return _ppe_predict(self._state, x, k)

    def eval(
        self,
        data: Data,
        k: int | None = None,
    ) -> Array:
        """
        Compute SQ loss for the data.

        Args:
            data: the data the model is going to be evaluated on.
                  It should be a tuple (nu_data, de_data) of NumPy or JAX arrays.
            k: the number of layers used to estimate density ratio (default: None).

        Returns:
            the evaluated SQ loss.
        """
        if self._state is None:
            raise RuntimeError("state should be initialized using train")
        return _ppe_eval(self._state, data, k)

    def _train_step(
        self,
        data: Data,
        lr: float,
        random_key: ArrayLike,
        *,
        df: int,
        ridge_lam: float,
        nepoch=100,
        verbose=2,
    ):
        if self._state is None:
            raise RuntimeError("state should be initialized using train")

        x, y = data
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        p = x.shape[1]
        start_t = time.time()

        random_key, random_subkey = random.split(random_key)
        trainable_params = _init_sieve_model_trainable_params(df, p, random_subkey)

        optimizer = optax.adam(lr)
        clipper = optax.clip_by_global_norm(max_norm=1.0)
        optimizer = optax.chain(clipper, optimizer)
        opt_state = optimizer.init(cast(optax.Params, trainable_params))

        losses = []
        epoch_loss = 0
        i = -1
        nu_pre_r = _ppe_predict(
            self._state,
            x=x,
            k=len(self._state.networks),
        )
        de_pre_r = _ppe_predict(
            self._state,
            x=y,
            k=len(self._state.networks),
        )
        coefficient = None
        for i in trange(nepoch, leave=False, disable=verbose < 2):
            if i > 1 and abs(losses[i - 2] - losses[i - 1]) < 1e-5:
                break

            (epoch_loss, coefficient), grads = jax.value_and_grad(
                _ppe_loss, has_aux=True
            )(trainable_params, x, y, nu_pre_r, de_pre_r, ridge_lam)

            updates, opt_state = optimizer.update(grads, opt_state)
            trainable_params = cast(
                _SieveModelTrainableParams,
                optax.apply_updates(cast(optax.Params, trainable_params), updates),
            )

            losses.append(epoch_loss)

            if i % 10 == 0 and verbose > 2:
                print(
                    f"iteration {len(self._state.networks) + 1}"
                    f", epoch {i}"
                    f", loss = {epoch_loss}"
                )
            if i > 0 and abs(losses[i - 1] - losses[i]) < 1e-6:
                break
            if len(losses) >= 20 and np.mean(losses[-5:]) > np.mean(losses[-20:-5]):
                break

        if verbose >= 1:
            print(
                f"iteration {len(self._state.networks) + 1}"
                f", epoch {i}"
                f", loss = {epoch_loss}"
                f", took = {time.time() - start_t} seconds"
            )

        if coefficient is None:
            raise RuntimeError("coefficient should not be none")

        self._state.networks.append(_SieveModelState(trainable_params, coefficient))

    def train(
        self,
        train_data: Data,
        test_data: Data | None = None,
        max_k=None,
        epochs=100,
        lr=1e-2,
        *,
        df=20,
        ridge_lam=0.01,
        seed=1,
        verbose=2,
    ):
        """
        Train the estimator iteratively.

        Args:
            train_data: the data the model is going to be trained on.
                        It should be a tuple (nu_data, de_data) of NumPy or JAX arrays.
            test_data: the data the model is evaluated on,
                        It should be a tuple (nu_data, de_data) of NumPy or JAX arrays.
                        If set to None, the train_data is used instead (default: None).
            max_k: the maximum layers (iterations) for the projection pursuit estimator.
                   If set to None, the data dimension will be used (default: None).
            epochs: the epochs per layer (iteration) (default: 100).
            lr: the learning rate.
            df: the number of basis functions.
            ridge_lam: the hyperparameter for ridge (l2) penalty (default: 0.01).
            seed: the random seed for reproducability.
            verbose: the verbosity, setting it to:
                     * 0: silent,
                     * 1: prints less information,
                     * 2: enables progress bar and prints information of layers
                     * 3: prints all information
                     (default: 2)
        """
        if max_k is None:
            x = jnp.asarray(train_data[0])
            max_k = x.shape[1]

        if test_data is None:
            test_data = train_data

        # clear old results
        self._state = _TrainState()

        random_key = random.key(seed)

        test_loss_history = []
        previous_test_loss = self.eval(test_data, k=0).item()
        for k in range(1, max_k + 1):
            random_key, random_subkey = random.split(random_key)
            self._train_step(
                data=train_data,
                lr=lr,
                random_key=random_subkey,
                df=df,
                ridge_lam=ridge_lam,
                nepoch=epochs,
                verbose=verbose,
            )

            test_loss = self.eval(test_data, k=k).item()
            test_loss_history.append(test_loss)

            loss_d = previous_test_loss - test_loss
            if verbose >= 1:
                print(
                    f"On k = {k} model"
                    f", the loss decrease on test set = {loss_d}"
                    f", pre test loss = {previous_test_loss}"
                    f", cur test loss = {test_loss}"
                )
            previous_test_loss = test_loss

            if loss_d <= 0:
                break
        self._state.k = int(np.argmin(test_loss_history) + 1)
        if self._state.k is None:
            self._state.k = 1
        if verbose >= 1:
            print(f"Training ended with best K={self._state.k}")

    def get_best_k(self):
        """
        Get the best K.

        Returns:
            the best K
        """
        if self._state is None:
            raise RuntimeError("state should be initialized using train")
        return self._state.k

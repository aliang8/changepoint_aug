import jax
import optax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from tensorflow_probability.substrates import jax as tfp
import dataclasses
from typing import NamedTuple
from utils import create_masks

dist = tfp.distributions


@dataclasses.dataclass
class GaussianPolicy(hk.Module):
    hidden_size: int
    action_dim: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        net = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )
        x = net(x)
        # predict mean and log_stddev
        mean = hk.Linear(self.action_dim)(x)
        log_stddev = hk.Linear(self.action_dim)(x)
        # clamp log_stddev
        log_stddev = jnp.clip(log_stddev, -20, 2)
        stddev = jnp.exp(log_stddev)
        action_dist = dist.Normal(loc=mean, scale=stddev)
        action = action_dist.sample(seed=hk.next_rng_key())
        log_prob = action_dist.log_prob(action)
        return action, action_dist, log_prob


# train ensemble of policies
class Policy(hk.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

    def __call__(self, obs):
        obs = hk.Flatten()(obs)
        obs = hk.nets.MLP([self.hidden_size, self.hidden_size, self.action_dim])(obs)
        return obs


class QFunction(hk.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def __call__(self, obs, action):
        obs = hk.Flatten()(obs)
        input = jnp.concatenate([obs, action], axis=-1)
        q = hk.nets.MLP(
            [self.hidden_size, self.hidden_size, 1],
            activation=jax.nn.leaky_relu,
        )(input)
        return q


@hk.transform
def q_fn(obs, action, hidden_size):
    return QFunction(hidden_size)(obs, action)


@dataclasses.dataclass
class Encoder(hk.Module):
    """Encoder model."""

    latent_size: int
    hidden_size: int

    def __call__(self, obs: jax.Array, cond: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Encodes state + cond as an isotropic Guassian latent code.
        2-layer MLP with leaky ReLU activations.

        Outputs mean and std parameters of the Gaussian distribution.
        """
        cond_embed = hk.Linear(self.hidden_size)(cond)  # embed the cond first
        obs_cond = jnp.concatenate([obs, cond_embed], axis=-1)
        output = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )(obs_cond)
        mean = hk.Linear(self.latent_size)(output)
        log_stddev = hk.Linear(self.latent_size)(output)
        # clamp this
        log_stddev = jnp.clip(log_stddev, -20, 2)
        stddev = jnp.exp(log_stddev)
        return mean, stddev


@dataclasses.dataclass
class Decoder(hk.Module):
    """Decoder model."""

    hidden_size: int
    obs_dim: int
    cond_dim: int

    def __call__(self, z: jax.Array, cond: jax.Array) -> jax.Array:
        cond_embed = hk.Linear(self.hidden_size)(cond)
        z = jnp.concatenate([z, cond_embed], axis=-1)
        """Decodes a latent code into observation."""
        output = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.obs_dim),
            ]
        )(z)
        return output


class VAEOutput(NamedTuple):
    mean: jax.Array
    stddev: jax.Array
    z: jax.Array
    recon: jax.Array
    prior_mean: jax.Array
    prior_stddev: jax.Array


class ConditionalPrior(hk.Module):
    """
    p(z | conditioning info)
    """

    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size

    def __call__(self, cond: jnp.ndarray):
        output = hk.Sequential(
            [
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
                hk.Linear(self.hidden_size),
                jax.nn.leaky_relu,
            ]
        )(cond)
        mean = hk.Linear(self.latent_size)(output)
        log_stddev = hk.Linear(self.latent_size)(output)
        log_stddev = jnp.clip(log_stddev, -20, 2)
        stddev = jnp.exp(log_stddev)
        return mean, stddev


@dataclasses.dataclass
class VAE(hk.Module):
    """Main VAE model class, uses Encoder & Decoder under the hood.

    Models a conditional distribution p(s | z, g) and q(z | s, g).
    """

    encoder: Encoder
    decoder: Decoder
    prior: ConditionalPrior

    def __call__(self, obs: jnp.ndarray, cond: jnp.ndarray) -> VAEOutput:
        obs = obs.astype(jnp.float32)
        cond = cond.astype(jnp.float32)

        # q(z| s, g)
        mean, stddev = self.encoder(obs, cond)

        latent_dist = dist.Normal(loc=mean, scale=stddev)

        # reparameterization trick
        z = latent_dist.sample(seed=hk.next_rng_key())

        # p(s | z, g)
        obs_pred = self.decoder(z, cond)

        # prior(z | g)
        prior_mean, prior_stddev = self.prior(cond)
        return VAEOutput(
            mean=mean,
            stddev=stddev,
            z=z,
            recon=obs_pred,
            prior_mean=prior_mean,
            prior_stddev=prior_stddev,
        )

    def decode(self, cond: jnp.ndarray, z: jnp.ndarray = None) -> jnp.ndarray:
        if z is None:
            # sample from prior
            prior_mean, prior_stddev = self.prior(cond)
            z = dist.Normal(loc=prior_mean, scale=prior_stddev).sample(
                seed=hk.next_rng_key()
            )
        obs_pred = self.decoder(z, cond)
        return obs_pred


@hk.transform
def decode_fn(z, cond, latent_size, hidden_size, obs_dim, cond_dim):
    encoder = Encoder(latent_size=latent_size, hidden_size=hidden_size)
    decoder = Decoder(hidden_size=hidden_size, obs_dim=obs_dim, cond_dim=cond_dim)
    prior = ConditionalPrior(latent_size=latent_size, hidden_size=hidden_size)
    vae = VAE(encoder=encoder, decoder=decoder, prior=prior)
    return vae.decode(cond, z)


@hk.transform
def get_prior_fn(cond, latent_size, hidden_size):
    prior = ConditionalPrior(latent_size=latent_size, hidden_size=hidden_size)
    prior_mean, prior_stddev = prior(cond)
    return dist.Normal(loc=prior_mean, scale=prior_stddev)


@hk.transform
def vae_fn(obs, cond, latent_size, hidden_size, obs_dim, cond_dim):
    encoder = Encoder(latent_size=latent_size, hidden_size=hidden_size)
    decoder = Decoder(hidden_size=hidden_size, obs_dim=obs_dim, cond_dim=cond_dim)
    prior = ConditionalPrior(latent_size=latent_size, hidden_size=hidden_size)
    vae = VAE(encoder=encoder, decoder=decoder, prior=prior)
    return vae(obs, cond)


class MaskedLinear(hk.Module):
    """MADE building block layer"""

    def __init__(self, n_outputs: int, mask: jnp.ndarray, cond_size=None):
        super().__init__()
        self.cond_size = cond_size
        self.n_outputs = n_outputs
        self.mask = mask

    def __call__(self, x, y=None):
        input_size = x.shape[-1]
        w = hk.get_parameter("w", [input_size, self.n_outputs], init=jnp.ones)
        b = hk.get_parameter("b", [self.n_outputs], init=jnp.zeros)
        out = jnp.dot(x, w * self.mask) + b
        if y is not None:
            w_cond = hk.get_parameter(
                "w_cond", [self.cond_size, self.n_outputs], init=jnp.ones
            )
            cond_out = jnp.dot(y, w_cond)
            out = out + cond_out
        return out


class MaskedAF(hk.Module):
    """
    Masked Autoregressive Flow
    https://arxiv.org/pdf/1705.07057.pdf
    """

    def __init__(
        self, input_size: int, hidden_size: int, cond_size: int, num_layers: int
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # base distribution for calculation of log prob under the model
        self.base_dist_mean = hk.get_parameter(
            "base_dist_mean", [input_size], init=jnp.zeros
        )
        self.base_dist_var = hk.get_parameter(
            "base_dist_var", [input_size], init=jnp.ones
        )

        # create masks
        masks, self.input_degrees = create_masks(
            input_size,
            hidden_size,
            num_layers,
            input_order="sequential",
            input_degrees=None,
        )

        self.net_input = MaskedLinear(hidden_size, masks[0], cond_size=cond_size)
        net = []
        for m in masks[1:-1]:
            net.append(jax.nn.leaky_relu)
            net.append(MaskedLinear(hidden_size, m))
        net.append(jax.nn.leaky_relu)
        net.append(MaskedLinear(2 * input_size, masks[-1].repeat(2, 1)))
        self.net = hk.Sequential(net)

    @property
    def base_dist(self):
        return dist.Normal(self.base_dist_mean, self.base_dist_var)

    def __call__(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        inp = self.net_input(x, y)
        net_out = self.net(inp)
        m, loga = jnp.split(net_out, 2, axis=1)
        u = (x - m) * jnp.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = jnp.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            inp = self.net_input(x, y)
            net_out = self.net(inp)
            m, loga = jnp.split(net_out, 2, axis=1)
            x[:, i] = u[:, i] * jnp.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return jnp.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


@hk.transform
def maf_fn(obs, cond, input_size, hidden_size, cond_size, num_layers):
    u, log_abs_det_jacobian = MaskedAF(input_size, hidden_size, cond_size, num_layers)(
        obs, y=cond
    )
    return u, log_abs_det_jacobian


def maf_logp_fn(obs, cond, input_size, hidden_size, cond_size, num_layers):
    model = MaskedAF(input_size, hidden_size, cond_size, num_layers)
    logp = model.log_prob(obs, y=cond)
    return logp

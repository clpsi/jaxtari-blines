"""Latent-model MuZero for JAXtari.

The implementation follows MuZero (Schrittwieser et al., 2019): representation,
dynamics, and prediction networks; latent-space MCTS; and trajectory replay with
unrolled policy, value, and reward losses.
"""

import os
import random
import time
from functools import partial

import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training.train_state import TrainState
from rtpt import RTPT

import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, LogWrapper, NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper

from agents.muzero.muzero_eval import evaluate


def make_env(env_id, mods=None, pixel_based=True, native_downscaling=True, eval=False):
    """Build the wrapped JAXtari environment used for collection or evaluation."""
    if mods is not None and not isinstance(mods, list):
        raise TypeError("mods must be a list of strings or None")

    def thunk():
        env = jaxatari.make(env_id, mods=mods or None)
        env = AtariWrapper(env, sticky_actions=0.0, episodic_life=not eval, first_fire=True, noop_max=30, full_action_space=False)
        if pixel_based:
            env = PixelObsWrapper(env, do_pixel_resize=True, pixel_resize_shape=(84, 84), grayscale=True,
                                  use_native_downscaling=native_downscaling, smooth_image=False,
                                  frame_stack_size=4, frame_skip=4, max_pooling=True, clip_reward=not eval)
        else:
            env = FlattenObservationWrapper(NormalizeObservationWrapper(ObjectCentricWrapper(
                env, frame_stack_size=4, frame_skip=4, clip_reward=not eval)))
        return LogWrapper(env)
    return thunk


def value_transform(x, eps=0.001):
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


def inverse_value_transform(x, eps=0.001):
    root = jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps))
    return jnp.sign(x) * (jnp.square((root - 1.0) / (2.0 * eps)) - 1.0)


def scalar_to_support(x, support_size, eps=0.001):
    """Two-hot encode scalar targets on MuZero's transformed support."""
    x = jnp.clip(value_transform(x, eps), -support_size, support_size)
    lower = jnp.floor(x).astype(jnp.int32)
    upper_weight = x - lower
    atoms = 2 * support_size + 1
    index = lower + support_size
    return (jax.nn.one_hot(index, atoms) * (1.0 - upper_weight)[..., None]
            + jax.nn.one_hot(jnp.minimum(index + 1, atoms - 1), atoms) * upper_weight[..., None])


def support_to_scalar(logits, support_size, eps=0.001):
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    return inverse_value_transform(jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1), eps)


def scale_gradient(x, scale):
    """Identity in the forward pass with a scaled backward gradient."""
    return jax.lax.stop_gradient(x) + scale * (x - jax.lax.stop_gradient(x))


class CNNTorso(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1)).astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        return nn.relu(nn.Dense(512)(x.reshape((x.shape[0], -1))))


class MLPTorso(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(461, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=constant(0.0))(x))
        return nn.relu(nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2.0)), bias_init=constant(0.0))(x))


def _normalise_latent(x):
    lo = jnp.min(x, axis=-1, keepdims=True)
    hi = jnp.max(x, axis=-1, keepdims=True)
    return (x - lo) / jnp.maximum(hi - lo, 1e-5)


class Representation(nn.Module):
    embedding_dim: int
    pixel_based: bool
    scale_hidden_state: bool

    @nn.compact
    def __call__(self, obs):
        hidden = CNNTorso()(obs) if self.pixel_based else MLPTorso()(obs)
        latent = nn.relu(nn.Dense(self.embedding_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(hidden))
        return _normalise_latent(latent) if self.scale_hidden_state else latent


class Dynamics(nn.Module):
    action_dim: int
    embedding_dim: int
    support_size: int
    num_blocks: int
    scale_hidden_state: bool
    gradient_scale: float

    @nn.compact
    def __call__(self, latent, action):
        x = jnp.concatenate((latent, jax.nn.one_hot(action, self.action_dim)), axis=-1)
        for _ in range(self.num_blocks):
            x = nn.relu(nn.Dense(self.embedding_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x))
        reward_logits = nn.Dense(2 * self.support_size + 1, kernel_init=orthogonal(0.01))(x)
        next_latent = nn.relu(nn.Dense(self.embedding_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x))
        if self.scale_hidden_state:
            next_latent = _normalise_latent(next_latent)
        return reward_logits, scale_gradient(next_latent, self.gradient_scale)


class Prediction(nn.Module):
    action_dim: int
    embedding_dim: int
    support_size: int

    @nn.compact
    def __call__(self, latent):
        x = nn.relu(nn.Dense(self.embedding_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(latent))
        policy = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        value = nn.Dense(2 * self.support_size + 1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return policy, value


class MuZeroNetwork(nn.Module):
    action_dim: int
    support_size: int
    pixel_based: bool
    embedding_dim: int = 256
    num_dynamics_blocks: int = 2
    scale_hidden_state: bool = True
    dynamics_gradient_scale: float = 0.5

    def setup(self):
        self.representation_network = Representation(self.embedding_dim, self.pixel_based, self.scale_hidden_state)
        self.dynamics_network = Dynamics(self.action_dim, self.embedding_dim, self.support_size,
                                         self.num_dynamics_blocks, self.scale_hidden_state,
                                         self.dynamics_gradient_scale)
        self.prediction_network = Prediction(self.action_dim, self.embedding_dim, self.support_size)

    def __call__(self, obs):
        latent = self.representation_network(obs)
        policy, value = self.prediction_network(latent)
        # Initialise dynamics parameters at the same time as the other two functions.
        self.dynamics_network(latent, jnp.zeros((latent.shape[0],), dtype=jnp.int32))
        return latent, policy, value

    def recurrent(self, latent, action):
        reward, next_latent = self.dynamics_network(latent, action)
        policy, value = self.prediction_network(next_latent)
        return reward, next_latent, policy, value


@dataclass
class Trajectory:
    """One real-environment time step, stored as contiguous replay trajectories."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    policy: jnp.ndarray
    root_value: jnp.ndarray


def _n_step_target(rewards, dones, bootstrap, gamma):
    target = bootstrap
    for reward, done in zip(reversed(rewards), reversed(dones)):
        target = reward + gamma * (1.0 - done.astype(jnp.float32)) * target
    return target


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config.get("PIXEL_BASED", False)
    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{config['SEED']}"
    wandb.init(project=config.get("PROJECT", "jaxtari-blines"), entity=config.get("ENTITY") or None,
               config=config, name=run_name, save_code=True, mode=config.get("WANDB_MODE", "offline"))
    wandb.define_metric("*", step_metric="charts/global_step")
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    env = make_env(config["ENV_ID"], list(config.get("TRAIN_MODS", [])), pixel_based,
                   config.get("NATIVE_DOWNSCALING", True))()
    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape[:-1] if pixel_based else env.observation_space().shape
    num_envs, num_steps = config["NUM_ENVS"], config.get("NUM_STEPS", config.get("TRAIN_FREQUENCY", 4))
    support_size, transform_eps = config.get("SUPPORT_SIZE", 300), config.get("VALUE_TRANSFORM_EPS", 0.001)
    gamma, n_step, unroll_steps = config.get("GAMMA", 0.997), config.get("N_STEP", 10), config.get("UNROLL_STEPS", 5)
    sequence_length = unroll_steps + n_step + 1
    batch_size, num_simulations = config.get("BATCH_SIZE", 256), config.get("NUM_SIMULATIONS", 16)
    max_considered = int(min(config.get("MAX_NUM_CONSIDERED_ACTIONS") or num_simulations, action_dim))

    @jax.jit
    def vmap_reset(keys):
        obs, state = jax.vmap(env.reset)(keys)
        return obs.reshape((keys.shape[0],) + obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        obs, next_state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        return (obs.reshape((action.shape[0],) + obs_shape), next_state,
                reward.astype(jnp.float32), terminated | truncated, info)

    network_kwargs = dict(action_dim=action_dim, support_size=support_size, pixel_based=pixel_based,
                          embedding_dim=config.get("EMBEDDING_DIM", 256),
                          num_dynamics_blocks=config.get("NUM_DYNAMICS_BLOCKS", 2),
                          scale_hidden_state=config.get("SCALE_HIDDEN_STATE", True),
                          dynamics_gradient_scale=config.get("DYNAMICS_GRADIENT_SCALE", 0.5))
    network = MuZeroNetwork(**network_kwargs)
    key, init_key = jax.random.split(key)
    params = network.init(init_key, jnp.zeros((1,) + obs_shape, dtype=jnp.float32))
    # Flashbax fixes dtypes from this exemplar. Obtain it from the actual wrapper
    # rather than assuming pixel and object observations share a dtype.
    key, reset_key = jax.random.split(key)
    initial_obs, initial_env_state = vmap_reset(jax.random.split(reset_key, num_envs))

    total_timesteps, gradient_steps, scan_steps = config.get("TOTAL_TIMESTEPS", 10_000_000), config.get("GRADIENT_STEPS", 1), config.get("SCAN_STEPS", 100)
    steps_per_block, steps_per_iteration = num_envs * num_steps, num_envs * num_steps * scan_steps
    total_gradient_steps = max(1, total_timesteps // steps_per_block * gradient_steps)
    learning_rate = config.get("LEARNING_RATE", 3e-4)
    lr_schedule = lambda count: learning_rate * jnp.maximum(1.0 - count / total_gradient_steps, 0.0)
    tx = optax.chain(optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                     optax.adamw(lr_schedule if config.get("ANNEAL_LR", True) else learning_rate,
                                 weight_decay=config.get("WEIGHT_DECAY", 1e-4)))
    agent_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    def recurrent_fn(params, rng_key, action, latent):
        del rng_key
        reward, next_latent, policy, value = network.apply(params, latent, action, method=MuZeroNetwork.recurrent)
        return mctx.RecurrentFnOutput(reward=support_to_scalar(reward, support_size, transform_eps),
                                      discount=jnp.full(action.shape, gamma, dtype=jnp.float32),
                                      prior_logits=policy, value=support_to_scalar(value, support_size, transform_eps)), next_latent

    qtransform = partial(mctx.qtransform_completed_by_mix_value, value_scale=config.get("GUMBEL_C_SCALE", 0.1),
                         maxvisit_init=config.get("GUMBEL_C_VISIT", 50))

    def run_search(params, rng_key, obs):
        latent, policy, value = network.apply(params, obs)
        root = mctx.RootFnOutput(prior_logits=policy, value=support_to_scalar(value, support_size, transform_eps), embedding=latent)
        return mctx.gumbel_muzero_policy(params=params, rng_key=rng_key, root=root, recurrent_fn=recurrent_fn,
                                         num_simulations=num_simulations, qtransform=qtransform,
                                         max_num_considered_actions=max_considered)

    max_time = max(sequence_length, config.get("BUFFER_SIZE", 100_000) // num_envs)
    min_time = min(max_time, max(sequence_length, config.get("LEARNING_STARTS", 20_000) // num_envs))
    prioritised_replay = config.get("PRIORITIZED_REPLAY", False)
    replay_args = dict(add_batch_size=num_envs, sample_batch_size=batch_size,
                       sample_sequence_length=sequence_length, period=1,
                       min_length_time_axis=min_time, max_length_time_axis=max_time)
    if prioritised_replay:
        replay = fbx.make_prioritised_trajectory_buffer(
            **replay_args, priority_exponent=config.get("PER_ALPHA", 1.0)
        )
        replay = replay.replace(init=jax.jit(replay.init), add=jax.jit(replay.add, donate_argnums=0),
                                sample=jax.jit(replay.sample), can_sample=jax.jit(replay.can_sample),
                                set_priorities=jax.jit(replay.set_priorities, donate_argnums=0))
    else:
        replay = fbx.make_trajectory_buffer(**replay_args)
        replay = replay.replace(init=jax.jit(replay.init), add=jax.jit(replay.add, donate_argnums=0),
                                sample=jax.jit(replay.sample), can_sample=jax.jit(replay.can_sample))
    replay_obs_dtype = jnp.uint8 if pixel_based else initial_obs.dtype
    buffer_state = replay.init(Trajectory(jnp.zeros(obs_shape, replay_obs_dtype), jnp.array(0, jnp.int32),
                                          jnp.array(0.0, jnp.float32), jnp.array(False),
                                          jnp.zeros((action_dim,), jnp.float32), jnp.array(0.0, jnp.float32)))

    def collect_rollout(params, rng, env_state, obs):
        def step_fn(carry, _):
            env_state, obs, rng = carry
            rng, search_key = jax.random.split(rng)
            search = run_search(params, search_key, obs)
            next_obs, next_state, reward, done, info = vmap_step(env_state, search.action)
            item = Trajectory(obs, search.action, reward, done, search.action_weights, search.search_tree.node_values[:, 0])
            return (next_state, next_obs, rng), (item, info)
        (env_state, obs, rng), (trajectory, infos) = jax.lax.scan(step_fn, (env_state, obs, rng), None, length=num_steps)
        return env_state, obs, rng, trajectory, infos

    def masked_cross_entropy(logits, target, mask, importance_weights):
        loss = optax.softmax_cross_entropy(logits, target)
        weights = mask * importance_weights
        return jnp.sum(loss * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    def loss_fn(params, batch, importance_weights):
        """Unroll the learned dynamics K times and apply the three MuZero losses."""
        latent, policy_logits, value_logits = network.apply(params, batch.obs[:, 0])
        active = jnp.ones((batch.obs.shape[0],), jnp.float32)
        policy_loss = masked_cross_entropy(policy_logits, batch.policy[:, 0], active, importance_weights)
        value_target = _n_step_target([batch.reward[:, i] for i in range(n_step)],
                                      [batch.done[:, i] for i in range(n_step)], batch.root_value[:, n_step], gamma)
        root_prediction = support_to_scalar(value_logits, support_size, transform_eps)
        value_loss = masked_cross_entropy(value_logits, scalar_to_support(value_target, support_size, transform_eps), active, importance_weights)
        reward_loss = jnp.array(0.0, jnp.float32)
        for step in range(unroll_steps):
            reward_logits, latent, policy_logits, value_logits = network.apply(
                params, latent, batch.action[:, step], method=MuZeroNetwork.recurrent)
            reward_loss += masked_cross_entropy(reward_logits,
                                                scalar_to_support(batch.reward[:, step], support_size, transform_eps), active, importance_weights)
            active = active * (1.0 - batch.done[:, step].astype(jnp.float32))
            target_step = step + 1
            policy_loss += masked_cross_entropy(policy_logits, batch.policy[:, target_step], active, importance_weights)
            value_target = _n_step_target([batch.reward[:, target_step + i] for i in range(n_step)],
                                          [batch.done[:, target_step + i] for i in range(n_step)],
                                          batch.root_value[:, target_step + n_step], gamma)
            value_loss += masked_cross_entropy(value_logits,
                                               scalar_to_support(value_target, support_size, transform_eps), active, importance_weights)
        scale = float(unroll_steps + 1) if config.get("LOSS_SCALE_BY_UNROLL", True) else 1.0
        policy_loss, value_loss = policy_loss / scale, value_loss / scale
        reward_loss /= max(float(unroll_steps), 1.0) if config.get("LOSS_SCALE_BY_UNROLL", True) else 1.0
        total = (config.get("POLICY_LOSS_COEF", 1.0) * policy_loss + config.get("VALUE_LOSS_COEF", 1.0) * value_loss
                 + config.get("REWARD_LOSS_COEF", 1.0) * reward_loss)
        # The paper prioritises sequences by the root value error. It is detached
        # from gradients and applied by Flashbax after this optimizer update.
        priorities = jnp.abs(root_prediction - value_target) + 1e-6
        return total, (policy_loss, value_loss, reward_loss, priorities)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def full_step(agent_state, buffer_state, env_state, obs, rng, global_step):
        env_state, next_obs, rng, trajectory, infos = collect_rollout(agent_state.params, rng, env_state, obs)
        trajectory = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory)
        # Pixel replay stores unnormalised frames.  Keeping this uint8 avoids a
        # fourfold buffer expansion; CNNTorso converts it to float32 on sampling.
        if pixel_based:
            trajectory = trajectory.replace(obs=trajectory.obs.astype(jnp.uint8))
        buffer_state = replay.add(buffer_state, trajectory)
        def update(carry, _):
            state, buffer, rng = carry
            rng, sample_key = jax.random.split(rng)
            sample = replay.sample(buffer, sample_key)
            if prioritised_replay:
                importance = sample.probabilities ** (-config.get("PER_BETA", 1.0))
                importance = importance / jnp.maximum(jnp.max(importance), 1.0)
            else:
                importance = jnp.ones((batch_size,), dtype=jnp.float32)
            (loss, parts), grads = grad_fn(state.params, sample.experience, importance)
            policy_loss, value_loss, reward_loss, priorities = parts
            buffer = replay.set_priorities(buffer, sample.indices, priorities) if prioritised_replay else buffer
            return (state.apply_gradients(grads=grads), buffer, rng), (loss, policy_loss, value_loss, reward_loss)
        def learn(carry):
            carry, losses = jax.lax.scan(update, carry, None, length=gradient_steps)
            return carry, jax.tree.map(lambda x: x[-1], losses)
        (agent_state, buffer_state, rng), losses = jax.lax.cond(
            replay.can_sample(buffer_state), learn,
            lambda c: (c, tuple(jnp.array(0.0) for _ in range(4))),
            (agent_state, buffer_state, rng),
        )
        return (agent_state, buffer_state, env_state, next_obs, rng, global_step + steps_per_block), (infos, losses)

    carry = (agent_state, buffer_state, initial_env_state, initial_obs, key, jnp.array(0, jnp.int32))
    def scanned_steps(carry):
        return jax.lax.scan(lambda c, _: full_step(*c), carry, None, length=scan_steps)

    print("[muzero] compiling latent search, trajectory replay, and unrolled update ...")
    compilation_start = time.perf_counter()
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[muzero] compilation time: {time.perf_counter() - compilation_start:.2f}s")
    rtpt = RTPT(name_initials=config.get("NAME_INITIALS", "MU"), experiment_name=run_name,
                max_iterations=max(1, total_timesteps // steps_per_iteration))
    rtpt.start()

    def save_and_eval(step_count, current_params):
        root = config.get("SAVE_PATH", "./models")
        if root is None:
            return {}
        model_path = f"{root}/{run_name}/{config['EXP_NAME']}_{step_count}_{int(time.time())}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as model_file:
            model_file.write(flax.serialization.to_bytes([config, current_params]))
        metrics = {}
        for mods, label in [([], "default")] + [([m] if isinstance(m, str) else list(m), str(m)) for m in (config.get("EVAL_MODS", []) or config.get("TRAIN_MODS", []))]:
            returns, states = evaluate(model_path, partial(make_env, mods=mods, pixel_based=pixel_based,
                native_downscaling=config.get("NATIVE_DOWNSCALING", True), eval=True), config["ENV_ID"], 10,
                MuZeroNetwork, support_to_scalar, network_kwargs, support_size, transform_eps, num_simulations,
                max_considered, config.get("GUMBEL_C_VISIT", 50), config.get("GUMBEL_C_SCALE", 0.1), gamma, config["SEED"] + 42)
            metrics[label] = float(np.mean(jax.device_get(returns)))
            wandb.log({f"eval/episodic_return_{label}": metrics[label]}, step=step_count)
            if config.get("CAPTURE_VIDEO", False):
                frames = jnp.transpose(jax.vmap(jaxatari.make(config["ENV_ID"], mods=mods).renderer.render)(states), (0, 3, 1, 2))
                wandb.log({f"eval/video_{label}": wandb.Video(np.array(frames), fps=30, format="mp4")}, step=step_count)
        return metrics

    start_time, global_step = time.perf_counter(), 0
    while global_step < total_timesteps:
        iteration_start = time.perf_counter()
        carry, (infos, losses) = compiled(carry)
        global_step = int(carry[-1])
        loss, policy_loss, value_loss, reward_loss = (float(x[-1]) for x in losses)
        sps = int(global_step / max(time.perf_counter() - start_time, 1e-6))
        print(f"[muzero] step {global_step} | return {float(infos['returned_episode_returns'][-1].mean()):.2f} | loss {loss:.4f} | SPS {sps}")
        wandb.log({"charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
                   "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
                   "losses/loss": loss, "losses/policy_loss": policy_loss, "losses/value_loss": value_loss,
                   "losses/reward_loss": reward_loss, "charts/SPS": sps,
                   "charts/SPS_update": int(steps_per_iteration / max(time.perf_counter() - iteration_start, 1e-6)),
                   "charts/global_step": global_step}, step=global_step)
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config.get("EVAL_DURING_TRAIN", False) and iteration and iteration % config.get("EVAL_EVERY", 1) == 0:
            save_and_eval(global_step, carry[0].params)
    metrics = save_and_eval(global_step, carry[0].params)
    wandb.finish()
    return metrics

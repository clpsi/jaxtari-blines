# AlphaZero-style agent for JAXtari: Gumbel MuZero search running on the real
# environment instead of a learned model.
#
# The search calls env.step inside recurrent_fn, which is only possible because
# jaxatari environments are pure functions over pytree states - a state can be
# branched and batched inside the tree. There is no self-play and no learned
# dynamics, so this is "AlphaZero" in the sense of planning with a perfect
# model rather than a reproduction of Silver et al. (2017).
#
# Search:  Danihelka et al. (2022), Policy improvement by planning with Gumbel,
#          ICLR. https://openreview.net/forum?id=bERaNdoegnO  (via deepmind/mctx)
# Targets: Schrittwieser et al. (2019), Mastering Atari, Go, Chess and Shogi by
#          Planning with a Learned Model. https://arxiv.org/abs/1911.08265
# Structure, env setup and logging follow agents/c51/c51.py from this repo.
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
from flax.training.train_state import TrainState
from rtpt import RTPT

import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)

from agents.alphazero.alphazero_eval import evaluate


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env

    return thunk


# --------------------------------------------------------------------------- #
# Value transform
#
# Atari scores span several orders of magnitude between games, which a scalar
# regression head does not survive. MuZero instead squashes the target with an
# invertible transform and predicts a distribution over a fixed integer support
# (601 atoms for [-300, 300]); the scalar is recovered as the expectation under
# that distribution. Section "Network Architecture" of arXiv 1911.08265.
# --------------------------------------------------------------------------- #

def value_transform(x, eps=0.001):
    """h(x) = sign(x) * (sqrt(|x| + 1) - 1 + eps * x)."""
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x


def inverse_value_transform(x, eps=0.001):
    """Inverse of value_transform, derived by solving the quadratic."""
    numerator = jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps)) - 1.0
    return jnp.sign(x) * (jnp.square(numerator / (2.0 * eps)) - 1.0)


def scalar_to_support(x, support_size, eps=0.001):
    """Transform scalars into two-hot vectors over [-support_size, support_size].

    A value of 3.7 becomes weight 0.3 on the atom for 3 and 0.7 on the atom
    for 4, so the head can be trained with a plain cross-entropy loss.
    """
    x = value_transform(x, eps)
    x = jnp.clip(x, -support_size, support_size)

    lower = jnp.floor(x)
    upper_weight = x - lower
    lower_index = (lower + support_size).astype(jnp.int32)

    n_atoms = 2 * support_size + 1
    lower_onehot = jax.nn.one_hot(lower_index, n_atoms)
    upper_onehot = jax.nn.one_hot(jnp.clip(lower_index + 1, 0, n_atoms - 1), n_atoms)
    return lower_onehot * (1.0 - upper_weight)[..., None] + upper_onehot * upper_weight[..., None]


def support_to_scalar(logits, support_size, eps=0.001):
    """Expectation under the softmax over the support, then undo h(x)."""
    probs = jax.nn.softmax(logits, axis=-1)
    atoms = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    return inverse_value_transform(jnp.sum(probs * atoms, axis=-1), eps)


# --------------------------------------------------------------------------- #
# Networks
#
# Only a policy head and a value head are learned. Rewards and transitions come
# from the environment, so unlike MuZero there is no representation, dynamics
# or reward network here.
# --------------------------------------------------------------------------- #

class CNNTorso(nn.Module):
    """DQN-style encoder for stacked grayscale frames, shape (B, F, H, W)."""

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # frame stack becomes the channel axis
        x = x.astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(512)(x))


class MLPTorso(nn.Module):
    """Encoder for flattened object-centric observations.

    Layer sizes match the other agents in this repo so that a comparison is
    not confounded by network capacity.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(461, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        x = nn.relu(nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return x


class AlphaZeroNetwork(nn.Module):
    """Shared torso with a policy head and a categorical value head."""

    action_dim: int
    support_size: int
    pixel_based: bool

    @nn.compact
    def __call__(self, obs):
        hidden = CNNTorso()(obs) if self.pixel_based else MLPTorso()(obs)
        # Small policy init keeps the prior close to uniform at the start, so
        # the search explores instead of committing to noise.
        policy_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(hidden)
        value_logits = nn.Dense(
            2 * self.support_size + 1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(hidden)
        return policy_logits, value_logits


@flax.struct.dataclass
class Transition:
    """One training item. No sequences needed: without a learned model the
    targets for a state do not depend on the surrounding trajectory."""

    obs: jnp.ndarray
    target_policy: jnp.ndarray
    target_value: jnp.ndarray


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    pixel_based = config.get("PIXEL_BASED", False)
    if pixel_based and config.get("NUM_ENVS", 1) > 32:
        print("Warning: pixel observations with many envs can exhaust GPU memory.")

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{config['SEED']}"

    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
    )
    # Everything is plotted against env steps. Iterations mean different things
    # for different agents, so they are useless as a shared x-axis.
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    train_mods = list(config.get("TRAIN_MODS", []))
    env = make_env(
        config["ENV_ID"],
        train_mods,
        pixel_based,
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    if pixel_based:
        # (F, H, W, 1) -> (F, H, W): the frame stack is the channel axis
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]
    num_steps = config.get("NUM_STEPS", config.get("TRAIN_FREQUENCY", 4))
    support_size = config.get("SUPPORT_SIZE", 300)
    gamma = config.get("GAMMA", 0.997)
    n_step = config.get("N_STEP", 10)
    batch_size = config.get("BATCH_SIZE", 512)
    num_simulations = config.get("NUM_SIMULATIONS", 16)

    # The paper samples m = min(n, num_actions) root actions without
    # replacement on Atari. mctx already caps this at the number of valid
    # actions, but computing it here keeps the config game-independent.
    max_considered = config.get("MAX_NUM_CONSIDERED_ACTIONS") or min(num_simulations, action_dim)
    max_considered = int(min(max_considered, action_dim))

    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info

    network = AlphaZeroNetwork(
        action_dim=action_dim, support_size=support_size, pixel_based=pixel_based
    )
    key, net_key = jax.random.split(key)
    params = network.init(net_key, jnp.zeros((1, *obs_shape)))

    total_timesteps = config.get("TOTAL_TIMESTEPS", 10_000_000)
    gradient_steps = config.get("GRADIENT_STEPS", 1)
    # One collection block is num_envs * num_steps env steps and yields
    # gradient_steps optimiser updates; SCAN_STEPS blocks are fused into one
    # compiled iteration.
    steps_per_block = num_envs * num_steps
    steps_per_iteration = steps_per_block * config.get("SCAN_STEPS", 100)

    # The schedule counter advances once per optimiser step, so it has to be
    # normalised by the number of optimiser steps - not by the number of
    # compiled iterations. Getting this wrong anneals the learning rate to
    # zero within the first few iterations and silently freezes the network.
    total_gradient_steps = max(1, (total_timesteps // steps_per_block) * gradient_steps)

    def linear_schedule(count):
        frac = 1.0 - count / total_gradient_steps
        return config.get("LEARNING_RATE", 3e-4) * jnp.maximum(frac, 0.0)

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
        optax.adamw(
            learning_rate=linear_schedule if config.get("ANNEAL_LR", True) else config.get("LEARNING_RATE", 3e-4),
            weight_decay=config.get("WEIGHT_DECAY", 1e-4),
        ),
    )
    agent_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # ----------------------------------------------------------------------- #
    # Search
    # ----------------------------------------------------------------------- #

    def recurrent_fn(params, rng_key, action, embedding):
        """Transition inside the search tree.

        embedding is the batched environment state. Stepping the real env here
        is what separates this agent from MuZero: rewards and transitions are
        exact, and terminal states end the return properly instead of being
        bootstrapped from the value network.
        """
        del rng_key
        next_obs, next_state, reward, done, _ = vmap_step(embedding, action)

        policy_logits, value_logits = network.apply(params, next_obs)
        value = support_to_scalar(value_logits, support_size)

        # A terminal transition must not propagate value from whatever the
        # (auto-reset) environment returns next, so the discount is zeroed.
        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(done, 0.0, gamma).astype(jnp.float32),
            prior_logits=policy_logits,
            value=jnp.where(done, 0.0, value),
        )
        return recurrent_output, next_state

    # c_visit and c_scale from the Gumbel paper. Atari uses c_scale = 0.1
    # because reward scales differ hugely between games; a larger value makes
    # the search ignore the prior, which measurably hurt beam_rider.
    qtransform = partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=config.get("GUMBEL_C_SCALE", 0.1),
        maxvisit_init=config.get("GUMBEL_C_VISIT", 50),
    )

    def run_search(params, rng_key, obs, env_state):
        policy_logits, value_logits = network.apply(params, obs)
        root = mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=support_to_scalar(value_logits, support_size),
            embedding=env_state,
        )
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            qtransform=qtransform,
            max_num_considered_actions=max_considered,
        )

    # ----------------------------------------------------------------------- #
    # Replay
    # ----------------------------------------------------------------------- #

    replay_buffer = fbx.make_item_buffer(
        max_length=config.get("BUFFER_SIZE", 200_000),
        min_length=config.get("LEARNING_STARTS", 20_000),
        sample_batch_size=batch_size,
        add_batches=True,
    )
    replay_buffer = replay_buffer.replace(
        init=jax.jit(replay_buffer.init),
        add=jax.jit(replay_buffer.add, donate_argnums=0),
        sample=jax.jit(replay_buffer.sample),
        can_sample=jax.jit(replay_buffer.can_sample),
    )
    dummy_item = Transition(
        obs=jnp.zeros(obs_shape, dtype=jnp.float32),
        target_policy=jnp.zeros((action_dim,), dtype=jnp.float32),
        target_value=jnp.zeros((), dtype=jnp.float32),
    )
    buffer_state = replay_buffer.init(dummy_item)

    # ----------------------------------------------------------------------- #
    # Collection and learning
    # ----------------------------------------------------------------------- #

    def collect_rollout(params, rng, env_state, obs):
        """Act for num_steps with search, returning the data needed for targets."""

        def step_once(carry, _):
            env_state, obs, rng = carry
            rng, search_key = jax.random.split(rng)

            policy_output = run_search(params, search_key, obs, env_state)
            # The acted action is the one Sequential Halving proposes; the
            # training target is the improved policy over all actions, not the
            # raw visit counts.
            action = policy_output.action
            search_policy = policy_output.action_weights
            root_value = policy_output.search_tree.node_values[:, 0]

            next_obs, next_env_state, reward, done, info = vmap_step(env_state, action)
            step_data = (obs, search_policy, root_value, reward, done)
            return (next_env_state, next_obs, rng), (step_data, info)

        (env_state, obs, rng), ((obs_t, policy_t, value_t, reward_t, done_t), infos) = jax.lax.scan(
            step_once, (env_state, obs, rng), None, length=num_steps
        )
        return env_state, obs, rng, (obs_t, policy_t, value_t, reward_t, done_t), infos

    def compute_value_targets(values, rewards, dones, bootstrap_value):
        """n-step returns bootstrapped from the search value.

        z_t = r_{t+1} + gamma * r_{t+2} + ... + gamma^(n-1) * r_{t+n}
              + gamma^n * v_{t+n}

        Implemented as a reverse scan over the rollout. Steps closer than n to
        the end of the rollout bootstrap earlier, which is slightly biased but
        avoids carrying a queue across iterations; with num_steps >> n_step the
        affected fraction is small.
        """
        # values[t] is the search value of the state at t; shift by one so that
        # index t holds the bootstrap value for the state that follows t.
        next_values = jnp.concatenate([values[1:], bootstrap_value[None, :]], axis=0)
        continues = (1.0 - dones.astype(jnp.float32)) * gamma

        def backward(carry, inputs):
            # carry holds the discounted return accumulated so far and how many
            # rewards have gone into it, so the bootstrap can be capped at n.
            future_return, steps_used = carry
            reward, cont, next_value = inputs

            # Reset at episode boundaries: cont is 0 there.
            future_return = jnp.where(steps_used >= n_step, next_value, future_return)
            steps_used = jnp.where(steps_used >= n_step, 0.0, steps_used)

            target = reward + cont * future_return
            steps_used = jnp.where(cont > 0.0, steps_used + 1.0, 0.0)
            return (target, steps_used), target

        init = (bootstrap_value, jnp.zeros_like(bootstrap_value))
        _, targets = jax.lax.scan(
            backward, init, (rewards, continues, next_values), reverse=True
        )
        return targets

    def loss_fn(params, batch):
        policy_logits, value_logits = network.apply(params, batch.obs)

        # Cross-entropy against the improved search policy (Gumbel MuZero uses
        # the completed Q-values, which mctx already folded into action_weights).
        policy_loss = optax.softmax_cross_entropy(policy_logits, batch.target_policy).mean()

        value_target = scalar_to_support(batch.target_value, support_size)
        value_loss = optax.softmax_cross_entropy(value_logits, value_target).mean()

        total = (
            config.get("POLICY_LOSS_COEF", 1.0) * policy_loss
            + config.get("VALUE_LOSS_COEF", 1.0) * value_loss
        )
        return total, (policy_loss, value_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def full_step(agent_state, buffer_state, env_state, obs, rng, global_step):
        """One collection block followed by the gradient steps it pays for."""
        env_state, next_obs, rng, rollout, infos = collect_rollout(
            agent_state.params, rng, env_state, obs
        )
        obs_t, policy_t, value_t, reward_t, done_t = rollout

        # Bootstrap the tail of the rollout with the value of the state we
        # stopped in, evaluated by the network rather than by another search.
        _, bootstrap_logits = network.apply(agent_state.params, next_obs)
        bootstrap_value = support_to_scalar(bootstrap_logits, support_size)
        target_v = compute_value_targets(value_t, reward_t, done_t, bootstrap_value)

        # (T, B, ...) -> (T * B, ...): every state is an independent item.
        flatten = lambda x: x.reshape((-1,) + x.shape[2:])
        items = Transition(
            obs=flatten(obs_t),
            target_policy=flatten(policy_t),
            target_value=flatten(target_v),
        )
        buffer_state = replay_buffer.add(buffer_state, items)

        def do_update(carry, _):
            state, k = carry
            k, sample_key = jax.random.split(k)
            batch = replay_buffer.sample(buffer_state, sample_key).experience
            (loss, (policy_loss, value_loss)), grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)
            return (state, k), (loss, policy_loss, value_loss)

        def run_updates(carry):
            carry, losses = jax.lax.scan(
                do_update, carry, None, length=gradient_steps
            )
            return carry, jax.tree.map(lambda x: x[-1], losses)

        # Nothing to learn from until the buffer has passed LEARNING_STARTS.
        (agent_state, rng), losses = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            run_updates,
            lambda c: (c, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))),
            (agent_state, rng),
        )

        global_step = global_step + num_envs * num_steps
        return (agent_state, buffer_state, env_state, next_obs, rng, global_step), (infos, losses)

    def save_and_eval(step_count):
        model_path = None
        if config.get("SAVE_PATH", "./models") is not None:
            model_path = (
                f'{config.get("SAVE_PATH", "./models")}/{run_name}/'
                f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model'
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, carry[0].params]))
            print(f"model saved to {model_path}")

        if model_path is None:
            print("SAVE_PATH is None, skipping evaluation (the evaluator loads from disk).")
            return {}

        eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
        eval_configs = [([], "default")]
        for mod in list(eval_mods):
            mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
            mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
            eval_configs.append((mods_config, mod_label))

        metrics = {}
        for mods_cfg, mod_label in eval_configs:
            print(f"Evaluating on {mod_label} ...")
            episodic_returns, env_states = evaluate(
                model_path,
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=pixel_based,
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                Model=AlphaZeroNetwork,
                support_to_scalar=support_to_scalar,
                support_size=support_size,
                pixel_based=pixel_based,
                num_simulations=num_simulations,
                max_num_considered_actions=max_considered,
                gumbel_c_visit=config.get("GUMBEL_C_VISIT", 50),
                gumbel_c_scale=config.get("GUMBEL_C_SCALE", 0.1),
                gamma=gamma,
                seed=config["SEED"] + 42,  # a different seed than training
            )
            metrics[mod_label] = np.mean(jax.device_get(episodic_returns))
            wandb.log(
                {f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count
            )

            if config["CAPTURE_VIDEO"]:
                clean_renderer = jaxatari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jax.vmap(clean_renderer.render)(env_states)
                frames = jnp.transpose(frames, (0, 3, 1, 2))  # (N,H,W,C) -> (N,C,H,W)
                wandb.log(
                    {f"eval/video_{mod_label}": wandb.Video(np.array(frames), fps=30, format="mp4")},
                    step=step_count,
                )
        return metrics

    # ----------------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------------- #

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    global_step = jnp.array(0, dtype=jnp.int32)
    carry = (agent_state, buffer_state, env_state, obs, key, global_step)

    def scanned_steps(carry):
        def step_fn(c, _):
            return full_step(*c)

        return jax.lax.scan(step_fn, carry, None, length=config.get("SCAN_STEPS", 100))

    print("[alphazero] start compile...")
    start_compile = time.perf_counter()
    # Donate the carry so XLA overwrites the replay buffer in place instead of
    # allocating a second copy; lower/compile ahead of time so nothing is
    # donated before the loop starts.
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[alphazero] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(
        name_initials=config.get("NAME_INITIALS", "XX"),
        experiment_name=run_name,
        max_iterations=max(1, total_timesteps // steps_per_iteration),
    )
    rtpt.start()

    run_time = time.perf_counter()
    global_step = 0
    print(f"[alphazero] training for {total_timesteps} steps ...")
    while global_step < total_timesteps:
        rtpt.step()
        iteration = global_step // steps_per_iteration
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
            save_and_eval(global_step)

        iteration_start = time.perf_counter()
        carry, (infos, (loss, policy_loss, value_loss)) = compiled(carry)
        global_step = int(carry[-1])

        sps = int(global_step / (time.perf_counter() - run_time))
        print(
            f"[alphazero] it {iteration} | step {global_step} "
            f"| return {infos['returned_episode_returns'][-1].mean():.2f} "
            f"| loss {loss[-1]:.4f} | SPS {sps}"
        )
        wandb.log(
            {
                "charts/avg_episodic_return": infos["returned_episode_returns"][-1].mean(),
                "charts/avg_episodic_length": infos["returned_episode_lengths"][-1].mean(),
                "losses/loss": loss[-1].item(),
                "losses/policy_loss": policy_loss[-1].item(),
                "losses/value_loss": value_loss[-1].item(),
                "charts/SPS": sps,
                "charts/SPS_update": int(
                    steps_per_iteration / (time.perf_counter() - iteration_start)
                ),
                "charts/time": time.perf_counter() - run_time,
                "charts/global_step": global_step,
            },
            step=global_step,
        )

    eval_metrics = save_and_eval(global_step + 1)
    wandb.finish()
    return eval_metrics
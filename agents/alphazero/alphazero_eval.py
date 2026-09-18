# Evaluation for the AlphaZero-style agent. Follows agents/c51/c51_eval.py from
# this repo; the difference is that acting here runs a Gumbel MuZero search over
# the real environment rather than a single forward pass.
#
# The network class and the support decoder are passed in as arguments (as C51
# passes its Model) rather than imported: alphazero.py already imports this
# module, so importing back would be a circular import.
from functools import partial
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import mctx

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model,
    support_to_scalar,
    support_size: int = 300,
    num_simulations: int = 16,
    max_num_considered_actions: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 0.1,
    gamma: float = 0.997,
    pixel_based: bool = False,
    seed: int = 1,
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    action_dim = env.action_space().n
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def wrapped_reset(key):
        """Reset and add the batch axis the networks expect."""
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    network = Model(
        action_dim=action_dim, support_size=support_size, pixel_based=pixel_based
    )
    key, network_key = jax.random.split(key)
    dummy_obs = env.observation_space().sample(network_key).squeeze()[None, ...]
    params = network.init(network_key, dummy_obs)

    with open(model_path, "rb") as f:
        (args, params) = flax.serialization.from_bytes((None, params), f.read())

    qtransform = partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=gumbel_c_scale,
        maxvisit_init=gumbel_c_visit,
    )

    def recurrent_fn(params, rng_key, action, embedding):
        """Same perfect model as during training: step the real environment."""
        del rng_key
        next_obs, next_state, reward, done, _ = jax.vmap(wrapped_step)(embedding, action)
        # wrapped_step already adds a batch axis per episode, so collapse the
        # duplicated axis before the networks see it.
        next_obs = next_obs.reshape((-1,) + next_obs.shape[2:])

        policy_logits, value_logits = network.apply(params, next_obs)
        value = support_to_scalar(value_logits, support_size)

        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(done, 0.0, gamma).astype(jnp.float32),
            prior_logits=policy_logits,
            value=jnp.where(done, 0.0, value),
        )
        return output, next_state

    @jax.jit
    def get_action(params, obs, env_state, key):
        policy_logits, value_logits = network.apply(params, obs)
        root = mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=support_to_scalar(value_logits, support_size),
            embedding=env_state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            qtransform=qtransform,
            max_num_considered_actions=min(max_num_considered_actions, action_dim),
            # Gumbel noise is what makes the agent try different actions across
            # episodes during training. At evaluation we want the agent's best
            # guess, so the noise is switched off.
            gumbel_scale=0.0,
        )
        return policy_output.action

    def step_fn(carry, _):
        obs, env_state, keys = carry
        # obs is (E, 1, ...) from the per-episode reset; the search batches over
        # episodes, so drop the inner axis.
        search_obs = obs.reshape((-1,) + obs.shape[2:])
        key = keys[0]

        actions = get_action(params, search_obs, env_state, key)
        keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)

        obs, env_state, reward, done, info = jax.vmap(wrapped_step)(env_state, actions)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (obs, env_state, keys), (first_states, done, reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    obs, env_states = jax.vmap(wrapped_reset)(reset_keys)

    carry = (obs, env_states, reset_keys)
    all_first_states, all_dones, all_rewards = [], [], []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)

    @jax.jit
    def scanned_step(carry):
        return jax.lax.scan(step_fn, carry, None, length=1000)

    # Search makes each step far more expensive than in the model-free agents,
    # so the chunk length is kept modest and the loop stops as soon as every
    # episode has finished once.
    while not jnp.all(done_ever):
        carry, (first_states_chunk, dones_chunk, rewards_chunk, _) = scanned_step(carry)
        all_first_states.append(first_states_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)

    first_states_history = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states)
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)

    # Only count rewards up to and including the first termination of each
    # episode; the env auto-resets and keeps producing reward afterwards.
    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)

    print(
        f"Evaluated {eval_episodes} episodes, mean return: {episodic_returns.mean():.2f}, "
        f"std return: {episodic_returns.std():.2f}"
    )

    env_states_until_done = jax.tree.map(
        lambda x: x[: first_done[0] + 1],
        first_states_history.atari_state.atari_state.env_state,
    )
    return episodic_returns, env_states_until_done
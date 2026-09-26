"""Evaluation for the MuZero agent.

The evaluator receives the network class from ``muzero.py`` to avoid a circular
import.  Search uses its learned latent dynamics; the real environment is only
advanced once for the action selected at the root.
"""

from functools import partial

import flax
import jax
import jax.numpy as jnp
import mctx


def evaluate(model_path, make_env, env_id, eval_episodes, Model, support_to_scalar,
             network_kwargs, support_size, transform_eps, num_simulations,
             max_num_considered_actions, search_algorithm, gumbel_c_visit,
             gumbel_c_scale, puct_c_init, puct_c_base, gamma, seed):
    env = make_env(env_id)()
    action_dim = env.action_space().n
    pixel_based = network_kwargs["pixel_based"]
    key = jax.random.PRNGKey(seed)
    network = Model(**network_kwargs)

    def remove_channel(obs):
        return obs[..., 0] if pixel_based else obs

    @jax.jit
    def reset(keys):
        obs, states = jax.vmap(env.reset)(keys)
        return remove_channel(obs), states

    @jax.jit
    def step(states, actions):
        obs, states, rewards, terminated, truncated, info = jax.vmap(env.step)(states, actions)
        return remove_channel(obs), states, rewards, terminated | truncated, info

    observation_shape = env.observation_space().shape[:-1] if pixel_based else env.observation_space().shape
    key, init_key = jax.random.split(key)
    variables = network.init(init_key, jnp.zeros((1,) + observation_shape, dtype=jnp.float32))
    with open(model_path, "rb") as model_file:
        _, variables = flax.serialization.from_bytes((None, variables), model_file.read())

    gumbel_qtransform = partial(mctx.qtransform_completed_by_mix_value, value_scale=gumbel_c_scale,
                                maxvisit_init=gumbel_c_visit)

    def recurrent_fn(params, rng_key, action, latent):
        del rng_key
        reward, next_latent, policy, value = network.apply(params, latent, action, method=Model.recurrent)
        output = mctx.RecurrentFnOutput(
            reward=support_to_scalar(reward, support_size, transform_eps),
            discount=jnp.full(action.shape, gamma, dtype=jnp.float32),
            prior_logits=policy,
            value=support_to_scalar(value, support_size, transform_eps),
        )
        return output, next_latent

    @jax.jit
    def select_action(params, obs, rng):
        latent, policy, value = network.apply(params, obs)
        root = mctx.RootFnOutput(
            prior_logits=policy,
            value=support_to_scalar(value, support_size, transform_eps),
            embedding=latent,
        )
        if search_algorithm == "puct":
            # MuZero §3 / Appendix I: greedy visit-count action, no root noise.
            output = mctx.muzero_policy(
                params=params, rng_key=rng, root=root, recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                qtransform=mctx.qtransform_by_parent_and_siblings,
                dirichlet_fraction=0.0,
                pb_c_init=puct_c_init,
                pb_c_base=puct_c_base,
                temperature=1e-8,
            )
        else:
            output = mctx.gumbel_muzero_policy(
                params=params, rng_key=rng, root=root, recurrent_fn=recurrent_fn,
                num_simulations=num_simulations, qtransform=gumbel_qtransform,
                max_num_considered_actions=min(max_num_considered_actions, action_dim),
                gumbel_scale=0.0,
            )
        return output.action

    keys = jax.random.split(key, eval_episodes)
    obs, states = reset(keys)
    returns = jnp.zeros((eval_episodes,), jnp.float32)
    finished = jnp.zeros((eval_episodes,), jnp.bool_)
    state_history = []
    for _ in range(100_000):
        key, action_key = jax.random.split(key)
        actions = select_action(variables, obs, action_key)
        obs, states, rewards, done, _ = step(states, actions)
        returns = returns + jnp.where(finished, 0.0, rewards)
        state_history.append(jax.tree.map(lambda x: x[0], states))
        finished = finished | done
        if bool(jax.device_get(jnp.all(finished))):
            break

    states = jax.tree.map(lambda *items: jnp.stack(items), *state_history)
    print(f"Evaluated {eval_episodes} episodes, mean return: {float(returns.mean()):.2f}")
    return returns, states.atari_state.atari_state.env_state

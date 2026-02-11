import datetime
import os
import time
from dataclasses import dataclass
from typing import Any, cast

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pgx
import tyro
import wandb
from flax.training.train_state import TrainState
from pgx import State
from pgx.experimental import auto_reset


@dataclass
class Args:
    # seed
    seed: int = 0
    """random seed"""
    num_seeds: int = 20
    """the number of random seeds to use for training"""
    hyp_tune: bool = False
    """if toggled, will use hyperparameter tuning"""

    # wandb
    wandb_mode: str = "online"
    """whether to track training with Weights & Biases"""
    wandb_project_name: str = "rl-template"
    """the name of the Weights & Biases project"""
    wandb_entity: str = ""
    """the name of the Weights & Biases entity"""

    # save path
    save_model: bool = False
    """if toggled, will save the trained model"""
    save_path: str = "./models"
    """the path to save the trained model"""
    max_to_keep: int = 5
    """the maximum number of checkpoints to keep"""

    # rl setting
    alg_name: str = "online-ppo"
    """the name of the algorithm to use"""
    num_envs: int = 128
    """the number of parallel environments to run"""
    num_steps: int = 64
    """the number of steps to run in each environment per update"""
    total_timesteps: int = int(1e7)
    """the total number of timesteps to train for"""
    total_timesteps_decay: int = int(1e7)
    """the total number of timesteps to use for linear decay of lr"""
    lr: float = 0.0005
    """the learning rate for the optimizer"""
    lr_linear_decay: bool = True
    """whether to decay the learning rate linearly to zero"""
    gamma: float = 0.99
    """the discount factor for future rewards"""
    tau: float = 1.0
    """the soft update factor for the target network"""
    gae_lambda: float = 0.95
    """the lambda parameter for TD"""
    reward_scale: float = 1.0
    """the scale for rewards"""
    num_batches: int = 128
    """the number of batches to use per update"""
    num_epochs: int = 3
    """the number of epochs to use per update"""
    norm_type: str = "layer_norm"
    """the type of normalization to use in the network"""
    norm_input: bool = False
    """if toggled, will normalize the input observations"""
    max_grad_norm: float = 10.0
    """the maximum gradient norm for gradient clipping"""
    clip_eps: float = 0.2
    """the epsilon value for clipping in PPO"""
    vf_coef: float = 0.5
    """the coefficient for the value function loss"""
    ent_coef: float = 0.01
    """the coefficient for the entropy bonus"""

    # env params
    env_name: str = "minatar-breakout"
    """the name of the environment to train on"""
    sticky_action_prob: float = 0.1
    """the probability of repeating the previous action"""

    # evaluation
    eval_during_training: bool = True
    """if toggled, will evaluate the agent during training"""
    eval_interval: float = 0.02
    """the interval at which to evaluate the agent during training, in terms of total timesteps"""
    eval_num_envs: int = 100
    """the number of environments to use for evaluation"""

    # run name
    run_name: str = ""
    """the name of the run, used for logging"""


class ActorCritic(nn.Module):
    num_actions: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: chex.Array):
        x = x.astype(jnp.float32)
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh

        x = nn.Conv(32, kernel_size=(2, 2))(x)
        x = jax.nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(64)(x)
        x = jax.nn.relu(x)

        # Actor head
        actor_mean = nn.Dense(64)(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.num_actions)(actor_mean)

        # Critic head
        critic = nn.Dense(64)(x)
        critic = activation(critic)
        critic = nn.Dense(64)(critic)
        critic = activation(critic)
        critic = nn.Dense(1)(critic)

        return actor_mean, jnp.squeeze(critic, axis=-1)


@chex.dataclass(frozen=True)
class Transition:
    env_state: State
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array


class CustomTrainState(TrainState):
    timesteps: int = 0
    n_updates: int = 0


def make_train(config: Args):
    eval_interval = (
        int(config.eval_interval * config.total_timesteps)
        // config.num_envs
        // config.num_steps
    )
    num_updates = (
        config.total_timesteps // config.num_envs // config.num_steps // eval_interval
    )
    num_updates_decay = (
        config.total_timesteps_decay // config.num_envs // config.num_steps
    )

    env = pgx.make(config.env_name)
    env.sticky_action_prob = config.sticky_action_prob

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.init)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda n_envs: lambda env_state, action, rng: jax.vmap(
        auto_reset(env.step, env.init), in_axes=(0, 0, 0)
    )(env_state, action, jax.random.split(rng, n_envs))

    def train(rng):
        # init scheduler
        lr_scheduler = optax.linear_schedule(
            config.lr,
            1e-20,
            transition_steps=num_updates_decay * config.num_batches * config.num_epochs,
        )

        lr = lr_scheduler if config.lr_linear_decay else config.lr

        # init network
        network = ActorCritic(num_actions=env.num_actions)

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_shape))
            net_vars = network.init(rng, init_x)
            tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm), optax.radam(lr)
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=net_vars["params"],
                tx=tx,
            )

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(_rng)

        def _update_step(runner_state, unused):
            train_state, env_state, rng = cast(
                tuple[CustomTrainState, State, Any], runner_state
            )

            def _step_env(carry, unused):
                env_state, rng = cast(tuple[State, Any], carry)
                obs = env_state.observation

                # select action
                logits, value = cast(
                    tuple[chex.Array, chex.Array],
                    network.apply({"params": train_state.params}, obs),
                )
                pi = distrax.Categorical(logits=logits)
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # step env
                rng, _rng = jax.random.split(rng)
                next_env_state = vmap_step(config.num_envs)(env_state, action, _rng)

                transition = Transition(
                    env_state=env_state,
                    obs=obs,
                    action=action,
                    reward=config.reward_scale * next_env_state.rewards[:, 0],
                    done=next_env_state.terminated,
                    value=value,
                    log_prob=log_prob,
                )

                return (next_env_state, rng), (transition)

            rng, _rng = jax.random.split(rng)
            (env_state, rng), (transitions) = jax.lax.scan(
                _step_env,
                (env_state, _rng),
                None,
                config.num_steps,
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config.num_envs * config.num_steps,
            )

            # calc advantage
            _, last_value = network.apply(
                {"params": train_state.params}, env_state.observation
            )

            def _calculate_gae(traj_batch: Transition, last_val):
                def _get_advantages(carry, traj: Transition):
                    gae, next_value = cast(tuple[chex.Array, chex.Array], carry)
                    done, value, reward = (
                        traj.done,
                        traj.value,
                        traj.reward,
                    )
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(transitions, last_value)

            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state: TrainState, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(
                        params, traj_batch: Transition, gae: chex.Array, targets
                    ):
                        # return network
                        logits, value = cast(
                            tuple[chex.Array, chex.Array],
                            network.apply({"params": params}, traj_batch.obs),
                        )
                        pi = distrax.Categorical(logits=logits)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calc value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped)
                        ).mean()

                        # calc actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_losses = ratio * gae
                        actor_losses_clipped = (
                            jnp.clip(
                                ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
                            )
                            * gae
                        )
                        actor_loss = -jnp.minimum(
                            actor_losses, actor_losses_clipped
                        ).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            actor_loss
                            + config.vf_coef * value_loss
                            - config.ent_coef * entropy
                        )

                        return total_loss, (value_loss, actor_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, aux), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, aux)

                train_state, traj_batch, advantages, targets, rng = update_state

                batch_size = config.num_envs * config.num_steps

                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
                permutation = jax.random.permutation(_rng, batch_size)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_batches, -1, *x.shape[1:]]),
                    shuffled_batch,
                )
                train_state, (loss, aux) = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, (loss, aux)

            update_state = (train_state, transitions, advantages, targets, rng)
            update_state, (loss, aux) = jax.lax.scan(
                _update_epoch, update_state, None, config.num_epochs
            )

            train_state: CustomTrainState = update_state[0]
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "timesteps": train_state.timesteps,
                "n_updates": train_state.n_updates,
                "loss": loss.mean(),
                "value_loss": aux[0].mean(),
                "actor_loss": aux[1].mean(),
                "entropy": aux[2].mean(),
            }

            runner_state = (train_state, env_state, rng)

            return runner_state, metrics

        def eval_iteration(runner_state, unused):
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, eval_interval
            )
            R = eval_callback(runner_state, metrics)
            metrics["return"] = R
            metrics = {
                k: (v[-1] if k == "timesteps" else v.mean()) for k, v in metrics.items()
            }

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        env_state = vmap_reset(config.num_envs)(_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, _rng)

        runner_state, metrics = jax.lax.scan(
            eval_iteration, runner_state, None, num_updates
        )

        return {"runner_state": runner_state, "metrics": metrics}

    def evaluate(train_state: CustomTrainState, rng, num_episodes=config.eval_num_envs):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, num_episodes)
        env_state = jax.vmap(env.init)(rngs)
        R = jnp.zeros_like(env_state.rewards)

        def cond_fn(carry):
            env_state, _, _ = cast(tuple[State, chex.Array, Any], carry)
            return ~env_state.terminated.all()

        def body_fn(carry):
            env_state, R, rng = cast(tuple[State, chex.Array, Any], carry)
            logits, _ = cast(
                tuple[chex.Array, Any],
                train_state.apply_fn(
                    {"params": train_state.params}, env_state.observation
                ),
            )
            action = logits.argmax(axis=-1)
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, num_episodes)
            env_state = jax.vmap(env.step)(env_state, action, rngs)
            return (env_state, R + env_state.rewards, rng)

        env_state, R, rng = jax.lax.while_loop(cond_fn, body_fn, (env_state, R, rng))

        return R.mean()

    def eval_callback(runner_state, metrics):
        train_state, env_state, rng = cast(
            tuple[CustomTrainState, State, Any], runner_state
        )
        rng, _rng = jax.random.split(rng)
        R = evaluate(train_state, _rng)
        jax.debug.print(
            "step: {:>8d}, return: {:>7.3f}, loss: {:>8.7f}",
            metrics["timesteps"][-1],
            R.mean(),
            metrics["loss"].mean(),
        )

        return R

    return train


def log_eval_returns(metrics, config: Args):
    metric_name_map = {
        "timesteps": "eval/timesteps",
        "loss": "eval/loss",
        "value_loss": "eval/value_loss",
        "actor_loss": "eval/actor_loss",
        "entropy": "eval/entropy",
        "return": "eval/return",
    }
    metric_keys = tuple(metric_name_map.keys())

    for seed, ts in enumerate(metrics["timesteps"]):
        seed_metrics = {key: metrics[key][seed] for key in metric_keys}
        wandb.init(
            project=config.wandb_project_name,
            config=config,
            reinit=True,
            name=f"{config.run_name}-seed{seed}",
            mode=config.wandb_mode,
            entity=config.wandb_entity,
        )
        for step, t in enumerate(ts):
            log_payload = {
                metric_name_map[key]: seed_metrics[key][step] for key in metric_keys
            }
            wandb.log(
                log_payload,
                step=int(t.mean()),
            )
        wandb.finish()


def save_model(config: Args, train_state: CustomTrainState, returns: chex.Array):
    train_state = [
        jax.tree.map(lambda leaf: leaf[i], train_state) for i in range(config.num_seeds)
    ]

    k = min(10, returns.shape[1])
    returns = returns[:, -k:]
    mean_returns = returns.mean(axis=1)

    save_dir = os.path.join(config.save_path, f"{config.run_name}")
    save_dir = os.path.abspath(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep, best_fn=lambda metrics: metrics["return"]
    )

    with ocp.CheckpointManager(save_dir, options=options) as mngr:
        mngr: ocp.CheckpointManager = mngr
        for i, train_state in enumerate(train_state):
            save_dict = {
                "params": train_state.params,
            }
            mngr.save(
                i,
                args=ocp.args.StandardSave(save_dict),
                metrics={"return": float(mean_returns[i])},
            )

        mngr.wait_until_finished()

        print(f"Best model index: {mngr.best_step()}, Path: {save_dir}")


def single_run(config: Args):
    rng = jax.random.key(config.seed)
    rngs = jax.random.split(rng, config.num_seeds)

    t0 = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f} seconds")

    log_eval_returns(outs["metrics"], config)

    if config.save_model:
        save_model(config, outs["runner_state"][0], outs["metrics"]["return"])


def main():
    args = tyro.cli(Args)

    run_name = f"ppo-reset-pgx-{args.env_name}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M:%S')}"
    args.run_name = run_name

    if args.hyp_tune:
        raise NotImplementedError("Hyperparameter tuning not implemented yet.")
    else:
        single_run(args)


if __name__ == "__main__":
    main()

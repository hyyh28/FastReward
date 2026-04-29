import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from src.envs.frozenlake_reward_init import initial_reward_function
from src.envs.frozenlake_wrappers import make_frozenlake_vec_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on FrozenLake with custom shaped reward.")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/frozenlake_ppo")
    parser.add_argument("--map-name", type=str, default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--is-slippery", action="store_true", help="Enable slippery transition dynamics.")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use a faster profile to sanity-check pipelines quickly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fast_mode:
        args.total_timesteps = min(args.total_timesteps, 80_000)
        args.n_steps = min(args.n_steps, 128)
        args.n_epochs = min(args.n_epochs, 3)
        args.n_eval_episodes = min(args.n_eval_episodes, 10)

    log_dir = Path(args.log_dir)
    tb_dir = log_dir / "tb"
    ckpt_dir = log_dir / "checkpoints"
    best_dir = log_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_frozenlake_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        reward_fn=initial_reward_function,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )
    # Evaluate on true sparse reward to reflect task performance.
    eval_env = make_frozenlake_vec_env(
        n_envs=1,
        seed=args.seed + 10_000,
        reward_fn=None,
        map_name=args.map_name,
        is_slippery=args.is_slippery,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max((40_000 if args.fast_mode else 20_000) // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="ppo_frozenlake",
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_dir),
        eval_freq=max((40_000 if args.fast_mode else 10_000) // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=1, progress_bar=True)
        model.save(str(log_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

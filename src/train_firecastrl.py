import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.envs.scenario_factory import FirecastrlFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on firecastrl with CNN observations.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/firecastrl_maskable_cnn")
    parser.add_argument("--spray-radius", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    tb_dir = log_dir / "tb"
    ckpt_dir = log_dir / "checkpoints"
    best_dir = log_dir / "best_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    factory = FirecastrlFactory(gamma=0.997)
    train_env = factory.make_maskable_cnn_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        spray_radius=args.spray_radius,
        vec_env_type="subproc",
    )
    eval_env = factory.make_maskable_cnn_vec_env(
        n_envs=1,
        seed=args.seed + 10_000,
        spray_radius=args.spray_radius,
        vec_env_type="dummy",
    )

    model = MaskablePPO(
        policy=MaskableActorCriticCnnPolicy,
        env=train_env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.98,
        ent_coef=0.01,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="maskable_ppo_firecastrl",
    )
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_dir),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
        model.save(str(log_dir / "final_model"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

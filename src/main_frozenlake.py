from src.config import FrozenLakeConfig
from src.experiment.frozenlake_runner import FrozenLakeExperimentRunner
from src.reporting.run_logger import RunLogger
from src.strategies.ocba import OCBAllocationStrategy
from src.strategies.uniform import UniformAllocationStrategy


def run_project():
    candidates = ["baseline", "assist_distance", "assist_progress", "deceptive_hole_bias"]
    config = FrozenLakeConfig(candidates=candidates)

    uniform_runner = FrozenLakeExperimentRunner(
        config=config,
        allocation_strategy=UniformAllocationStrategy(),
    )
    results_uniform = uniform_runner.run(strategy_name="uniform")

    ocba_runner = FrozenLakeExperimentRunner(
        config=config,
        allocation_strategy=OCBAllocationStrategy(),
    )
    results_ocba = ocba_runner.run(strategy_name="ocba")

    logger = RunLogger(base_dir="logs")
    run_dir = logger.save_experiment(
        config_dict=config.__dict__,
        strategy_results={
            "uniform": results_uniform,
            "ocba": results_ocba,
        },
    )
    print(f"Run logs saved to: {run_dir}")


if __name__ == "__main__":
    run_project()

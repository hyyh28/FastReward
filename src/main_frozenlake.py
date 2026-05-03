from src.config import FrozenLakeConfig
from src.experiment.frozenlake_runner import FrozenLakeExperimentRunner
from src.reporting.run_logger import RunLogger
from src.strategies.ocba import OCBAllocationStrategy, RewardAdaptedOCBAAllocationStrategy
from src.strategies.uniform import UniformAllocationStrategy


def run_project():
    candidates = ["safe_distance", "risk_aware", "deceptive", "bad"]
    config = FrozenLakeConfig(candidates=candidates)


    ocba_runner = FrozenLakeExperimentRunner(
        config=config,
        allocation_strategy=OCBAllocationStrategy(),
    )
    results_ocba = ocba_runner.run(strategy_name="ocba")

    adapted_ocba_runner = FrozenLakeExperimentRunner(config=config, allocation_strategy=RewardAdaptedOCBAAllocationStrategy())
    results_adapted_ocba = adapted_ocba_runner.run(strategy_name="adapted_ocba")

    uniform_runner = FrozenLakeExperimentRunner(
        config=config,
        allocation_strategy=UniformAllocationStrategy(),
    )
    results_uniform = uniform_runner.run(strategy_name="uniform")

    logger = RunLogger(base_dir="logs")
    run_dir = logger.save_experiment(
        config_dict=config.__dict__,
        strategy_results={
            "uniform": results_uniform,
            "ocba": results_ocba,
            "adapted_ocba": results_adapted_ocba,
        },
    )
    print(f"Run logs saved to: {run_dir}")


if __name__ == "__main__":
    run_project()

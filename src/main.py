from src.config import MountainCarConfig
from src.experiment.runner import ExperimentRunner
from src.reporting.run_logger import RunLogger
from src.strategies.ocba import RewardAdaptedOCBAAllocationStrategy, OCBAllocationStrategy
from src.strategies.uniform import UniformAllocationStrategy


def run_project():
    candidates = ["baseline", "assist_pos_vel", "assist_energy_gate", "deceptive_left"]
    config = MountainCarConfig(candidates=candidates)

    uniform_runner = ExperimentRunner(config=config, allocation_strategy=UniformAllocationStrategy())
    results_uniform = uniform_runner.run(strategy_name="uniform")

    ocba_runner = ExperimentRunner(config=config, allocation_strategy=OCBAllocationStrategy())
    results_ocba = ocba_runner.run(strategy_name="ocba")

    adapted_ocba_runner = ExperimentRunner(config=config, allocation_strategy=RewardAdaptedOCBAAllocationStrategy())
    results_adapted_ocba = adapted_ocba_runner.run(strategy_name="adapted_ocba")

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

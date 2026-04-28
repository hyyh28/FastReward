import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, base_dir="logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: RunLogger._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [RunLogger._to_serializable(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    def save_experiment(self, config_dict, strategy_results):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "run_id": run_id,
            "config": self._to_serializable(config_dict),
            "results": self._to_serializable(strategy_results),
        }

        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        for strategy_name, data in strategy_results.items():
            rounds = data.get("round_records", [])
            jsonl_path = run_dir / f"{strategy_name}_rounds.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in rounds:
                    f.write(json.dumps(self._to_serializable(rec), ensure_ascii=False) + "\n")

        return str(run_dir)

"""
ユーティリティ関数
"""

import pandas as pd
import numpy as np
import yaml
import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """YAML設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Dict, path: str, indent: int = 2):
    """JSON形式でデータを保存"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """JSON形式でデータを読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_experiments_csv(row_data: Dict[str, Any], path: str = "experiments.csv"):
    """実験台帳にエントリを追加"""
    header = [
        "exp_id",
        "date",
        "git_sha",
        "wandb_url",
        "cv_metric",
        "cv_mean",
        "cv_std",
        "lb_public",
        "lb_private",
        "data_rev",
        "seed",
        "n_splits",
        "cv_method",
        "split_id",
        "notes",
        "submission_id",
        "submitted_at",
    ]

    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row_data)


def get_git_sha() -> str:
    """現在のGit SHAを取得"""
    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return sha[:8]  # 短縮版
    except:
        return "unknown"


def create_submission_manifest(
    exp_id: str,
    model_paths: List[str],
    config_path: str,
    threshold: float = 0.5,
    prediction_stats: Optional[Dict] = None,
    kaggle_info: Optional[Dict] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """提出用manifestを作成"""

    manifest = {
        "exp_id": exp_id,
        "generated_at": datetime.now().isoformat(),
        "models": model_paths,
        "threshold": float(threshold),
        "postprocess": None,
        "oof_path": "oof.parquet",
        "config_path": config_path,
        "git_sha": get_git_sha(),
        "wandb_run": None,  # 後で追加
        "prediction_stats": prediction_stats or {},
        "kaggle_submission": kaggle_info or {},
        "notes": notes,
    }

    return manifest


def validate_experiment_structure(exp_dir: str) -> Dict[str, bool]:
    """実験ディレクトリの構造をチェック"""
    exp_path = Path(exp_dir)

    required_files = {
        "config.yaml": exp_path / "config.yaml",
        "training.ipynb": exp_path / "training.ipynb",
        "evaluation.ipynb": exp_path / "evaluation.ipynb",
        "inference.ipynb": exp_path / "inference.ipynb",
        "model_dir": exp_path / "model",
        "submissions_dir": exp_path / "submissions",
    }

    validation_result = {}
    for name, path in required_files.items():
        validation_result[name] = path.exists()

    return validation_result


def calculate_prediction_stats(predictions: np.ndarray) -> Dict[str, float]:
    """予測値の統計情報を計算"""
    return {
        "mean": float(predictions.mean()),
        "std": float(predictions.std()),
        "min": float(predictions.min()),
        "max": float(predictions.max()),
        "q25": float(np.percentile(predictions, 25)),
        "q50": float(np.percentile(predictions, 50)),
        "q75": float(np.percentile(predictions, 75)),
    }


def create_feature_list_file(features: List[str], output_path: str):
    """特徴量リストをファイルに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        for feature in features:
            f.write(f"{feature}\n")


def load_feature_list_file(file_path: str) -> List[str]:
    """特徴量リストをファイルから読み込み"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str):
    """ディレクトリが存在しない場合は作成"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_experiment_id(base_name: str = "exp", existing_experiments: Optional[List[str]] = None) -> str:
    """新しい実験IDを生成"""
    if existing_experiments is None:
        # experiments/ディレクトリから既存実験を取得
        exp_dirs = list(Path("experiments").glob("exp*"))
        existing_experiments = [d.name for d in exp_dirs]

    # 数値部分を抽出してソート
    exp_numbers = []
    for exp_name in existing_experiments:
        if exp_name.startswith(base_name):
            try:
                num = int(exp_name[len(base_name) :])
                exp_numbers.append(num)
            except ValueError:
                continue

    # 次の番号を計算
    next_num = max(exp_numbers) + 1 if exp_numbers else 1
    return f"{base_name}{next_num:04d}"


class ExperimentLogger:
    """実験ログ管理クラス"""

    def __init__(self, exp_id: str, log_file: str = "experiment.log"):
        self.exp_id = exp_id
        self.log_file = log_file
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """ログメッセージを記録"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] [{self.exp_id}] {message}\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_config(self, config: Dict[str, Any]):
        """設定情報をログに記録"""
        self.log(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")

    def log_metrics(self, metrics: Dict[str, Any]):
        """メトリクス情報をログに記録"""
        self.log(f"Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    def log_completion(self):
        """実験完了をログに記録"""
        duration = (datetime.now() - self.start_time).total_seconds()
        self.log(f"Experiment completed in {duration:.1f} seconds")

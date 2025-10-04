from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}


def _sort_actions(actions: pd.DataFrame, match_col: str) -> pd.DataFrame:
    # 安全な安定ソート（mergesort）で時系列順を担保
    sort_cols = [match_col]
    if "period_id" in actions.columns:
        sort_cols.append("period_id")
    if "time_seconds" in actions.columns:
        sort_cols.append("time_seconds")
    actions = actions.sort_values(sort_cols, kind="mergesort").copy()
    return actions


def build_nstep_chain_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    n_steps: int = 3,
    gamma: float = 0.7,
) -> pd.DataFrame:
    """N手先のショートホライゾン連鎖（同一チーム内）と、xtデルタの割引和を集計。

    戻り値は match×player の集計 DataFrame（nstep_to_shot, nstep_xt_delta）。
    xtデルタ列が存在しない場合は nstep_xt_delta は0で埋めます。
    """
    sa = _sort_actions(actions, match_col)

    # 次のN手の type / team / player を用意
    for k in range(1, n_steps + 1):
        sa[f"next{k}_type"] = sa.groupby(match_col)[type_col].shift(-k)
        sa[f"next{k}_team"] = sa.groupby(match_col)[team_col].shift(-k)
        sa[f"next{k}_player"] = sa.groupby(match_col)[player_col].shift(-k)

    # 自身のアクションが、同一チーム内でN手以内にシュートに到達したか（割引和）
    weights = {k: (gamma ** (k - 1)) for k in range(1, n_steps + 1)}
    contrib = np.zeros(len(sa), dtype=float)
    for k in range(1, n_steps + 1):
        mask_k = (sa[f"next{k}_team"] == sa[team_col]) & (sa[f"next{k}_type"].isin(SHOT_TYPES))
        contrib += weights[k] * mask_k.astype(float)

    # xtデルタの割引和（存在する場合のみ）
    if "xt_learned_delta" in sa.columns:
        xt_contrib = np.zeros(len(sa), dtype=float)
        for k in range(1, n_steps + 1):
            future_same_team = (sa[f"next{k}_team"] == sa[team_col])
            # 次k手のxt_deltaを取るため、未来行の値を現在行に合わせてシフト
            future_xt = sa.groupby(match_col)["xt_learned_delta"].shift(-k)
            xt_contrib += weights[k] * np.where(future_same_team, future_xt.fillna(0.0), 0.0)
    else:
        xt_contrib = np.zeros(len(sa), dtype=float)

    out = (
        sa.assign(nstep_to_shot=contrib, nstep_xt_delta=xt_contrib)
        .groupby([match_col, player_col], as_index=False)[["nstep_to_shot", "nstep_xt_delta"]]
        .sum()
    )
    return out


def build_second_assist_sca_gca(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """セカンドアシスト、SCA(1/2手前)、GCA(1/2手前) の各集計を返す。

    Returns: (second_assist, sca1, sca2, gca1, gca2)
    """
    sa = _sort_actions(actions, match_col)
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_team"] = sa.groupby(match_col)[team_col].shift(-1)
    sa["next2_type"] = sa.groupby(match_col)[type_col].shift(-2)
    sa["next2_team"] = sa.groupby(match_col)[team_col].shift(-2)

    second_assist_mask = (
        (sa[type_col] == "pass")
        & (sa["next_type"] == "pass")
        & (sa["next2_type"].isin(SHOT_TYPES))
        & (sa["next_team"] == sa[team_col])
        & (sa["next2_team"] == sa[team_col])
    )
    second_assist = (
        sa.loc[second_assist_mask]
        .groupby([match_col, player_col], as_index=False)
        .size()
        .rename(columns={"size": "second_assist_count"})
    )

    # SCA / GCA
    sa["prev_type"] = sa.groupby(match_col)[type_col].shift(1)
    sa["prev_team"] = sa.groupby(match_col)[team_col].shift(1)
    sa["prev_player"] = sa.groupby(match_col)[player_col].shift(1)
    sa["prev2_type"] = sa.groupby(match_col)[type_col].shift(2)
    sa["prev2_team"] = sa.groupby(match_col)[team_col].shift(2)
    sa["prev2_player"] = sa.groupby(match_col)[player_col].shift(2)

    is_shot = sa[type_col].isin(SHOT_TYPES)
    same1 = sa["prev_team"].eq(sa[team_col])
    same2 = sa["prev2_team"].eq(sa[team_col])

    sca1 = (
        sa[is_shot & same1]
        .groupby([match_col, "prev_player"], as_index=False)
        .size()
        .rename(columns={"prev_player": player_col, "size": "SCA_1"})
    )
    sca2 = (
        sa[is_shot & same2]
        .groupby([match_col, "prev2_player"], as_index=False)
        .size()
        .rename(columns={"prev2_player": player_col, "size": "SCA_2"})
    )

    is_goal = is_shot & sa[result_col].eq("success")
    gca1 = (
        sa[is_goal & same1]
        .groupby([match_col, "prev_player"], as_index=False)
        .size()
        .rename(columns={"prev_player": player_col, "size": "GCA_1"})
    )
    gca2 = (
        sa[is_goal & same2]
        .groupby([match_col, "prev2_player"], as_index=False)
        .size()
        .rename(columns={"prev2_player": player_col, "size": "GCA_2"})
    )

    return second_assist, sca1, sca2, gca1, gca2


def build_pass_geometry_and_timing(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """パス幾何（距離/ゴール方向）と pass→shot のレイテンシ集計。

    Returns: (pass_geometry_agg, latency_agg)
    pass_geometry_agg: pass_dist_mean/max, to_goal_angle_abs_mean, to_goal_dist_mean
    latency_agg: pass_to_shot_latency_mean/min
    """
    sa = _sort_actions(actions, match_col)
    # パス幾何
    p = sa[sa[type_col] == "pass"].copy()
    if p.empty:
        pass_geom_agg = pd.DataFrame(columns=[match_col, player_col, "pass_dist_mean", "pass_dist_max", "to_goal_angle_abs_mean", "to_goal_dist_mean"])  # empty
    else:
        p["dx"] = (p["end_x"] - p["start_x"]).fillna(0.0)
        p["dy"] = (p["end_y"] - p["start_y"]).fillna(0.0)
        p["pass_dist"] = np.hypot(p["dx"], p["dy"]).astype(float)
        # ゴール座標を(105,34)とみなす（座標はホーム基準へ標準化済み前提）
        p["to_goal_dx"] = 105.0 - p["end_x"]
        p["to_goal_dy"] = 34.0 - p["end_y"]
        # 角度が小さいほどゴールへ直進
        p["to_goal_angle_abs"] = np.arctan2(np.abs(p["to_goal_dy"]), np.maximum(p["to_goal_dx"], 1e-6))
        p["to_goal_dist"] = np.hypot(p["to_goal_dx"], p["to_goal_dy"]).astype(float)
        pass_geom_agg = (
            p.groupby([match_col, player_col], as_index=False)
            .agg(
                pass_dist_mean=("pass_dist", "mean"),
                pass_dist_max=("pass_dist", "max"),
                to_goal_angle_abs_mean=("to_goal_angle_abs", "mean"),
                to_goal_dist_mean=("to_goal_dist", "mean"),
            )
        )

    # pass→shotのレイテンシ
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_team"] = sa.groupby(match_col)["team_id"].shift(-1) if "team_id" in sa.columns else pd.NA
    sa["next_time"] = sa.groupby(match_col)["time_seconds"].shift(-1)
    lat_df = sa[(sa[type_col] == "pass") & (sa["next_type"].isin(SHOT_TYPES))].copy()
    if "team_id" in sa.columns:
        lat_df = lat_df[lat_df["next_team"] == lat_df["team_id"]]
    if lat_df.empty:
        latency_agg = pd.DataFrame(columns=[match_col, player_col, "pass_to_shot_latency_mean", "pass_to_shot_latency_min"])  # empty
    else:
        lat_df["pass_to_shot_latency"] = (lat_df["next_time"] - lat_df["time_seconds"]).clip(lower=0.0).astype(float)
        latency_agg = (
            lat_df.groupby([match_col, player_col], as_index=False)["pass_to_shot_latency"]
            .agg(pass_to_shot_latency_mean=("mean"), pass_to_shot_latency_min=("min"))
        )

    return pass_geom_agg, latency_agg


def build_xpass_risk_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
) -> pd.DataFrame:
    """xPassベースのリスク/創造性指標を集計して返す。

    - risk_creativity_sum = Σ( (1 - xpass_prob) * xt_learned_delta_on_success )
    - xpass_mean, xpass_min
    - pass_success_minus_xpass = 実成功率 - xpass_mean
    - xpass_deep_mean (end_x >= 70), xpass_box_mean (end_x >= 88 & 14<=y<=54)
    """
    if ("xpass_prob" not in actions.columns) or (type_col not in actions.columns):
        return pd.DataFrame(columns=[match_col, player_col])

    p = actions[actions[type_col] == "pass"].copy()
    if p.empty:
        return pd.DataFrame(columns=[match_col, player_col, "risk_creativity_sum", "xpass_mean", "xpass_min", "pass_success_minus_xpass", "xpass_deep_mean", "xpass_box_mean"])  # empty

    p["xpass_prob"] = p["xpass_prob"].astype(float)
    if "xt_learned_delta_on_success" in p.columns:
        delta_on_success = p["xt_learned_delta_on_success"].fillna(0.0).astype(float)
    elif "xt_learned_delta" in p.columns:
        # 成功時のみ価値前進とみなす
        success_flag = p[result_col].eq("success").astype(int)
        delta_on_success = np.where(success_flag == 1, p["xt_learned_delta"].fillna(0.0), 0.0)
    else:
        delta_on_success = 0.0

    p_success = p[result_col].eq("success").astype(float)
    p["risk_creativity_term"] = (1.0 - p["xpass_prob"]) * delta_on_success

    # 深い位置の判定
    deep_mask = p["end_x"].ge(70.0)
    box_mask = p["end_x"].ge(88.0) & p["end_y"].between(14.0, 54.0)

    agg = (
        p.groupby([match_col, player_col])
        .agg(
            risk_creativity_sum=("risk_creativity_term", "sum"),
            xpass_mean=("xpass_prob", "mean"),
            xpass_min=("xpass_prob", "min"),
            empirical_success=(p_success.name, "mean"),
            xpass_deep_mean=(pd.Series(p.loc[deep_mask, "xpass_prob"]).groupby([p.loc[deep_mask, match_col], p.loc[deep_mask, player_col]]).transform("mean") if deep_mask.any() else ("xpass_prob", "mean")),
        )
        .reset_index()
    )

    # xpass_deep_mean の計算を安定化（上の複雑な式のフォールバック）
    if "xpass_deep_mean" not in agg.columns or agg["xpass_deep_mean"].isna().all():
        deep = p.loc[deep_mask, [match_col, player_col, "xpass_prob"]].copy()
        if not deep.empty:
            deep_agg = deep.groupby([match_col, player_col], as_index=False)["xpass_prob"].mean().rename(columns={"xpass_prob": "xpass_deep_mean"})
            agg = agg.merge(deep_agg, on=[match_col, player_col], how="left")
        else:
            agg["xpass_deep_mean"] = np.nan

    box = p.loc[box_mask, [match_col, player_col, "xpass_prob"]].copy()
    if not box.empty:
        box_agg = box.groupby([match_col, player_col], as_index=False)["xpass_prob"].mean().rename(columns={"xpass_prob": "xpass_box_mean"})
        agg = agg.merge(box_agg, on=[match_col, player_col], how="left")
    else:
        agg["xpass_box_mean"] = np.nan

    agg["pass_success_minus_xpass"] = agg["empirical_success"].fillna(0.0) - agg["xpass_mean"].fillna(0.0)
    agg = agg.drop(columns=["empirical_success"], errors="ignore")

    for c in ["risk_creativity_sum", "xpass_mean", "xpass_min", "pass_success_minus_xpass", "xpass_deep_mean", "xpass_box_mean"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    return agg


def add_player_trend(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "Date",
    group_key: str = "player_id",
    target_col: str = "xAG",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """選手の時系列フォームをリーク対策付きで付与（expanding/rolling3/diff）。

    trainとtestを結合し、日付→match_id順でソートして groupby rolling を計算し、最後に再分割します。
    test行の特徴量は過去（train側）情報のみを参照します。
    """
    both = train_df.copy()
    both["__split__"] = "train"
    test_tag = test_df.copy()
    test_tag["__split__"] = "test"
    both = pd.concat([both, test_tag], axis=0, ignore_index=True, sort=False)

    # 型安全
    both[date_col] = pd.to_datetime(both[date_col], errors="coerce")

    both = both.sort_values([date_col, "match_id"], kind="mergesort").copy()

    def _safe_series(x: pd.Series) -> pd.Series:
        if x.dtype.kind in {"f", "i"}:
            return x
        return pd.to_numeric(x, errors="coerce")

    # expanding/rolling（未来遮断のため shift(1)）
    val = _safe_series(both[target_col])
    grp = both.groupby(group_key)
    both[f"{target_col}_expanding_mean"] = grp[target_col].apply(lambda s: _safe_series(s).expanding().mean()).reset_index(level=0, drop=True).shift(1)
    both[f"{target_col}_rolling3_mean"] = grp[target_col].apply(lambda s: _safe_series(s).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True).shift(1)
    both[f"{target_col}_diff_prev"] = grp[target_col].diff().shift(0)

    train_new = both[both["__split__"] == "train"].drop(columns=["__split__"])  # type: ignore
    test_new = both[both["__split__"] == "test"].drop(columns=["__split__"])  # type: ignore
    return train_new, test_new


def merge_blocks(
    base: pd.DataFrame,
    blocks: Iterable[pd.DataFrame],
    match_col: str = "match_id",
    player_col: str = "player_id",
    how: str = "left",
) -> pd.DataFrame:
    df = base.copy()
    for b in blocks:
        if b is None or b.empty:
            continue
        df = df.merge(b, on=[match_col, player_col], how=how)
    return df



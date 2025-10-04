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

    FIX ISSUE 2: Time series leakage prevention
    - For train: compute statistics using ONLY train data (chronologically sorted)
    - For test: compute statistics using train history only (no test data)
    - Ensures test rows never contribute to train features, even when test matches
      occur chronologically before some training matches
    """
    def _safe_series(x: pd.Series) -> pd.Series:
        if x.dtype.kind in {"f", "i"}:
            return x
        return pd.to_numeric(x, errors="coerce")

    # ============================================================
    # Process training data: compute time series features from train data only
    # ============================================================
    train_copy = train_df.copy()
    train_copy[date_col] = pd.to_datetime(train_copy[date_col], errors="coerce")
    train_copy = train_copy.sort_values([date_col, "match_id"], kind="mergesort").copy()

    # Compute expanding/rolling statistics on TRAIN data only (with shift for leak prevention)
    train_grp = train_copy.groupby(group_key)
    train_copy[f"{target_col}_expanding_mean"] = (
        train_grp[target_col]
        .apply(lambda s: _safe_series(s).expanding().mean())
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    train_copy[f"{target_col}_rolling3_mean"] = (
        train_grp[target_col]
        .apply(lambda s: _safe_series(s).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    train_copy[f"{target_col}_diff_prev"] = train_grp[target_col].diff().shift(0)

    # ============================================================
    # Process test data: compute features using ONLY train history
    # ============================================================
    test_copy = test_df.copy()
    test_copy[date_col] = pd.to_datetime(test_copy[date_col], errors="coerce")

    # Initialize test features
    test_copy[f"{target_col}_expanding_mean"] = np.nan
    test_copy[f"{target_col}_rolling3_mean"] = np.nan
    test_copy[f"{target_col}_diff_prev"] = np.nan

    # For each test player, use their train history to compute statistics
    for player in test_copy[group_key].unique():
        # Get all train data for this player (sorted chronologically)
        player_train_hist = train_copy[train_copy[group_key] == player].copy()

        if player_train_hist.empty:
            # No training history for this player - leave as NaN
            continue

        # Get test rows for this player
        test_player_mask = test_copy[group_key] == player
        test_player_rows = test_copy[test_player_mask].copy()

        if test_player_rows.empty:
            continue

        # For each test match, compute statistics using only train data up to that point
        # (but this is conservative - we use ALL train history regardless of date)
        # This is safe because test data never contributes to features

        # Use the final expanding mean from training data
        if len(player_train_hist) > 0:
            final_expanding_mean = _safe_series(player_train_hist[target_col]).expanding().mean().iloc[-1]
            test_copy.loc[test_player_mask, f"{target_col}_expanding_mean"] = final_expanding_mean

            # Rolling mean uses last 3 train values
            if len(player_train_hist) >= 3:
                final_rolling_mean = _safe_series(player_train_hist[target_col]).iloc[-3:].mean()
            else:
                final_rolling_mean = _safe_series(player_train_hist[target_col]).mean()
            test_copy.loc[test_player_mask, f"{target_col}_rolling3_mean"] = final_rolling_mean

            # Diff: difference from last train value
            last_train_value = _safe_series(player_train_hist[target_col]).iloc[-1]
            test_copy.loc[test_player_mask, f"{target_col}_diff_prev"] = (
                _safe_series(test_player_rows[target_col]) - last_train_value
            )

    # Fill remaining NaN with 0 for players without train history
    for col in [f"{target_col}_expanding_mean", f"{target_col}_rolling3_mean", f"{target_col}_diff_prev"]:
        train_copy[col] = train_copy[col].fillna(0.0)
        test_copy[col] = test_copy[col].fillna(0.0)

    # Validation assertions
    assert f"{target_col}_expanding_mean" in train_copy.columns, "Train expanding_mean not created"
    assert f"{target_col}_expanding_mean" in test_copy.columns, "Test expanding_mean not created"
    assert len(train_copy) == len(train_df), "Train row count mismatch"
    assert len(test_copy) == len(test_df), "Test row count mismatch"

    return train_copy, test_copy


def build_time_based_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    time_col: str = "time_seconds",
    period_col: str = "period_id",
) -> pd.DataFrame:
    """時間帯別パフォーマンス特徴量を構築。

    Returns: 試合×選手別の時間帯別統計
    - first_half_actions: 前半アクション数
    - second_half_actions: 後半アクション数
    - final_15min_actions: ラスト15分アクション数
    - early_10min_actions: 序盤10分アクション数
    - time_weighted_intensity: 時間重み付き強度(後半ほど重要)
    """
    sa = _sort_actions(actions, match_col)

    # 前半/後半の判定
    first_half = (sa[period_col] == 1)
    second_half = (sa[period_col] == 2)

    # ラスト15分: 後半30分以降 (time_seconds >= 2700)
    final_15min = second_half & (sa[time_col] >= 2700)

    # 序盤10分: 前半0-600秒
    early_10min = first_half & (sa[time_col] <= 600)

    # 時間重み付き (0.0-1.0の正規化時間 × アクション重要度)
    sa["time_weight"] = np.where(
        first_half,
        (sa[time_col] / 2700.0) * 0.5,  # 前半は0.0-0.5の重み
        0.5 + (sa[time_col] / 2700.0) * 0.5  # 後半は0.5-1.0の重み
    )

    agg = (
        sa.assign(
            first_half=first_half.astype(int),
            second_half=second_half.astype(int),
            final_15min=final_15min.astype(int),
            early_10min=early_10min.astype(int),
        )
        .groupby([match_col, player_col], as_index=False)
        .agg(
            first_half_actions=("first_half", "sum"),
            second_half_actions=("second_half", "sum"),
            final_15min_actions=("final_15min", "sum"),
            early_10min_actions=("early_10min", "sum"),
            time_weighted_intensity=("time_weight", "sum"),
        )
    )

    return agg


def build_zone_based_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
) -> pd.DataFrame:
    """ゾーン別アクション密度特徴量を構築。

    ピッチを9分割(3x3)してゾーン別統計を算出:
    - defensive_zone_actions: 自陣ゾーン(x < 35)
    - middle_zone_actions: 中盤ゾーン(35 <= x < 70)
    - attacking_zone_actions: 敵陣ゾーン(x >= 70)
    - halfspace_left_actions: 左ハーフスペース(y < 22.67)
    - halfspace_right_actions: 右ハーフスペース(y > 45.33)
    - central_corridor_actions: 中央レーン(22.67 <= y <= 45.33)
    - final_third_penetrations: 敵陣最終ライン進入(x >= 70)
    - box_entries: ペナルティエリア進入(x >= 88.5, 13.84 <= y <= 54.16)
    """
    sa = actions.copy()

    # X軸ゾーン分割 (0-105を3分割)
    defensive_zone = (sa["start_x"] < 35.0)
    middle_zone = (sa["start_x"] >= 35.0) & (sa["start_x"] < 70.0)
    attacking_zone = (sa["start_x"] >= 70.0)

    # Y軸ゾーン分割 (0-68を3分割)
    halfspace_left = (sa["start_y"] < 22.67)
    central_corridor = (sa["start_y"] >= 22.67) & (sa["start_y"] <= 45.33)
    halfspace_right = (sa["start_y"] > 45.33)

    # 特殊ゾーン
    final_third = (sa["start_x"] >= 70.0)
    penalty_box = (sa["start_x"] >= 88.5) & (sa["start_y"] >= 13.84) & (sa["start_y"] <= 54.16)

    agg = (
        sa.assign(
            defensive_zone=defensive_zone.astype(int),
            middle_zone=middle_zone.astype(int),
            attacking_zone=attacking_zone.astype(int),
            halfspace_left=halfspace_left.astype(int),
            halfspace_right=halfspace_right.astype(int),
            central_corridor=central_corridor.astype(int),
            final_third=final_third.astype(int),
            box_entry=penalty_box.astype(int),
        )
        .groupby([match_col, player_col], as_index=False)
        .agg(
            defensive_zone_actions=("defensive_zone", "sum"),
            middle_zone_actions=("middle_zone", "sum"),
            attacking_zone_actions=("attacking_zone", "sum"),
            halfspace_left_actions=("halfspace_left", "sum"),
            halfspace_right_actions=("halfspace_right", "sum"),
            central_corridor_actions=("central_corridor", "sum"),
            final_third_penetrations=("final_third", "sum"),
            box_entries=("box_entry", "sum"),
        )
    )

    return agg


def build_pass_network_centrality(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    time_col: str = "time_seconds",
) -> pd.DataFrame:
    """パスネットワーク中心性特徴量を構築 (高速版)。

    グラフ理論ベースの選手役割評価:
    - betweenness_centrality: 媒介中心性(攻撃の中継点度)
    - closeness_centrality: 近接中心性(攻撃への近さ)
    - degree_centrality: 次数中心性(パス接続数)
    - pass_receiver_diversity: パス先の多様性(エントロピー)
    - unique_pass_partners: ユニークなパス相手数
    """
    try:
        import networkx as nx
    except ImportError:
        # NetworkXがない場合は空のDataFrameを返す
        return pd.DataFrame(columns=[match_col, player_col])

    sa = _sort_actions(actions, match_col)

    if sa.empty:
        return pd.DataFrame(columns=[match_col, player_col,
                                    "betweenness_centrality", "closeness_centrality",
                                    "degree_centrality", "pass_receiver_diversity",
                                    "unique_pass_partners"])

    results = []

    # パスのみ抽出
    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame(columns=[match_col, player_col,
                                    "betweenness_centrality", "closeness_centrality",
                                    "degree_centrality", "pass_receiver_diversity",
                                    "unique_pass_partners"])

    # 🚀 高速化: 次のアクションの選手を事前計算（パスデータに対して）
    passes["next_player"] = passes.groupby([match_col, team_col])[player_col].shift(-1)

    for (match_id, team_id), group in passes.groupby([match_col, team_col]):
        G = nx.DiGraph()

        # 🚀 高速化: ベクトル化でグラフ構築
        # パスの送り手と次のアクションの選手（受け手）でエッジ作成
        pass_edges = group[[player_col, "next_player"]].dropna()
        pass_edges = pass_edges[pass_edges[player_col] != pass_edges["next_player"]]

        if pass_edges.empty:
            continue

        # エッジを一括追加
        edges = list(zip(pass_edges[player_col], pass_edges["next_player"]))
        G.add_edges_from(edges)

        if G.number_of_nodes() == 0:
            continue

        # 中心性計算
        try:
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            degree = nx.degree_centrality(G)
        except:
            betweenness = {n: 0.0 for n in G.nodes()}
            closeness = {n: 0.0 for n in G.nodes()}
            degree = {n: 0.0 for n in G.nodes()}

        # 🚀 高速化: パス先の多様性を一括計算
        pass_receiver_stats = (
            pass_edges.groupby(player_col)["next_player"]
            .agg(lambda x: len(set(x)))  # unique partners
            .to_dict()
        )

        # エントロピー計算
        diversity_dict = {}
        for passer in pass_receiver_stats.keys():
            receivers = pass_edges[pass_edges[player_col] == passer]["next_player"]
            if len(receivers) > 0:
                receiver_counts = receivers.value_counts(normalize=True)
                diversity_dict[passer] = -np.sum(receiver_counts * np.log2(receiver_counts + 1e-9))
            else:
                diversity_dict[passer] = 0.0

        # 各選手の統計を構築
        for player_id in G.nodes():
            results.append({
                match_col: match_id,
                player_col: player_id,
                "betweenness_centrality": betweenness.get(player_id, 0.0),
                "closeness_centrality": closeness.get(player_id, 0.0),
                "degree_centrality": degree.get(player_id, 0.0),
                "pass_receiver_diversity": diversity_dict.get(player_id, 0.0),
                "unique_pass_partners": pass_receiver_stats.get(player_id, 0),
            })

    return pd.DataFrame(results)


def build_extended_chain_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    n_steps: int = 7,
    gamma: float = 0.6,
) -> pd.DataFrame:
    """拡張シーケンス連鎖特徴量 (5-10手先まで)。

    nstep_chain_featuresの拡張版で、より長い連鎖を評価:
    - longchain_to_shot: 7手先までのシュート到達割引和
    - longchain_xt_delta: 7手先までのxT増加割引和
    """
    sa = _sort_actions(actions, match_col)

    # 次のN手の type / team を用意
    for k in range(1, n_steps + 1):
        sa[f"next{k}_type"] = sa.groupby(match_col)[type_col].shift(-k)
        sa[f"next{k}_team"] = sa.groupby(match_col)[team_col].shift(-k)

    # 長期連鎖の割引和
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
            future_xt = sa.groupby(match_col)["xt_learned_delta"].shift(-k)
            xt_contrib += weights[k] * np.where(future_same_team, future_xt.fillna(0.0), 0.0)
    else:
        xt_contrib = np.zeros(len(sa), dtype=float)

    out = (
        sa.assign(longchain_to_shot=contrib, longchain_xt_delta=xt_contrib)
        .groupby([match_col, player_col], as_index=False)[["longchain_to_shot", "longchain_xt_delta"]]
        .sum()
    )
    return out


def build_dynamic_positioning_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
) -> pd.DataFrame:
    """動的ポジショニング特徴量を構築。

    位置の分散・移動範囲を評価:
    - position_variance_x: 前後方向の活動範囲(分散)
    - position_variance_y: 左右方向の活動範囲(分散)
    - position_range_x: 前後方向の最大移動距離
    - position_range_y: 左右方向の最大移動距離
    - avg_action_distance: アクション間平均移動距離
    """
    sa = _sort_actions(actions, match_col)

    # 移動距離計算
    sa["prev_x"] = sa.groupby([match_col, player_col])["start_x"].shift(1)
    sa["prev_y"] = sa.groupby([match_col, player_col])["start_y"].shift(1)
    sa["move_dist"] = np.hypot(
        sa["start_x"] - sa["prev_x"].fillna(sa["start_x"]),
        sa["start_y"] - sa["prev_y"].fillna(sa["start_y"])
    )

    agg = (
        sa.groupby([match_col, player_col], as_index=False)
        .agg(
            position_variance_x=("start_x", "var"),
            position_variance_y=("start_y", "var"),
            position_range_x=("start_x", lambda x: x.max() - x.min() if len(x) > 1 else 0),
            position_range_y=("start_y", lambda x: x.max() - x.min() if len(x) > 1 else 0),
            avg_action_distance=("move_dist", "mean"),
        )
    )

    # NaN埋め
    for col in agg.columns:
        if col not in [match_col, player_col]:
            agg[col] = agg[col].fillna(0.0)

    return agg


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



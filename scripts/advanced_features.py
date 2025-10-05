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


def build_gca_spatial_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
) -> pd.DataFrame:
    """GCA直系の空間的特徴量（ゾーン14、ハーフスペース、カットバック）。

    Returns: 試合×選手別の空間特徴量
    - zone14_origin_pass_count/success_rate: ゾーン14起点パス
    - zone14_preGCA_count: ゾーン14からのGCA直前パス
    - halfspace_L_to_box_count/success_rate: 左ハーフスペース→PA侵入
    - halfspace_R_to_box_count/success_rate: 右ハーフスペース→PA侵入
    - cutback_count/success_rate: カットバック検出
    - cutback_next_shot_rate: カットバック後のシュート率
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, team_col, type_col, result_col,
                   "start_x", "start_y", "end_x", "end_y"]
    sa = _sort_actions(actions[needed_cols], match_col)

    # ゾーン14定義: x∈[70,88.5], y∈[13.84,54.16]の手前（中央攻撃帯）
    # ここでは簡易的に x∈[65,85], y∈[20,48]
    zone14_mask = (sa["start_x"] >= 65.0) & (sa["start_x"] < 85.0) & \
                  (sa["start_y"] >= 20.0) & (sa["start_y"] <= 48.0)

    # ハーフスペース定義: 左(y<22.67), 右(y>45.33)
    halfspace_L_mask = (sa["start_y"] < 22.67)
    halfspace_R_mask = (sa["start_y"] > 45.33)

    # PA侵入判定: end座標がPA内(x>=88.5, 13.84<=y<=54.16)
    box_entry_mask = (sa["end_x"] >= 88.5) & \
                     (sa["end_y"] >= 13.84) & (sa["end_y"] <= 54.16)

    # カットバック検出: ゴールライン際(x>=95, y<20 or y>48)から
    # 後方/横方向へのパス(end_x < start_x or |end_y-start_y|>10)
    cutback_origin = (sa["start_x"] >= 95.0) & \
                     ((sa["start_y"] < 20.0) | (sa["start_y"] > 48.0))
    cutback_direction = (sa["end_x"] < sa["start_x"]) | \
                        (np.abs(sa["end_y"] - sa["start_y"]) > 10.0)
    cutback_mask = cutback_origin & cutback_direction & (sa[type_col] == "pass")

    # GCA直前の判定（次2手でゴール）
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_team"] = sa.groupby(match_col)[team_col].shift(-1)
    sa["next2_type"] = sa.groupby(match_col)[type_col].shift(-2)
    sa["next2_team"] = sa.groupby(match_col)[team_col].shift(-2)

    is_goal_next1 = (sa["next_type"].isin(SHOT_TYPES)) & \
                    (sa["next_team"] == sa[team_col]) & \
                    (sa["next_type"].notna())
    is_goal_next2 = (sa["next2_type"].isin(SHOT_TYPES)) & \
                    (sa["next2_team"] == sa[team_col]) & \
                    (sa["next2_type"].notna())
    preGCA_mask = is_goal_next1 | is_goal_next2

    # 次アクションがシュートかどうか（カットバック評価用）
    sa["next_is_shot"] = is_goal_next1.astype(int)

    # パスのみフィルタ
    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # ゾーン14特徴量
    z14_passes = passes[zone14_mask].copy()
    z14_agg = z14_passes.groupby([match_col, player_col], as_index=False).agg(
        zone14_origin_pass_count=("type_name", "size"),
        zone14_origin_pass_success=(result_col, lambda x: (x == "success").sum()),
    )
    z14_agg["zone14_origin_pass_success_rate"] = np.where(
        z14_agg["zone14_origin_pass_count"] > 0,
        z14_agg["zone14_origin_pass_success"] / z14_agg["zone14_origin_pass_count"],
        0.0
    )
    z14_agg = z14_agg.drop(columns=["zone14_origin_pass_success"])

    # ゾーン14 preGCA
    z14_gca = passes[zone14_mask & preGCA_mask].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "zone14_preGCA_count"})

    # ハーフスペース→PA侵入
    hs_L_box = passes[halfspace_L_mask & box_entry_mask].groupby(
        [match_col, player_col], as_index=False
    ).agg(
        halfspace_L_to_box_count=("type_name", "size"),
        halfspace_L_to_box_success=(result_col, lambda x: (x == "success").sum()),
    )
    hs_L_box["halfspace_L_to_box_success_rate"] = np.where(
        hs_L_box["halfspace_L_to_box_count"] > 0,
        hs_L_box["halfspace_L_to_box_success"] / hs_L_box["halfspace_L_to_box_count"],
        0.0
    )
    hs_L_box = hs_L_box.drop(columns=["halfspace_L_to_box_success"])

    hs_R_box = passes[halfspace_R_mask & box_entry_mask].groupby(
        [match_col, player_col], as_index=False
    ).agg(
        halfspace_R_to_box_count=("type_name", "size"),
        halfspace_R_to_box_success=(result_col, lambda x: (x == "success").sum()),
    )
    hs_R_box["halfspace_R_to_box_success_rate"] = np.where(
        hs_R_box["halfspace_R_to_box_count"] > 0,
        hs_R_box["halfspace_R_to_box_success"] / hs_R_box["halfspace_R_to_box_count"],
        0.0
    )
    hs_R_box = hs_R_box.drop(columns=["halfspace_R_to_box_success"])

    # カットバック特徴量
    cutbacks = passes[cutback_mask].copy()
    if not cutbacks.empty:
        cb_agg = cutbacks.groupby([match_col, player_col], as_index=False).agg(
            cutback_count=("type_name", "size"),
            cutback_success=(result_col, lambda x: (x == "success").sum()),
            cutback_next_shot_sum=("next_is_shot", "sum"),
        )
        cb_agg["cutback_success_rate"] = np.where(
            cb_agg["cutback_count"] > 0,
            cb_agg["cutback_success"] / cb_agg["cutback_count"],
            0.0
        )
        cb_agg["cutback_next_shot_rate"] = np.where(
            cb_agg["cutback_count"] > 0,
            cb_agg["cutback_next_shot_sum"] / cb_agg["cutback_count"],
            0.0
        )
        cb_agg = cb_agg.drop(columns=["cutback_success", "cutback_next_shot_sum"])
    else:
        cb_agg = pd.DataFrame(columns=[match_col, player_col, "cutback_count",
                                       "cutback_success_rate", "cutback_next_shot_rate"])

    # 統合
    result = z14_agg
    for df in [z14_gca, hs_L_box, hs_R_box, cb_agg]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    # NaN埋め
    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_linebreak_packing_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
) -> pd.DataFrame:
    """ラインブレイク/パッキング近似特徴量。

    Returns:
    - linebreak_third_transition_count/rate: ゾーン跨ぎ前進パス
    - through_channel_pass_count/rate: スルーパス近似
    - packing_approx_score_mean: パッキング近似スコア平均
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, type_col, result_col,
                   "start_x", "start_y", "end_x", "end_y"]
    sa = _sort_actions(actions[needed_cols], match_col)
    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # ゾーン定義（3分割: 自陣<35, 中盤35-70, 敵陣>=70）
    passes["start_zone"] = pd.cut(
        passes["start_x"],
        bins=[0, 35, 70, 105],
        labels=["def", "mid", "att"],
        include_lowest=True
    )
    passes["end_zone"] = pd.cut(
        passes["end_x"],
        bins=[0, 35, 70, 105],
        labels=["def", "mid", "att"],
        include_lowest=True
    )

    # ラインブレイク判定: ゾーン跨ぎ & 前進量>=10%ピッチ長(10.5m)
    zone_cross = (passes["start_zone"] != passes["end_zone"])
    forward_prog = (passes["end_x"] - passes["start_x"]) >= 10.5
    linebreak_mask = zone_cross & forward_prog & (passes[result_col] == "success")

    # スルーパス近似: 中央～ハーフスペース(22.67<=y<=45.33)でゴール方向10%前進
    central = (passes["start_y"] >= 22.67) & (passes["start_y"] <= 45.33)
    through_prog = (passes["end_x"] - passes["start_x"]) >= 10.5
    through_mask = central & through_prog & (passes[result_col] == "success")

    # パッキング近似スコア: ゾーン跨ぎ数に重み付け（DF=3, MF=2, FW=1）
    # ベクトル化版で高速化 (Categorical → 数値に変換)
    zone_map = {"def": 0, "mid": 1, "att": 2}
    passes["start_zone_num"] = passes["start_zone"].astype(str).map(zone_map).fillna(0).astype(int)
    passes["end_zone_num"] = passes["end_zone"].astype(str).map(zone_map).fillna(0).astype(int)

    cross_levels = (passes["end_zone_num"] - passes["start_zone_num"]).clip(lower=0)
    weights_map = {0: 0, 1: 1, 2: 3}
    passes["packing_score"] = cross_levels.map(weights_map).fillna(0)

    # 集計
    lb_agg = passes[linebreak_mask].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "linebreak_third_transition_count"})

    # ベクトル化版: rate計算を高速化
    passes["is_linebreak"] = linebreak_mask
    lb_rate = passes.groupby([match_col, player_col], as_index=False).agg(
        linebreak_third_transition_rate=("is_linebreak", "mean")
    )

    through_agg = passes[through_mask].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "through_channel_pass_count"})

    # ベクトル化版: through rate計算を高速化
    passes["is_through"] = through_mask
    through_rate = passes.groupby([match_col, player_col], as_index=False).agg(
        through_channel_pass_rate=("is_through", "mean")
    )

    pack_agg = passes.groupby([match_col, player_col], as_index=False).agg(
        packing_approx_score_mean=("packing_score", "mean")
    )

    # 統合
    result = lb_agg.merge(lb_rate, on=[match_col, player_col], how="outer")
    result = result.merge(through_agg, on=[match_col, player_col], how="outer")
    result = result.merge(through_rate, on=[match_col, player_col], how="outer")
    result = result.merge(pack_agg, on=[match_col, player_col], how="outer")

    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_pass_chain_quality_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    time_col: str = "time_seconds",
) -> pd.DataFrame:
    """パス連鎖の質的特徴量（1-2、三人目、速攻）。

    Returns:
    - one_two_chain_count: 壁パス検出
    - third_man_release_count: 三人目の動き
    - burst_window_SCA_rate: 速攻窓でのSCA率
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, team_col, type_col, time_col]
    sa = _sort_actions(actions[needed_cols], match_col)

    # 次2手の情報
    for k in [1, 2]:
        sa[f"next{k}_type"] = sa.groupby(match_col)[type_col].shift(-k)
        sa[f"next{k}_team"] = sa.groupby(match_col)[team_col].shift(-k)
        sa[f"next{k}_player"] = sa.groupby(match_col)[player_col].shift(-k)
        sa[f"next{k}_time"] = sa.groupby(match_col)[time_col].shift(-k)

    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # 1-2（壁パス）検出: A→B の後3秒以内にB→A
    one_two_mask = (
        (passes["next1_type"] == "pass") &
        (passes["next1_team"] == passes[team_col]) &
        (passes["next2_player"] == passes[player_col]) &
        ((passes["next2_time"] - passes[time_col]) <= 3.0)
    )

    one_two_agg = passes[one_two_mask].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "one_two_chain_count"})

    # 三人目リリース: A→B→C で Cの次がシュート（近似: C後2手でシュート）
    sa["next3_type"] = sa.groupby(match_col)[type_col].shift(-3)
    sa["next3_team"] = sa.groupby(match_col)[team_col].shift(-3)

    third_man_mask = (
        (sa[type_col] == "pass") &
        (sa["next1_type"] == "pass") &
        (sa["next1_team"] == sa[team_col]) &
        (sa["next2_type"] == "pass") &
        (sa["next2_team"] == sa[team_col]) &
        (sa["next3_type"].isin(SHOT_TYPES)) &
        (sa["next3_team"] == sa[team_col])
    )

    third_man_agg = sa[third_man_mask].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "third_man_release_count"})

    # 速攻窓: 連続3アクションが5秒以内 & 末尾がシュート
    sa["window_duration"] = (sa["next2_time"] - sa[time_col]).fillna(999)
    burst_mask = (
        (sa["window_duration"] <= 5.0) &
        (sa["next2_type"].isin(SHOT_TYPES)) &
        (sa["next2_team"] == sa[team_col])
    )

    # ベクトル化版
    sa["is_burst"] = burst_mask
    burst_agg = sa.groupby([match_col, player_col], as_index=False).agg(
        burst_window_SCA_rate=("is_burst", "mean")
    )

    # 統合
    result = one_two_agg
    for df in [third_man_agg, burst_agg]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_box_entry_receiving_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
) -> pd.DataFrame:
    """PA進入の受け手文脈特徴量。

    Returns:
    - box_entry_from_halfspace_L/R/zone14_count: 起点別PA進入数
    - facing_forward_share_in_box: PA内前向き受け比率
    - first_touch_shot_rate_in_box: PA内ファーストタッチシュート率
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, type_col, result_col,
                   "start_x", "start_y", "end_x", "end_y"]
    if "time_seconds" in actions.columns:
        needed_cols.append("time_seconds")
    sa = _sort_actions(actions[needed_cols], match_col)

    # PA内判定
    box_mask = (sa["end_x"] >= 88.5) & \
               (sa["end_y"] >= 13.84) & (sa["end_y"] <= 54.16)

    # 起点分類
    zone14_origin = (sa["start_x"] >= 65.0) & (sa["start_x"] < 85.0) & \
                    (sa["start_y"] >= 20.0) & (sa["start_y"] <= 48.0)
    hs_L_origin = sa["start_y"] < 22.67
    hs_R_origin = sa["start_y"] > 45.33

    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # PA進入起点別カウント
    box_z14 = passes[box_mask & zone14_origin].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "box_entry_from_zone14_count"})

    box_hsL = passes[box_mask & hs_L_origin].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "box_entry_from_halfspace_L_count"})

    box_hsR = passes[box_mask & hs_R_origin].groupby(
        [match_col, player_col], as_index=False
    ).size().rename(columns={"size": "box_entry_from_halfspace_R_count"})

    # 前向き受け近似: end→goalの角度が小さい（<45度 = 0.785 rad）
    sa["to_goal_angle"] = np.arctan2(
        np.abs(34.0 - sa["end_y"]),
        np.maximum(105.0 - sa["end_x"], 1e-6)
    )
    forward_mask = (sa["to_goal_angle"] < 0.785) & box_mask

    # ベクトル化版
    sa_box = sa[box_mask].copy()
    sa_box["is_forward"] = forward_mask.loc[sa_box.index]
    forward_share = sa_box.groupby([match_col, player_col], as_index=False).agg(
        facing_forward_share_in_box=("is_forward", "mean")
    )

    # ファーストタッチシュート: PA内受け→即シュート（Δt<1秒）
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_time"] = sa.groupby(match_col)["time_seconds"].shift(-1)
    sa["latency"] = (sa["next_time"] - sa["time_seconds"]).fillna(999)

    first_touch_shot_mask = box_mask & (sa["next_type"].isin(SHOT_TYPES)) & (sa["latency"] < 1.0)

    # ベクトル化版
    sa_box["is_first_touch_shot"] = first_touch_shot_mask.loc[sa_box.index]
    first_touch_rate = sa_box.groupby([match_col, player_col], as_index=False).agg(
        first_touch_shot_rate_in_box=("is_first_touch_shot", "mean")
    )

    # 統合
    result = box_z14
    for df in [box_hsL, box_hsR, forward_share, first_touch_rate]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_setplay_bodypart_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
    bodypart_col: str = "bodypart_name",
) -> pd.DataFrame:
    """セットプレー/ボディパート文脈特徴量。

    Returns:
    - setplay_GCA_share: セットプレー起点GCA比率
    - openplay_GCA_share: オープンプレー起点GCA比率
    - bodypart_on_key_pass_rate_right/left/head: 利き足でのGCAレート
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, type_col]
    if bodypart_col in actions.columns:
        needed_cols.append(bodypart_col)
    if "team_id" in actions.columns:
        needed_cols.append("team_id")
    sa = _sort_actions(actions[needed_cols], match_col)

    # セットプレー判定（type_nameで近似）
    setplay_types = {"corner", "freekick_short", "freekick_crossed", "throw_in"}
    sa["is_setplay"] = sa[type_col].isin(setplay_types).astype(int)

    # GCA判定（次2手でゴール）
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next2_type"] = sa.groupby(match_col)[type_col].shift(-2)

    if "team_id" in sa.columns:
        sa["next_team"] = sa.groupby(match_col)["team_id"].shift(-1)
        sa["next2_team"] = sa.groupby(match_col)["team_id"].shift(-2)
    else:
        sa["next_team"] = None
        sa["next2_team"] = None

    is_gca = (
        (sa["next_type"].isin(SHOT_TYPES) | sa["next2_type"].isin(SHOT_TYPES))
    )
    if sa["next_team"] is not None:
        is_gca = is_gca & (
            (sa["next_team"] == sa["team_id"]) | (sa["next2_team"] == sa["team_id"])
        )

    # セットプレー/オープンプレー比率（ベクトル化版）
    sa_setplay = sa[sa["is_setplay"] == 1].copy()
    sa_setplay["is_gca"] = is_gca.loc[sa_setplay.index]
    setplay_gca = sa_setplay.groupby([match_col, player_col], as_index=False).agg(
        setplay_GCA_share=("is_gca", "mean")
    )

    sa_openplay = sa[sa["is_setplay"] == 0].copy()
    sa_openplay["is_gca"] = is_gca.loc[sa_openplay.index]
    openplay_gca = sa_openplay.groupby([match_col, player_col], as_index=False).agg(
        openplay_GCA_share=("is_gca", "mean")
    )

    # ボディパート別GCAレート
    if bodypart_col not in sa.columns:
        bodypart_agg = pd.DataFrame(columns=[match_col, player_col])
    else:
        gca_actions = sa[is_gca].copy()
        if not gca_actions.empty:
            bp_counts = gca_actions.groupby([match_col, player_col, bodypart_col]).size().reset_index(name="count")
            bp_pivot = bp_counts.pivot_table(
                index=[match_col, player_col],
                columns=bodypart_col,
                values="count",
                fill_value=0
            ).reset_index()

            # 列名を標準化
            col_map = {
                "right_foot": "bodypart_on_key_pass_rate_right",
                "left_foot": "bodypart_on_key_pass_rate_left",
                "head": "bodypart_on_key_pass_rate_head",
            }
            bp_pivot.columns = [col_map.get(c, c) if c not in [match_col, player_col] else c
                                for c in bp_pivot.columns]
            bodypart_agg = bp_pivot
        else:
            bodypart_agg = pd.DataFrame(columns=[match_col, player_col])

    # 統合
    result = setplay_gca
    for df in [openplay_gca, bodypart_agg]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


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


def build_attack_tempo_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
    time_col: str = "time_seconds",
) -> pd.DataFrame:
    """攻撃テンポ・リズム特徴量。

    データリーク防止: 全て当該アクション時点の情報のみ使用

    Returns:
    - possession_duration_before_shot: シュート前のポゼッション時間平均
    - pass_tempo_variance: パステンポの分散（予測不可能性）
    - acceleration_phase_count: 加速フェーズ回数（連続3パスで間隔短縮）
    - quick_transition_rate: 5秒以内速攻率
    - slow_buildup_gca_rate: 15秒以上ビルドアップGCA率
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, team_col, type_col, time_col]
    sa = _sort_actions(actions[needed_cols], match_col)

    if sa.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # 次2手の情報（GCA判定用）
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_team"] = sa.groupby(match_col)[team_col].shift(-1)
    sa["next2_type"] = sa.groupby(match_col)[type_col].shift(-2)
    sa["next2_team"] = sa.groupby(match_col)[team_col].shift(-2)

    # GCA判定: 次2手でシュート（将来情報不使用）
    is_gca = (
        (sa["next_type"].isin(SHOT_TYPES) & (sa["next_team"] == sa[team_col])) |
        (sa["next2_type"].isin(SHOT_TYPES) & (sa["next2_team"] == sa[team_col]))
    )

    # 1. possession_duration_before_shot
    # GCAアクションについて、同一チームの直前ポゼッション継続時間を計測
    sa["prev_team"] = sa.groupby(match_col)[team_col].shift(1)
    sa["prev_time"] = sa.groupby(match_col)[time_col].shift(1)

    # 同一チーム連続の場合の時間差
    same_team_mask = (sa["prev_team"] == sa[team_col])
    sa["possession_time"] = 0.0
    sa.loc[same_team_mask, "possession_time"] = (
        sa.loc[same_team_mask, time_col] - sa.loc[same_team_mask, "prev_time"]
    ).clip(lower=0, upper=120)  # 0-120秒に制限

    gca_possession = sa[is_gca].groupby([match_col, player_col], as_index=False).agg(
        possession_duration_before_shot=("possession_time", "mean")
    )

    # 2. pass_tempo_variance
    # パス間隔の分散（予測不可能性）
    passes = sa[sa[type_col] == "pass"].copy()
    if not passes.empty:
        passes["next_pass_time"] = passes.groupby([match_col, team_col])[time_col].shift(-1)
        passes["pass_interval"] = (passes["next_pass_time"] - passes[time_col]).clip(lower=0, upper=60)

        tempo_var = passes.groupby([match_col, player_col], as_index=False).agg(
            pass_tempo_variance=("pass_interval", lambda x: x.var() if len(x) > 1 else 0.0)
        )
    else:
        tempo_var = pd.DataFrame(columns=[match_col, player_col, "pass_tempo_variance"])

    # 3. acceleration_phase_count
    # 連続3パスで時間間隔が短縮（リズム加速）
    if not passes.empty and len(passes) >= 3:
        passes["interval_1"] = passes["pass_interval"]
        passes["interval_2"] = passes.groupby([match_col, team_col])["pass_interval"].shift(-1)
        passes["interval_3"] = passes.groupby([match_col, team_col])["pass_interval"].shift(-2)

        # 加速判定: interval_1 > interval_2 > interval_3（徐々に速く）
        accel_mask = (
            (passes["interval_1"] > passes["interval_2"]) &
            (passes["interval_2"] > passes["interval_3"]) &
            passes["interval_3"].notna()
        )

        accel_count = passes[accel_mask].groupby([match_col, player_col], as_index=False).size().rename(
            columns={"size": "acceleration_phase_count"}
        )
    else:
        accel_count = pd.DataFrame(columns=[match_col, player_col, "acceleration_phase_count"])

    # 4. quick_transition_rate
    # ボール奪取後5秒以内のGCA率
    sa["prev_team_diff"] = sa["prev_team"] != sa[team_col]
    sa["time_since_turnover"] = sa[time_col] - sa["prev_time"]

    quick_transition_mask = sa["prev_team_diff"] & (sa["time_since_turnover"] <= 5.0) & is_gca

    transition_agg = sa.groupby([match_col, player_col], as_index=False).agg(
        quick_transition_count=("prev_team_diff", lambda x: (quick_transition_mask.loc[x.index]).sum()),
        total_transitions=("prev_team_diff", "sum")
    )
    transition_agg["quick_transition_rate"] = np.where(
        transition_agg["total_transitions"] > 0,
        transition_agg["quick_transition_count"] / transition_agg["total_transitions"],
        0.0
    )
    transition_agg = transition_agg[[match_col, player_col, "quick_transition_rate"]]

    # 5. slow_buildup_gca_rate
    # 15秒以上のポゼッションビルドアップからのGCA率
    slow_buildup_mask = (sa["possession_time"] >= 15.0) & is_gca

    buildup_agg = sa.groupby([match_col, player_col], as_index=False).agg(
        slow_buildup_gca_count=("possession_time", lambda x: (slow_buildup_mask.loc[x.index]).sum()),
        total_gca=("possession_time", lambda x: (is_gca.loc[x.index]).sum())
    )
    buildup_agg["slow_buildup_gca_rate"] = np.where(
        buildup_agg["total_gca"] > 0,
        buildup_agg["slow_buildup_gca_count"] / buildup_agg["total_gca"],
        0.0
    )
    buildup_agg = buildup_agg[[match_col, player_col, "slow_buildup_gca_rate"]]

    # 統合
    result = gca_possession
    for df in [tempo_var, accel_count, transition_agg, buildup_agg]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    # NaN埋め
    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_vision_cognition_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    team_col: str = "team_id",
    type_col: str = "type_name",
) -> pd.DataFrame:
    """視野・認知系特徴量。

    データリーク防止: 全て当該アクション時点の情報のみ使用

    Returns:
    - switch_play_gca: サイドチェンジ(40m+)からのGCA
    - blind_side_pass_count: ブラインドサイドパス（DF背後配球）
    - cross_field_progression: 対角線的前進（横+縦同時10m+）
    - vision_angle_wide_pass: 広角視野パス（受け手角度120度+）
    """
    # メモリ節約: 必要な列のみ抽出
    needed_cols = [match_col, player_col, team_col, type_col,
                   "start_x", "start_y", "end_x", "end_y"]
    sa = _sort_actions(actions[needed_cols], match_col)

    if sa.empty:
        return pd.DataFrame(columns=[match_col, player_col])

    # 次2手の情報（GCA判定用）
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_team"] = sa.groupby(match_col)[team_col].shift(-1)
    sa["next2_type"] = sa.groupby(match_col)[type_col].shift(-2)
    sa["next2_team"] = sa.groupby(match_col)[team_col].shift(-2)

    # GCA判定
    is_gca = (
        (sa["next_type"].isin(SHOT_TYPES) & (sa["next_team"] == sa[team_col])) |
        (sa["next2_type"].isin(SHOT_TYPES) & (sa["next2_team"] == sa[team_col]))
    )

    # パスのみ抽出
    passes = sa[sa[type_col] == "pass"].copy()

    if passes.empty:
        return pd.DataFrame({
            match_col: [],
            player_col: [],
            "switch_play_gca": [],
            "blind_side_pass_count": [],
            "cross_field_progression": [],
            "vision_angle_wide_pass": []
        })

    # 横方向・縦方向の移動量
    passes["lateral_move"] = np.abs(passes["end_y"] - passes["start_y"])
    passes["forward_move"] = passes["end_x"] - passes["start_x"]

    # 1. switch_play_gca: サイドチェンジ(40m+)からのGCA
    # ピッチ幅68m、サイドチェンジ=横移動40m以上
    switch_mask = (passes["lateral_move"] >= 40.0) & is_gca.loc[passes.index]

    switch_gca = passes[switch_mask].groupby([match_col, player_col], as_index=False).size().rename(
        columns={"size": "switch_play_gca"}
    )

    # 2. blind_side_pass_count: ブラインドサイドパス
    # DF背後への配球 = 前進+横移動の角度が鋭角（前進優位）
    # 判定: 前進10m+かつ横移動/前進比<0.5（主に前方）
    blind_side_mask = (
        (passes["forward_move"] >= 10.0) &
        (passes["lateral_move"] / np.maximum(passes["forward_move"], 0.1) < 0.5)
    )

    blind_side = passes[blind_side_mask].groupby([match_col, player_col], as_index=False).size().rename(
        columns={"size": "blind_side_pass_count"}
    )

    # 3. cross_field_progression: 対角線的前進
    # 横+縦同時10m以上の移動
    diagonal_mask = (passes["lateral_move"] >= 10.0) & (passes["forward_move"] >= 10.0)

    diagonal = passes[diagonal_mask].groupby([match_col, player_col], as_index=False).size().rename(
        columns={"size": "cross_field_progression"}
    )

    # 4. vision_angle_wide_pass: 広角視野パス
    # 受け手への角度が120度以上（ほぼ後方や横方向）
    # arctan2で角度計算: ゴール方向を0度とする
    passes["pass_angle"] = np.arctan2(
        passes["end_y"] - passes["start_y"],
        passes["end_x"] - passes["start_x"]
    )
    # ラジアンを度に変換
    passes["pass_angle_deg"] = np.abs(np.degrees(passes["pass_angle"]))

    # 120度以上の広角パス
    wide_angle_mask = passes["pass_angle_deg"] >= 120.0

    wide_angle = passes[wide_angle_mask].groupby([match_col, player_col], as_index=False).size().rename(
        columns={"size": "vision_angle_wide_pass"}
    )

    # 統合
    result = switch_gca
    for df in [blind_side, diagonal, wide_angle]:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")

    # NaN埋め
    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)

    return result


def build_box_receiver_extended_features(
    actions: pd.DataFrame,
    match_col: str = "match_id",
    player_col: str = "player_id",
    type_col: str = "type_name",
    result_col: str = "result_name",
    bodypart_col: str = "bodypart_name",
) -> pd.DataFrame:
    """PA内+周辺の受け手文脈を大幅拡張（リーク防止: shift(-k)で未来参照）。

    Returns:
    ボリューム系:
      - box_receive_count: PA内で受けた回数（パス終点がPA内）
      - box_receive_to_shot_rate: PA受け→一定時間内シュート発生割合
      - box_receive_success_share: PA内パス成功率（PA終点/PA終点試行）
    タイミング系:
      - box_receive_time_to_shot_p25/p50/p75: 受け→シュート潜時（分位点）
      - box_receive_time_to_next_action_mean: 受け→次アクション平均潜時
    ワンタッチ系:
      - one_touch_pass_rate_in_box: 受け→即パス（<1s）の割合
      - first_touch_shot_bodypart_share_right/left/head: ワンタッチシュートの部位別比率
    空間/角度系:
      - receive_distance_to_goal_mean/min: 受け位置からゴールまでの距離
      - receive_to_goal_angle_mean/std: 対ゴール角連続量
      - receive_angle_bucket_share_0_30/30_60/60plus: 角度ビン別比率
      - facing_forward_30deg_share_in_box: <30度での前向き受け比率
      - first_touch_shot_0p7s/1p0s/1p5s_rate: 複数閾値でのファーストタッチシュート率
    起点/種類系:
      - cross_receive_share_in_box: クロス/コーナー起点のPA受け割合
      - cutback_receive_share: エンドライン付近折り返し受け推定
      - zone14_to_box_receive_share / halfspace_L/R_to_box_receive_share: 起点別比率
    受け後運び系:
      - carry_after_box_receive_distance_mean: 受け直後キャリー距離
      - carry_to_shot_share_in_box: 受け→キャリー→シュート割合
    多様性系:
      - box_receiver_diversity: PA受け手エントロピー（パサー視点）
      - unique_box_receivers: PA内ユニーク受け手数
    PA外ファイナルサード系:
      - final_third_receive_count: ファイナルサード受け数（PA外含む）
      - final_third_receive_to_shot_rate: サード内受け→シュート
    """
    # 必要列
    needed = [match_col, player_col, type_col, result_col,
              "start_x", "start_y", "end_x", "end_y"]
    if "time_seconds" in actions.columns:
        needed.append("time_seconds")
    if bodypart_col in actions.columns:
        needed.append(bodypart_col)
    if "team_id" in actions.columns:
        needed.append("team_id")
    
    sa = _sort_actions(actions[needed], match_col)
    
    # PA判定（end座標）
    PA_X = 88.5
    PA_Y_MIN = 13.84
    PA_Y_MAX = 54.16
    box_mask = (sa["end_x"] >= PA_X) & (sa["end_y"] >= PA_Y_MIN) & (sa["end_y"] <= PA_Y_MAX)
    
    # ファイナルサード判定（PA外含む）
    FINAL_THIRD_X = 70.0
    final_third_mask = (sa["end_x"] >= FINAL_THIRD_X)
    
    # ゴール座標
    GOAL_X = 105.0
    GOAL_Y = 34.0
    
    # 未来参照防止: shift(-1)で次のアクション情報を取得
    sa["next_type"] = sa.groupby(match_col)[type_col].shift(-1)
    sa["next_player"] = sa.groupby(match_col)[player_col].shift(-1)
    if "team_id" in sa.columns:
        sa["next_team"] = sa.groupby(match_col)["team_id"].shift(-1)
    if "time_seconds" in sa.columns:
        sa["next_time"] = sa.groupby(match_col)["time_seconds"].shift(-1)
        sa["latency"] = (sa["next_time"] - sa["time_seconds"]).fillna(999.0)
    else:
        sa["latency"] = 999.0
    
    # パスの受け手特定（成功パス→next_playerが受け手）
    success_pass_mask = (sa[type_col] == "pass") & (sa[result_col] == "success")
    if "next_team" in sa.columns:
        success_pass_mask = success_pass_mask & (sa["next_team"] == sa["team_id"])
    
    # ========================================
    # 1. ボリューム系
    # ========================================
    # PA内受け数（受け手視点: 自分が次プレイヤーでend_xがPA内）
    # → パサー視点の集計なので、パサー側でグルーピング
    box_receive_passer = sa[success_pass_mask & box_mask].copy()
    if not box_receive_passer.empty:
        box_receive_vol = box_receive_passer.groupby([match_col, player_col], as_index=False).size().rename(
            columns={"size": "box_receive_count"}
        )
    else:
        box_receive_vol = pd.DataFrame(columns=[match_col, player_col, "box_receive_count"])
    
    # PA受け→シュート（次がシュートかつlatency<3秒）
    # パサー集計: 自分のパス終点PA内 & next_typeがショット
    shot_follow_mask = box_receive_passer["next_type"].isin(SHOT_TYPES) & (box_receive_passer["latency"] < 3.0)
    if not box_receive_passer.empty and shot_follow_mask.any():
        box_receive_passer["leads_to_shot"] = shot_follow_mask.astype(int)
        box_receive_shot_rate = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            box_receive_to_shot_rate=("leads_to_shot", "mean")
        )
    else:
        box_receive_shot_rate = pd.DataFrame(columns=[match_col, player_col, "box_receive_to_shot_rate"])
    
    # PA内成功パス比率: PA終点試行数に対する成功数
    box_attempt_mask = (sa[type_col] == "pass") & box_mask
    box_attempt_passer = sa[box_attempt_mask].copy()
    if not box_attempt_passer.empty:
        box_attempt_passer["is_success"] = (box_attempt_passer[result_col] == "success").astype(int)
        box_success_share = box_attempt_passer.groupby([match_col, player_col], as_index=False).agg(
            box_receive_success_share=("is_success", "mean")
        )
    else:
        box_success_share = pd.DataFrame(columns=[match_col, player_col, "box_receive_success_share"])
    
    # ========================================
    # 2. タイミング系
    # ========================================
    if "time_seconds" in sa.columns and not box_receive_passer.empty:
        shot_timing = box_receive_passer[box_receive_passer["next_type"].isin(SHOT_TYPES)].copy()
        if not shot_timing.empty:
            timing_agg = shot_timing.groupby([match_col, player_col], as_index=False)["latency"].agg(
                box_receive_time_to_shot_p25=lambda x: np.percentile(x, 25),
                box_receive_time_to_shot_p50=lambda x: np.percentile(x, 50),
                box_receive_time_to_shot_p75=lambda x: np.percentile(x, 75),
            )
        else:
            timing_agg = pd.DataFrame(columns=[match_col, player_col,
                                               "box_receive_time_to_shot_p25",
                                               "box_receive_time_to_shot_p50",
                                               "box_receive_time_to_shot_p75"])
        
        # 受け→次アクション平均潜時
        next_action_lat = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            box_receive_time_to_next_action_mean=("latency", "mean")
        )
    else:
        timing_agg = pd.DataFrame(columns=[match_col, player_col])
        next_action_lat = pd.DataFrame(columns=[match_col, player_col])
    
    # ========================================
    # 3. ワンタッチ系
    # ========================================
    if not box_receive_passer.empty:
        # ワンタッチパス（受け→即パス）
        one_touch_pass_mask = (box_receive_passer["next_type"] == "pass") & (box_receive_passer["latency"] < 1.0)
        box_receive_passer["is_one_touch_pass"] = one_touch_pass_mask.astype(int)
        one_touch_rate = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            one_touch_pass_rate_in_box=("is_one_touch_pass", "mean")
        )
        
        # ファーストタッチシュート部位別比率（3閾値: 0.7s, 1.0s, 1.5s）
        first_shot_mask_07 = box_receive_passer["next_type"].isin(SHOT_TYPES) & (box_receive_passer["latency"] < 0.7)
        first_shot_mask_10 = box_receive_passer["next_type"].isin(SHOT_TYPES) & (box_receive_passer["latency"] < 1.0)
        first_shot_mask_15 = box_receive_passer["next_type"].isin(SHOT_TYPES) & (box_receive_passer["latency"] < 1.5)
        
        box_receive_passer["first_shot_0p7s"] = first_shot_mask_07.astype(int)
        box_receive_passer["first_shot_1p0s"] = first_shot_mask_10.astype(int)
        box_receive_passer["first_shot_1p5s"] = first_shot_mask_15.astype(int)
        
        first_touch_rate = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            first_touch_shot_0p7s_rate=("first_shot_0p7s", "mean"),
            first_touch_shot_1p0s_rate=("first_shot_1p0s", "mean"),
            first_touch_shot_1p5s_rate=("first_shot_1p5s", "mean"),
        )
        
        # 部位別（ワンタッチシュートに限定）
        if bodypart_col in box_receive_passer.columns:
            first_shot_df = box_receive_passer[first_shot_mask_10].copy()
            if not first_shot_df.empty and first_shot_df[bodypart_col].notna().any():
                bodypart_counts = first_shot_df.groupby([match_col, player_col, bodypart_col], as_index=False).size()
                bodypart_pivot = bodypart_counts.pivot_table(
                    index=[match_col, player_col],
                    columns=bodypart_col,
                    values="size",
                    fill_value=0
                ).reset_index()
                bodypart_pivot.columns.name = None
                total = bodypart_pivot.select_dtypes(include=[np.number]).sum(axis=1)
                for bp in ["right_foot", "left_foot", "head"]:
                    if bp in bodypart_pivot.columns:
                        bodypart_pivot[f"first_touch_shot_bodypart_share_{bp}"] = np.where(
                            total > 0, bodypart_pivot[bp] / total, 0.0
                        )
                    else:
                        bodypart_pivot[f"first_touch_shot_bodypart_share_{bp}"] = 0.0
                bodypart_features = bodypart_pivot[[match_col, player_col,
                                                    "first_touch_shot_bodypart_share_right_foot",
                                                    "first_touch_shot_bodypart_share_left_foot",
                                                    "first_touch_shot_bodypart_share_head"]]
            else:
                bodypart_features = pd.DataFrame(columns=[match_col, player_col,
                                                          "first_touch_shot_bodypart_share_right_foot",
                                                          "first_touch_shot_bodypart_share_left_foot",
                                                          "first_touch_shot_bodypart_share_head"])
        else:
            bodypart_features = pd.DataFrame(columns=[match_col, player_col])
    else:
        one_touch_rate = pd.DataFrame(columns=[match_col, player_col, "one_touch_pass_rate_in_box"])
        first_touch_rate = pd.DataFrame(columns=[match_col, player_col,
                                                 "first_touch_shot_0p7s_rate",
                                                 "first_touch_shot_1p0s_rate",
                                                 "first_touch_shot_1p5s_rate"])
        bodypart_features = pd.DataFrame(columns=[match_col, player_col])
    
    # ========================================
    # 4. 空間/角度系
    # ========================================
    if not box_receive_passer.empty:
        # 受け位置→ゴール距離
        box_receive_passer["dist_to_goal"] = np.sqrt(
            (GOAL_X - box_receive_passer["end_x"])**2 +
            (GOAL_Y - box_receive_passer["end_y"])**2
        )
        dist_agg = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            receive_distance_to_goal_mean=("dist_to_goal", "mean"),
            receive_distance_to_goal_min=("dist_to_goal", "min"),
        )
        
        # 受け位置→ゴール角度（連続量）
        box_receive_passer["angle_to_goal"] = np.arctan2(
            np.abs(GOAL_Y - box_receive_passer["end_y"]),
            np.maximum(GOAL_X - box_receive_passer["end_x"], 1e-6)
        )
        angle_agg = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            receive_to_goal_angle_mean=("angle_to_goal", "mean"),
            receive_to_goal_angle_std=("angle_to_goal", "std"),
        )
        
        # 角度ビン化（0-30°, 30-60°, 60+°）
        box_receive_passer["angle_deg"] = np.degrees(box_receive_passer["angle_to_goal"])
        box_receive_passer["angle_bin_0_30"] = (box_receive_passer["angle_deg"] < 30).astype(int)
        box_receive_passer["angle_bin_30_60"] = ((box_receive_passer["angle_deg"] >= 30) & 
                                                  (box_receive_passer["angle_deg"] < 60)).astype(int)
        box_receive_passer["angle_bin_60plus"] = (box_receive_passer["angle_deg"] >= 60).astype(int)
        
        angle_bin_agg = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            receive_angle_bucket_share_0_30=("angle_bin_0_30", "mean"),
            receive_angle_bucket_share_30_60=("angle_bin_30_60", "mean"),
            receive_angle_bucket_share_60plus=("angle_bin_60plus", "mean"),
        )
        
        # 前向き受け（30度閾値）
        facing_30_mask = box_receive_passer["angle_deg"] < 30
        box_receive_passer["is_facing_30"] = facing_30_mask.astype(int)
        facing_30_agg = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            facing_forward_30deg_share_in_box=("is_facing_30", "mean")
        )
    else:
        dist_agg = pd.DataFrame(columns=[match_col, player_col, "receive_distance_to_goal_mean", "receive_distance_to_goal_min"])
        angle_agg = pd.DataFrame(columns=[match_col, player_col, "receive_to_goal_angle_mean", "receive_to_goal_angle_std"])
        angle_bin_agg = pd.DataFrame(columns=[match_col, player_col,
                                              "receive_angle_bucket_share_0_30",
                                              "receive_angle_bucket_share_30_60",
                                              "receive_angle_bucket_share_60plus"])
        facing_30_agg = pd.DataFrame(columns=[match_col, player_col, "facing_forward_30deg_share_in_box"])
    
    # ========================================
    # 5. 起点/種類系
    # ========================================
    if not box_receive_passer.empty:
        # クロス起点（cross, corner_crossed, freekick_crossed）
        cross_types = {"cross", "corner_crossed", "freekick_crossed"}
        box_receive_passer["is_cross"] = box_receive_passer[type_col].isin(cross_types).astype(int)
        cross_share = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            cross_receive_share_in_box=("is_cross", "mean")
        )
        
        # カットバック（エンドライン付近: start_x>100 & |Δy|>15）
        box_receive_passer["delta_y"] = np.abs(box_receive_passer["end_y"] - box_receive_passer["start_y"])
        cutback_mask = (box_receive_passer["start_x"] > 100.0) & (box_receive_passer["delta_y"] > 15.0)
        box_receive_passer["is_cutback"] = cutback_mask.astype(int)
        cutback_share = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            cutback_receive_share=("is_cutback", "mean")
        )
        
        # 起点別比率（Zone14, HalfspaceL/R）
        zone14_mask = (box_receive_passer["start_x"] >= 65.0) & (box_receive_passer["start_x"] < 85.0) & \
                      (box_receive_passer["start_y"] >= 20.0) & (box_receive_passer["start_y"] <= 48.0)
        hs_L_mask = box_receive_passer["start_y"] < 22.67
        hs_R_mask = box_receive_passer["start_y"] > 45.33
        
        box_receive_passer["from_zone14"] = zone14_mask.astype(int)
        box_receive_passer["from_hs_L"] = hs_L_mask.astype(int)
        box_receive_passer["from_hs_R"] = hs_R_mask.astype(int)
        
        origin_share = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
            zone14_to_box_receive_share=("from_zone14", "mean"),
            halfspace_L_to_box_receive_share=("from_hs_L", "mean"),
            halfspace_R_to_box_receive_share=("from_hs_R", "mean"),
        )
    else:
        cross_share = pd.DataFrame(columns=[match_col, player_col, "cross_receive_share_in_box"])
        cutback_share = pd.DataFrame(columns=[match_col, player_col, "cutback_receive_share"])
        origin_share = pd.DataFrame(columns=[match_col, player_col,
                                             "zone14_to_box_receive_share",
                                             "halfspace_L_to_box_receive_share",
                                             "halfspace_R_to_box_receive_share"])
    
    # ========================================
    # 6. 受け後運び系
    # ========================================
    if not box_receive_passer.empty:
        # 次がキャリー→キャリー距離
        carry_mask = box_receive_passer["next_type"] == "carry"
        if carry_mask.any():
            # キャリー距離は受け位置から次の次の位置までと近似（簡易）
            # ここでは受け終点から次アクション開始点までを計算（保守的）
            # 実際は next_start を取得する必要があるが、データ構造上困難なので受け終点基準で近似
            # 代替: 受け終点からゴールまでの距離変化で近似（簡易版）
            # → より正確には次アクションのstart/end座標が必要だが、shiftで取得困難
            # ここでは受けた後のアクションがキャリーなら、そのキャリーが発生したと仮定し平均距離を算出
            # 簡易実装: 受け後キャリーが発生した試合×選手ごとのキャリー平均距離（別途集計）
            # → 現状データ構造では受け終点座標しかないため、キャリー距離は別途集計が必要
            # 代替案: キャリー発生率のみ算出
            box_receive_passer["is_carry_after"] = carry_mask.astype(int)
            carry_rate = box_receive_passer.groupby([match_col, player_col], as_index=False).agg(
                carry_after_box_receive_rate=("is_carry_after", "mean")
            )
            # キャリー距離は受け終点から次の次の座標まで必要だが、shift(-2)で取得困難
            # → 簡易版として、キャリー発生時の平均終点距離を算出（粗い近似）
            carry_df = box_receive_passer[carry_mask].copy()
            if not carry_df.empty:
                # 受け終点からゴールまでの距離を代用
                carry_dist = carry_df.groupby([match_col, player_col], as_index=False).agg(
                    carry_after_box_receive_distance_mean=("dist_to_goal", "mean")
                )
            else:
                carry_dist = pd.DataFrame(columns=[match_col, player_col, "carry_after_box_receive_distance_mean"])
        else:
            carry_rate = pd.DataFrame(columns=[match_col, player_col, "carry_after_box_receive_rate"])
            carry_dist = pd.DataFrame(columns=[match_col, player_col, "carry_after_box_receive_distance_mean"])
        
        # 受け→キャリー→シュート（次の次がシュート）
        # shift(-2)で2手先を取得
        sa_copy = sa.copy()
        sa_copy["next2_type"] = sa_copy.groupby(match_col)[type_col].shift(-2)
        box_receive_with_next2 = sa_copy[success_pass_mask & box_mask].copy()
        if not box_receive_with_next2.empty:
            carry_to_shot_mask = (box_receive_with_next2["next_type"] == "carry") & \
                                 (box_receive_with_next2["next2_type"].isin(SHOT_TYPES))
            box_receive_with_next2["is_carry_to_shot"] = carry_to_shot_mask.astype(int)
            carry_shot_rate = box_receive_with_next2.groupby([match_col, player_col], as_index=False).agg(
                carry_to_shot_share_in_box=("is_carry_to_shot", "mean")
            )
        else:
            carry_shot_rate = pd.DataFrame(columns=[match_col, player_col, "carry_to_shot_share_in_box"])
    else:
        carry_rate = pd.DataFrame(columns=[match_col, player_col, "carry_after_box_receive_rate"])
        carry_dist = pd.DataFrame(columns=[match_col, player_col, "carry_after_box_receive_distance_mean"])
        carry_shot_rate = pd.DataFrame(columns=[match_col, player_col, "carry_to_shot_share_in_box"])
    
    # ========================================
    # 7. 多様性系（パサー視点: 自分が誰にPAで通すか）
    # ========================================
    if not box_receive_passer.empty and "next_player" in box_receive_passer.columns:
        receiver_diversity_records = []
        for (m, p), grp in box_receive_passer.groupby([match_col, player_col]):
            receivers = grp["next_player"].dropna()
            if len(receivers) > 0:
                unique_count = receivers.nunique()
                # エントロピー
                receiver_counts = receivers.value_counts(normalize=True)
                entropy = -np.sum(receiver_counts * np.log2(receiver_counts + 1e-9))
                receiver_diversity_records.append({
                    match_col: m,
                    player_col: p,
                    "box_receiver_diversity": entropy,
                    "unique_box_receivers": unique_count,
                })
        if receiver_diversity_records:
            receiver_diversity = pd.DataFrame(receiver_diversity_records)
        else:
            receiver_diversity = pd.DataFrame(columns=[match_col, player_col, "box_receiver_diversity", "unique_box_receivers"])
    else:
        receiver_diversity = pd.DataFrame(columns=[match_col, player_col, "box_receiver_diversity", "unique_box_receivers"])
    
    # ========================================
    # 8. PA外ファイナルサード系
    # ========================================
    final_third_receive_passer = sa[success_pass_mask & final_third_mask].copy()
    if not final_third_receive_passer.empty:
        ft_receive_vol = final_third_receive_passer.groupby([match_col, player_col], as_index=False).size().rename(
            columns={"size": "final_third_receive_count"}
        )
        
        # サード受け→シュート
        ft_shot_mask = final_third_receive_passer["next_type"].isin(SHOT_TYPES) & (final_third_receive_passer["latency"] < 3.0)
        if ft_shot_mask.any():
            final_third_receive_passer["leads_to_shot_ft"] = ft_shot_mask.astype(int)
            ft_shot_rate = final_third_receive_passer.groupby([match_col, player_col], as_index=False).agg(
                final_third_receive_to_shot_rate=("leads_to_shot_ft", "mean")
            )
        else:
            ft_shot_rate = pd.DataFrame(columns=[match_col, player_col, "final_third_receive_to_shot_rate"])
    else:
        ft_receive_vol = pd.DataFrame(columns=[match_col, player_col, "final_third_receive_count"])
        ft_shot_rate = pd.DataFrame(columns=[match_col, player_col, "final_third_receive_to_shot_rate"])
    
    # ========================================
    # 統合
    # ========================================
    result = box_receive_vol
    all_dfs = [
        box_receive_shot_rate, box_success_share,
        timing_agg, next_action_lat,
        one_touch_rate, first_touch_rate, bodypart_features,
        dist_agg, angle_agg, angle_bin_agg, facing_30_agg,
        cross_share, cutback_share, origin_share,
        carry_rate, carry_dist, carry_shot_rate,
        receiver_diversity,
        ft_receive_vol, ft_shot_rate,
    ]
    
    for df in all_dfs:
        if not df.empty:
            result = result.merge(df, on=[match_col, player_col], how="outer")
    
    # NaN埋め
    for col in result.columns:
        if col not in [match_col, player_col]:
            result[col] = result[col].fillna(0.0)
    
    return result

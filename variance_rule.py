# variance_rule.py
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from dataclasses import dataclass


@dataclass
class _LogCfg:
    eps: float = 1e-6
    shift: float = 0.0
    kind: str = "log"  # or "log_shift"


def _choose(df_norm: pd.Series):
    s = pd.to_numeric(df_norm, errors="coerce").dropna()
    if len(s) < 50:
        return "robust"
    skew = s.skew()
    q01, q99 = s.quantile([0.01, 0.99])
    q25, q75 = s.quantile([0.25, 0.75])
    iqr = q75 - q25
    outlier_like = (iqr == 0) or (q99 - q01 > 6 * (iqr if iqr > 0 else 1))
    has_nonpos = (s <= 0).any()
    if abs(skew) > 1.0 and not has_nonpos:
        return "log"
    if outlier_like:
        return "robust"
    return "standard"


class VarianceRuleDetector(BaseEstimator, ClassifierMixin):
    """Date/Lot 집계 테이블 기반 임계값 룰 감지기. joblib 직렬화 가능."""

    def __init__(self, k_min_hits: int = 1):
        self.k_min_hits = k_min_hits
        self.transformers_ = {}
        self.thresholds_ = {}
        self.vars_ = []
        self.fitted_ = False

    def _fit_scalers(self, df_mean: pd.DataFrame, vars_):
        df_norm = df_mean[df_mean["ClusterLabel"] == 0]
        for v in vars_:
            method = _choose(df_norm[v])
            s = (
                pd.to_numeric(df_norm[v], errors="coerce")
                .dropna()
                .values.reshape(-1, 1)
            )
            if method == "standard":
                sc = StandardScaler().fit(s) if len(s) else StandardScaler()
                self.transformers_[v] = ("standard", sc)
            elif method == "robust":
                sc = RobustScaler().fit(s) if len(s) else RobustScaler()
                self.transformers_[v] = ("robust", sc)
            else:
                has_nonpos = (pd.to_numeric(df_norm[v], errors="coerce") <= 0).any()
                if has_nonpos:
                    shift = (
                        float(-pd.to_numeric(df_norm[v], errors="coerce").min() + 1e-6)
                        if df_norm[v].notna().any()
                        else 0.0
                    )
                    self.transformers_[v] = (
                        "log_shift",
                        _LogCfg(shift=shift, kind="log_shift"),
                    )
                else:
                    self.transformers_[v] = ("log", _LogCfg())

    def _t_scalar(self, val, tr):
        if pd.isna(val):
            return np.nan
        kind, obj = tr
        if kind == "standard":
            return (val - float(obj.mean_[0])) / float(obj.scale_[0] or 1.0)
        if kind == "robust":
            c = float(getattr(obj, "center_", [0.0])[0])
            s = float(getattr(obj, "scale_", [1.0])[0] or 1.0)
            return (val - c) / s
        if kind == "log":
            return np.log1p(val + obj.eps)
        if kind == "log_shift":
            return np.log1p(val + obj.shift + obj.eps)
        return val

    def _t_series(self, s, tr):  # 전체 컬럼 변환
        return s.apply(lambda v: self._t_scalar(v, tr))

    def fit_from_tables(self, df_mean: pd.DataFrame, df_thresh_z: pd.DataFrame):
        df_t = df_thresh_z.copy()
        if "direction" not in df_t.columns:
            df_t["direction"] = 1
        vars_ = sorted(set(df_t["변수"]).intersection(df_mean.columns))
        self._fit_scalers(df_mean, vars_)
        self.vars_ = vars_
        thr_map = {}
        for c, g in df_t.groupby("ClusterLabel"):
            rows = []
            for _, r in g.iterrows():
                v = r["변수"]
                tr = self.transformers_.get(v)
                if tr is None:
                    continue
                thr_tr = self._t_scalar(r["임계값"], tr)
                rows.append(
                    {"변수": v, "임계값_tr": thr_tr, "direction": int(r["direction"])}
                )
            thr_map[int(c)] = pd.DataFrame(rows)
        self.thresholds_ = thr_map
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.fitted_, "fit_from_tables 필요"
        d = df.copy()
        for v in self.vars_:
            if v in d.columns:
                d[v] = self._t_series(
                    pd.to_numeric(d[v], errors="coerce"), self.transformers_[v]
                )
        flags = []
        for _, row in d.iterrows():
            c = int(row.get("ClusterLabel", 0))
            if c == 0 or c not in self.thresholds_:
                flags.append(0)
                continue
            hits = 0
            for _, t in self.thresholds_[c].iterrows():
                v, thr, dct = t["변수"], t["임계값_tr"], t["direction"]
                val = row.get(v, np.nan)
                if pd.isna(val) or pd.isna(thr):
                    continue
                if dct * (val - thr) > 0:
                    hits += 1
            flags.append(1 if hits >= self.k_min_hits else 0)
        return np.array(flags, dtype=int)

    def explain(self, df: pd.DataFrame):
        assert self.fitted_, "fit_from_tables 필요"
        d = df.copy()
        for v in self.vars_:
            if v in d.columns:
                d[v] = self._t_series(
                    pd.to_numeric(d[v], errors="coerce"), self.transformers_[v]
                )
        out = []
        for _, row in d.iterrows():
            c = int(row.get("ClusterLabel", 0))
            if c == 0 or c not in self.thresholds_:
                out.append("")
                continue
            vs = []
            for _, t in self.thresholds_[c].iterrows():
                v, thr, dct = t["변수"], t["임계값_tr"], t["direction"]
                val = row.get(v, np.nan)
                if pd.notna(val) and pd.notna(thr) and dct * (val - thr) > 0:
                    vs.append(v)
            out.append(",".join(vs))
        return out

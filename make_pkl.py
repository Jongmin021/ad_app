# make_pkl.py
# 실행: python make_pkl.py
# 전제: 같은 폴더에 variance.py 존재

from pathlib import Path
import runpy, joblib, sys, numpy as np, pandas as pd, sklearn
from datetime import datetime

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

# 1) 모델링 스크립트 실행
ns = runpy.run_path(str(ROOT / "variance.py"), run_name="__main__")

# 2) 산출물 추출
req = ["transformers", "vars_with_thresholds", "cluster_thr_df_map", "k_min_hits"]
miss = [k for k in req if k not in ns]
if miss:
    raise RuntimeError(f"필수 변수 없음: {miss}")

bundle = {
    "transformers": ns["transformers"],
    "vars": ns["vars_with_thresholds"],
    "cluster_thr_df_map": {int(k): v for k, v in ns["cluster_thr_df_map"].items()},
    "k_min_hits": int(ns["k_min_hits"]),
    "meta": {
        "created_at": datetime.now().isoformat(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    },
}

# 3) 저장
out_path = OUT_DIR / "variance.pkl"
joblib.dump(bundle, out_path, compress=3)
print(f"saved -> {out_path}")

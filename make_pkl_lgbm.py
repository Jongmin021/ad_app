# make_pkl.py
import runpy, joblib, sys, numpy as np, pandas as pd, sklearn
from datetime import datetime

ns = runpy.run_path("lgbm.py", run_name="__main__")  # 파일명 교체
bundle = {
    "model": ns["best_model_top15"],
    "features": ns["top_15_features"],
    "classes_": getattr(ns["best_model_top15"], "classes_", None),
    "best_params": ns["study_top15"].best_params,
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
joblib.dump(bundle, "lgbm.pkl", compress=3)
print("saved -> lgbm.pkl")

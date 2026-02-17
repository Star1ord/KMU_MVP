import joblib, pickle, numpy as np
from pathlib import Path

models = {}
for name, path in [('nlp', Path('models/nlp/early_fusion_linear_svc.pkl')),
                    ('cv', Path('models/cv/model.joblib'))]:
    if path.exists():
        try:
            m = joblib.load(path) if str(path).endswith('.joblib') else pickle.load(open(path, 'rb'))
            X = np.random.randn(1, 10)
            score = m.predict_proba(X)[0, 1]
            print(f"✓ {name}: {score:.3f}")
            models[name] = m
        except Exception as e:
            print(f"✗ {name}: {e}")
    else:
        print(f"⚠ {name} not found")

print(f"\nTo start: python api/main.py")
print(f"Visit: http://localhost:8000/docs")
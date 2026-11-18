# Model artifacts

`comi_lingua_dev_tuned_augmented.joblib` is the three-class token language-identification model used by the desktop application. It expects the dependency versions listed in the root `requirements.txt`; scikit-learn 1.9.1 was recorded in the serialized estimator metadata.

The JSON files in this folder preserve the held-out COMI-LINGUA test report and the development-time threshold tuning report. Large per-token predictions and error dumps are intentionally excluded.

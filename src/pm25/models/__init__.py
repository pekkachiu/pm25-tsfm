# torch and xgboost each bundle libomp on macOS; loading torch first makes xgboost
# segfault, so xgboost (when installed) must be imported before torch.
try:
    import xgboost  # noqa: F401
except ImportError:  # pragma: no cover - base install without the models extra
    pass

# Regression baselines

This directory is the home for golden files used in Phase 3+ regression
tests. Phase 2 only wires fixtures; there are no assertion tests here yet.

## Current baseline pointers (defined in `tests/conftest.py`)

- `golden_tf_bf` — `test/bf_tf.npy` — EE transfer function from `test/new_estimator_test.py` run with the filter-binned (FB) SO maps.
- `golden_tf_ml` — `test/ml_tf.npy` — same estimator run on the ML-solved SO maps.

Both files are ~1.3 KB and are tracked in git on the `refactor` branch.

## Phase 3 regression test shape (to be added)

```python
def test_tf_ee_matches_legacy(tiny_config, golden_tf_bf):
    # 1. run the new TransferFunctionEE on the same inputs used to produce bf_tf.npy
    # 2. np.testing.assert_allclose(new_tf, golden_tf_bf, rtol=1e-10)
```

Baselines should be regenerated and re-committed only when a scientifically
justified change to the estimator is intentional; document the reason in the
commit message.

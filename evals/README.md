# Evals

The golden-set evaluation suite is the pre-deployment gate for the "no fake / stale / uncited results" non-negotiables. Eight scenarios drive the full validation → scoring → evidence → red team pipeline end-to-end.

## Run

```powershell
python evals/run_evals.py            # pass/fail summary
python evals/run_evals.py --verbose  # show per-test lines
```

Or directly via pytest:
```powershell
pytest tests/golden -m golden -v
```

## Scenarios

| # | Scenario | Expected outcome |
|---|----------|------------------|
| 1 | Known active job (JSON-LD `datePosted` within 14d) | Accepted, freshness `RECENT` |
| 2 | Known expired job (`"this position is no longer available"`) | Rejected at the Validation Agent (`"expired"`) |
| 3 | Same canonical URL discovered twice via different UTM params | First accepted, second rejected by Red Team (`duplicate_in_run`) |
| 4 | JSON-LD `JobPosting` with no `datePosted` and no meta date | Accepted, freshness `UNKNOWN`, no `posted_at`, no evidence (no "RECENT" claim) |
| 5 | Junior frontend role for an SVP life-sciences candidate | Rejected by Red Team (`match_below_threshold`) |
| 6 | VP AI Transformation in life sciences for the same candidate | Accepted, match >= 60 |
| 7 | Misleading title "VP AI Marketing Coordinator" with junior body | Rejected (body wins over title; score < 60) |
| 8 | 302 redirect to an ATS posting page | Accepted, `canonical_url` is the post-redirect URL, `validation.redirected=True` |

## Adding a scenario

1. Add a test class under `tests/golden/test_golden_set.py` with the `@pytest.mark.golden` marker (inherited from the module-level `pytestmark`).
2. Use the `_run_full_pipeline(handler, lead, criteria)` helper so the scenario exercises the same agents the production flow does.
3. Assert specifically — pin the expected rejection reason (`Reasons.*`) so future drift surfaces loudly.

## Why we don't hit the real internet

Every scenario uses `httpx.MockTransport` to provide deterministic HTML/JSON-LD responses. Tests run in <1 second and are reproducible across machines. Integration tests against real ATS pages would be marked `@pytest.mark.integration` (the marker is registered in `pytest.ini`) and run only on demand.

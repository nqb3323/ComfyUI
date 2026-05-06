"""Helper functions for assets integration tests."""
import time

import requests


def trigger_sync_seed_assets(session: requests.Session, base_url: str) -> None:
    """Force a synchronous sync/seed pass by calling the seed endpoint with wait=true.

    Retries on 409 (already running) until the previous scan finishes.
    """
    deadline = time.monotonic() + 60
    while True:
        r = session.post(
            base_url + "/api/assets/seed?wait=true",
            json={"roots": ["models", "input", "output"]},
            timeout=60,
        )
        if r.status_code != 409:
            assert r.status_code == 200, f"seed endpoint returned {r.status_code}: {r.text}"
            return
        if time.monotonic() > deadline:
            raise TimeoutError("seed endpoint stuck in 409 (already running)")
        time.sleep(0.25)


def get_asset_filename(asset_hash: str, extension: str) -> str:
    return asset_hash.removeprefix("blake3:") + extension


def assert_job_id_prompt_id_match(body: dict) -> None:
    """Assert that job_id and prompt_id are both present with the same value, or both absent."""
    job_present = "job_id" in body
    prompt_present = "prompt_id" in body
    assert job_present == prompt_present, (
        f"job_id and prompt_id must both be present or both absent: "
        f"job_id present={job_present}, prompt_id present={prompt_present}"
    )
    if job_present:
        job_id = body["job_id"]
        prompt_id = body["prompt_id"]
        assert job_id == prompt_id, (
            f"job_id and prompt_id must match: job_id={job_id!r}, prompt_id={prompt_id!r}"
        )

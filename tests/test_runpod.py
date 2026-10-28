import urllib.error
import urllib.request
from typing import Any

import pytest

from rigless import runpod


@pytest.mark.parametrize("name", ["rigless-app", "rigless-app-migration-2"])
def test_the_apps_pod_is_found_even_after_runpod_moves_it(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pods = [
        {"id": "made-by-hand", "name": "rigless"},
        {"id": "someone-elses", "name": "rigless-app-copy"},
        {"id": "ours", "name": name},
    ]
    monkeypatch.setattr(runpod, "_call", lambda key, method, url, body=None: pods)

    found = runpod.pod("rk")

    assert found is not None
    assert found["id"] == "ours"


def test_requests_carry_their_own_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # RunPod's REST API is behind Cloudflare, which bans Python-urllib's default
    # agent with a 403 that read as a refused key (Oct 26).
    sent: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> None:
        sent.append(request)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(runpod.RunPodError):
        runpod.pod("rk")

    agent = sent[0].get_header("User-agent") or "Python-urllib"
    assert "Python-urllib" not in agent


def test_runpod_answering_too_slowly_reads_as_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # urlopen raises a timeout while waiting for the answer as it is, not as URLError.
    def urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    with pytest.raises(runpod.RunPodError, match="Couldn't reach RunPod"):
        runpod.pod("rk")


def _deploy(monkeypatch: pytest.MonkeyPatch, pub_key: str | None) -> dict[str, Any]:
    """Deploys against a fake RunPod and returns the pod settings it was sent."""
    sent: dict[str, Any] = {}

    def call(key: str, method: str, url: str, body: Any = None) -> Any:
        if url != runpod.GRAPHQL:
            return []  # no pod yet
        if "myself" in body["query"]:
            return {"data": {"myself": {"pubKey": pub_key}}}
        sent.update(body["variables"]["input"])
        return {"data": {"podFindAndDeployOnDemand": {"id": "new"}}}

    monkeypatch.setattr(runpod, "_call", call)
    runpod.start("rk", "NVIDIA H100 80GB HBM3", {"PROXY_TOKEN": "t"})
    return sent


def test_deploy_takes_any_cuda_13_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    # RunPod's H100s run CUDA 13.2; asking for exactly 13.0 found none (Oct 26).
    sent = _deploy(monkeypatch, pub_key=None)

    assert sent["minCudaVersion"] == "13.0"
    assert "allowedCudaVersions" not in sent


def test_deploy_lets_the_accounts_ssh_key_in(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _deploy(monkeypatch, pub_key="ssh-ed25519 AAAA user@pc")

    assert {"key": "PUBLIC_KEY", "value": "ssh-ed25519 AAAA user@pc"} in sent["env"]
    assert {"key": "PROXY_TOKEN", "value": "t"} in sent["env"]


def test_time_left_follows_the_downloads_measured_speed() -> None:
    progress = {
        "stage": "models",
        "done": 30e9,
        "total": 60e9,
        "since": 1000.0,
        "from": 10e9,
    }

    estimate = runpod.setup_estimate(progress, now=1100.0)  # 20 GB in 100 s

    left = 30e9 / 200e6 + runpod.SETUP_STEPS["start"]
    assert estimate["seconds_left"] == pytest.approx(left)
    assert 0.5 < estimate["fraction"] < 1


def test_a_step_running_long_still_leaves_time() -> None:
    progress = {"stage": "torch", "since": 0.0}

    estimate = runpod.setup_estimate(progress, now=10_000.0)

    assert estimate["seconds_left"] > 0
    assert estimate["fraction"] < 1


def test_a_failed_setup_has_no_estimate() -> None:
    assert runpod.setup_estimate({"stage": "failed"}, now=0.0) == {}

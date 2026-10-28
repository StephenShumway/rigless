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

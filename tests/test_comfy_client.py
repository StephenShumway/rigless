import httpx

from rigless import comfy_client

POD = "https://abc123-8189.proxy.runpod.net"


def test_server_links_to_an_internal_address_go_to_the_pod_instead() -> None:
    # comfy-api-proxy behind RunPod's proxy links to its internal address (Oct 26).
    request = httpx.Request("GET", "http://100.65.33.167:60799/api/v2/jobs/7/events")

    comfy_client.pin_to(POD, "token")(request)

    assert str(request.url) == f"{POD}/api/v2/jobs/7/events"
    assert request.headers["Host"] == "abc123-8189.proxy.runpod.net"
    assert request.headers["Authorization"] == "Bearer token"


def test_requests_elsewhere_are_left_alone() -> None:
    request = httpx.Request("GET", "https://huggingface.co/some/file")

    comfy_client.pin_to(POD, "token")(request)

    assert str(request.url) == "https://huggingface.co/some/file"
    assert "Authorization" not in request.headers


def test_a_file_check_cloudflare_refused_reads_as_not_there() -> None:
    check = httpx.Request("HEAD", f"{POD}/api/v2/assets/by-hash/blake3:abc")
    response = httpx.Response(405, headers={"Allow": "HEAD"}, request=check)

    comfy_client.read_refused_file_check(response)

    assert response.status_code == 404


def test_other_refusals_are_left_alone() -> None:
    submit = httpx.Request("POST", f"{POD}/api/v2/jobs")
    response = httpx.Response(405, request=submit)

    comfy_client.read_refused_file_check(response)

    assert response.status_code == 405


import os
import socket
import httpx
import requests
import json
import asyncio

def _log(msg):
    print(f"[DIAG] {msg}")

def test_connectivity():
    target = "api.telegram.org"
    port = 443
    
    _log("--- Environment ---")
    _log(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
    _log(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
    _log(f"NO_PROXY: {os.environ.get('NO_PROXY')}")
    
    _log("--- DNS Resolution ---")
    try:
        addr = socket.getaddrinfo(target, port)
        _log(f"Resolved {target} to: {addr}")
    except Exception as e:
        _log(f"DNS Resolution failed: {e}")

    _log("--- TCP Connect ---")
    try:
        ips = [a[4][0] for a in socket.getaddrinfo(target, port, socket.AF_INET)]
        for ip in ips:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            try:
                s.connect((ip, port))
                _log(f"TCP Connect to {ip}:{port} SUCCESS")
            except Exception as e:
                _log(f"TCP Connect to {ip}:{port} FAILED: {e}")
            finally:
                s.close()
    except Exception as e:
        _log(f"TCP Phase failed: {e}")

    _log("--- Request Test (requests) ---")
    try:
        # requests uses urllib3
        r = requests.get(f"https://{target}/", timeout=10)
        _log(f"Requests SUCCESS: {r.status_code}")
    except Exception as e:
        _log(f"Requests FAILED: {e}")

    _log("--- Request Test (httpx - default) ---")
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"https://{target}/")
            _log(f"HTTPX Sync SUCCESS: {r.status_code}")
    except Exception as e:
        _log(f"HTTPX Sync FAILED: {e}")

    _log("--- Request Test (httpx - forced IPv4) ---")
    try:
        transport = httpx.HTTPTransport(local_address="0.0.0.0")
        with httpx.Client(transport=transport, timeout=10) as client:
            r = client.get(f"https://{target}/")
            _log(f"HTTPX forced IPv4 SUCCESS: {r.status_code}")
    except Exception as e:
        _log(f"HTTPX forced IPv4 FAILED: {e}")

if __name__ == "__main__":
    test_connectivity()

import socket
import urllib.request
import json
import time

def test_conn(host, ip, port=443):
    print(f"Testing {host} via {ip}:{port}...")
    try:
        # Create a socket and try to connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        start = time.time()
        sock.connect((ip, port))
        end = time.time()
        print(f"  [OK] Connected in {end-start:.2f}s")
        sock.close()
        return True
    except Exception as e:
        print(f"  [FAILED] {e}")
        return False

def resolve_all(hostname):
    apis = [
        f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
        f"https://dns.google/resolve?name={hostname}&type=A"
    ]
    ips = set()
    for api_url in apis:
        try:
            req = urllib.request.Request(api_url, headers={"Accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                for ans in data.get("Answer", []):
                    if ans.get("type") == 1:
                        ips.add(ans["data"])
        except Exception as e:
            print(f"DoH error ({api_url.split('/')[2]}): {e}")
    return list(ips)

if __name__ == "__main__":
    target_host = "api.telegram.org"
    print(f"Diagnostic for {target_host}")
    
    # Standard Telegram IPs often used by bots
    hardcoded_ips = [
        "149.154.167.220",
        "149.154.166.110",
        "149.154.167.99",
        "149.154.175.10",
        "91.108.4.110"
    ]
    
    doh_ips = resolve_all(target_host)
    print(f"DoH resolved IPs: {doh_ips}")
    
    all_ips = list(set(hardcoded_ips + doh_ips))
    
    results = []
    for ip in all_ips:
        if test_conn(target_host, ip):
            results.append(ip)
    
    print("-" * 20)
    if results:
        print(f"SUCCESS: Reachable IPs: {results}")
    else:
        print("CRITICAL: No Telegram IPs are reachable from this environment.")

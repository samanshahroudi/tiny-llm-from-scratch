"""Exercise the real HTTP server on an ephemeral local port, then stop it."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    folder = ROOT / "artifacts" / "runs" / "http_verification"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "tinyllm.api:app", "--host", "127.0.0.1",
            "--port", str(port), "--workers", "1",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(timeout=20) as client:
                for _ in range(30):
                    try:
                        response = client.get(url + "/health")
                        if response.status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if process.poll() is not None:
                        raise RuntimeError("Uvicorn exited; inspect server.log")
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Server startup timed out")
                assert client.get(url + "/").status_code == 200
                assert client.get(url + "/docs").status_code == 200
                body = {"prompt": "The model ", "max_new_tokens": 40, "temperature": 0.8, "seed": 7}
                generated = client.post(url + "/generate", json=body)
                generated.raise_for_status()
                assert generated.json()["generated_tokens"] == 40
                assert client.post(url + "/generate", json={"prompt": "😀"}).status_code == 422
                report = {"transport": "real HTTP via Uvicorn", "status": "passed",
                          "health": response.json(), "sample": generated.json()}
                (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()

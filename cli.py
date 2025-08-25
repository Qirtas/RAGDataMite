# cli.py
"""
Cross-platform CLI for RAGForDatamite dev tasks.

Usage examples:
  # 1) Build index only
  python cli.py index --persist-dir RAG/ProcessedDocuments/chroma_db

  # 2) Run API (Ctrl+C to stop)
  python cli.py run-api --port 8000

  # 3) One-shot test: build index -> start API -> POST /ask -> stop
  python cli.py ask --question "What is CAPEX?" --k 3 --min-similarity 0.28 --port 8000
"""

import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests
import typer

# Optional: load keys from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

app = typer.Typer(help="Dev commands for RAG RAGForDatamite (cross-platform).")

# ---- helpers -----------------------------------------------------------------

def _echo(msg: str):
    typer.echo(msg)

def _check_env(llm_provider: str):
    # Extend this if you add more providers later
    if llm_provider.lower() == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise typer.Exit(
                _exit_with_error(
                    "ANTHROPIC_API_KEY is not set. Export it (or put it in a .env) and re-run."
                )
            )

def _exit_with_error(msg: str, code: int = 1):
    _echo(typer.style(msg, fg=typer.colors.RED, bold=True))
    return code

def _env(**extra):
    env = os.environ.copy()
    for k, v in extra.items():
        if v is not None:
            env[k] = str(v)
    return env

def _wait_for_api(host: str, port: int, timeout: float = 20.0):
    """Poll the API until it responds or timeout is reached."""
    url = f"http://{host}:{port}/docs"
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    if last_err:
        _echo(f"Last connection error: {last_err}")
    return False

def _start_uvicorn(app_path: str, host: str, port: int, reload: bool, log_file: Optional[str] = None):
    args = [sys.executable, "-m", "uvicorn", app_path, "--host", host, "--port", str(port)]
    if reload:
        args.append("--reload")

    stdout = subprocess.PIPE
    stderr = subprocess.STDOUT
    if log_file:
        # Write to a file (append mode)
        f = open(log_file, "a", buffering=1, encoding="utf-8")
        proc = subprocess.Popen(args, stdout=f, stderr=f, env=os.environ.copy())
        return proc, f
    else:
        proc = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=os.environ.copy())
        return proc, None

def _stop_process(proc: subprocess.Popen, log_handle=None):
    try:
        if os.name == "nt":
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            # Try graceful
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        if log_handle:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

# ---- commands ----------------------------------------------------------------

@app.command()
def index(
    persist_dir: str = typer.Option("RAG/ProcessedDocuments/chroma_db", "--persist-dir", help="Chroma/VectorDB path"),
):
    """
    Build vector DB (calls: python main.py --mode index --persist_dir ...).
    """
    _echo(f"Building vector DB at: {persist_dir}")
    cmd = [sys.executable, "main.py", "--mode", "index", "--persist_dir", persist_dir]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(_exit_with_error(f"Indexing failed (exit {e.returncode})."))

@app.command("run-api")
def run_api(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
    app_path: str = typer.Option("RAG.api:app", "--app"),
    reload: bool = typer.Option(True, "--reload/--no-reload"),
):
    """
    Start FastAPI via uvicorn. Press Ctrl+C to stop.
    """
    _echo(f"Starting API: {app_path} at http://{host}:{port} (reload={reload})")
    proc, log_handle = _start_uvicorn(app_path, host, port, reload, log_file=None)
    try:
        # foreground attach: print uvicorn output
        # stream until Ctrl+C
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            try:
                sys.stdout.write(line.decode("utf-8", errors="ignore"))
            except Exception:
                pass
    except KeyboardInterrupt:
        _echo("\nStopping server...")
    finally:
        _stop_process(proc, log_handle)

@app.command()
def ask(
    question: str = typer.Option("What is CAPEX?", "--question", "-q"),
    k: int = typer.Option(3, "--k"),
    min_similarity: float = typer.Option(0.28, "--min-similarity"),
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
    persist_dir: str = typer.Option("RAG/ProcessedDocuments/chroma_db", "--persist-dir"),
    llm_provider: str = typer.Option("claude", "--llm-provider"),
    app_path: str = typer.Option("RAG.api:app", "--app"),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
    log_file: str = typer.Option(".server_claude.log", "--log-file"),
    build_index: bool = typer.Option(True, "--index/--no-index", help="Build vector DB before starting API"),
):
    """
    Full one-shot test: (optional) index -> start server -> POST /ask -> stop.
    Mirrors your bash script, but cross-platform.
    """
    # Set env needed by your app
    os.environ.setdefault("LLM_PROVIDER", llm_provider)
    os.environ.setdefault("PERSIST_DIR", persist_dir)

    _echo(f"Using LLM_PROVIDER={os.environ.get('LLM_PROVIDER')}")
    _echo(f"PERSIST_DIR={os.environ.get('PERSIST_DIR')}")

    # Provider-specific checks
    _check_env(llm_provider)

    # 1) build vector DB (optional)
    if build_index:
        _echo("Building vector DB...")
        cmd = [sys.executable, "main.py", "--mode", "index", "--persist_dir", persist_dir]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise typer.Exit(_exit_with_error(f"Indexing failed (exit {e.returncode})."))

    # 2) start API (background)
    _echo(f"Starting Uvicorn on port {port} ...")
    proc, handle = _start_uvicorn(app_path, host="0.0.0.0", port=port, reload=reload, log_file=log_file)

    # 3) wait until it's ready
    ok = _wait_for_api("127.0.0.1", port, timeout=25.0)
    if not ok:
        _stop_process(proc, handle)
        raise typer.Exit(_exit_with_error("Server did not come up in time. Check the log file for details."))

    # 4) send request
    url = f"http://127.0.0.1:{port}/ask"
    payload = {"question": question, "k": k, "min_similarity": min_similarity}
    _echo(f"Sending test request to {url}")
    try:
        r = requests.post(url, json=payload, timeout=120)
        try:
            parsed = r.json()
            _echo(json.dumps(parsed, indent=2)[:20000])  # print up to 20k chars
        except Exception:
            _echo(f"HTTP {r.status_code}\n{r.text[:20000]}")
    except Exception as e:
        _echo(f"Request failed: {e}")

    # 5) stop server
    _echo(f"Stopping server (PID {proc.pid})")
    _stop_process(proc, handle)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app()

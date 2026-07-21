import os, subprocess, sys
from pathlib import Path

def test_cli_help_runs(tmp_path: Path):
    env = os.environ.copy()
    env["LLM_API_KEY"] = ""
    env["SESSIONS_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert "session" in r.stdout.lower()
"""ahg.rawfetch: logged, append-only raw fetches with a verified manifest (BUILD_RULES rule 5)."""

import http.server
import json
import threading

import pytest

from ahg import rawfetch


@pytest.fixture
def server():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            code = 404 if self.path.startswith("/missing") else 200
            body = f"page {self.path} accept={self.headers.get('Accept')}".encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_fetch_logs_provenance_and_never_overwrites(tmp_path, server):
    f = rawfetch.Fetcher(tmp_path / "snap", "scripts/x.py", pace=0)
    rec = f.fetch("a/p.txt", server + "/p", {"Accept": "application/json"})
    data = (tmp_path / "snap" / "a" / "p.txt").read_bytes()
    assert data == b"page /p accept=application/json"
    assert rec["http"] == 200 and rec["sha256"] == rawfetch.sha256(data) and rec["bytes"] == len(data)
    assert {"date_utc", "git_commit", "git_dirty", "script", "url", "content_type"} <= set(rec)
    again = f.fetch("a/p.txt", server + "/other")
    assert again["skipped"] == "exists" and (tmp_path / "snap" / "a" / "p.txt").read_bytes() == data
    log = (tmp_path / "snap" / "fetch_log.jsonl").read_text().splitlines()
    assert len(log) == 1  # a skipped fetch is not a new fetch


def test_failed_fetch_writes_nothing_but_is_logged(tmp_path, server):
    f = rawfetch.Fetcher(tmp_path, "s", pace=0, retry_wait=0)
    rec = f.fetch("m.txt", server + "/missing")
    assert rec["http"] == 404 and rec["sha256"] is None and not (tmp_path / "m.txt").exists()
    m = rawfetch.write_manifest(tmp_path)
    assert m["files"] == [] and [r["name"] for r in m["failed_fetches"]] == ["m.txt"]


def test_manifest_rejects_unlogged_or_modified_files_and_freezes_snapshot(tmp_path, server):
    f = rawfetch.Fetcher(tmp_path, "s", pace=0)
    f.fetch("p.txt", server + "/p")
    (tmp_path / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="no successful fetch record"):
        rawfetch.write_manifest(tmp_path)
    (tmp_path / "stray.txt").unlink()
    (tmp_path / "p.txt").write_text("edited")
    with pytest.raises(ValueError, match="differ"):
        rawfetch.write_manifest(tmp_path)
    (tmp_path / "p.txt").write_bytes(b"page /p accept=None")
    m = rawfetch.write_manifest(tmp_path, {"api": "t"})
    assert [x["name"] for x in m["files"]] == ["p.txt"] and m["api"] == "t"
    assert json.loads((tmp_path / "manifest.json").read_text())["files"][0]["sha256"] == m["files"][0]["sha256"]
    with pytest.raises(FileExistsError):
        rawfetch.Fetcher(tmp_path, "s", pace=0)  # a snapshot with a manifest is final

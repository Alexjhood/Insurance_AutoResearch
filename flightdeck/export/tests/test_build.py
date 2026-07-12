from __future__ import annotations

import json
from pathlib import Path

from flightdeck.etl.build import build
from flightdeck.etl.tests.fixtures.make_mini_fixture import ORCH_ID, make_mini_fixture
from flightdeck.export.build import MAX_INLINE_FILE_BYTES, _json_for_script, build_export


def _fake_dist(root: Path) -> None:
    dist = root / "flightdeck" / "app" / "dist-export"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<head><script type="module" crossorigin src="./assets/app.js"></script>\n'
        '<link rel="stylesheet" crossorigin href="./assets/app.css"></head><div id="root"></div>',
        encoding="utf-8",
    )
    # The "</head>" string mimics DOMPurify's bundled markup constants: payload
    # injection must target the document's real head close, not this one.
    (dist / "assets" / "app.js").write_text('const closer = "</head>"; export {};', encoding="utf-8")
    (dist / "assets" / "app.css").write_text("body{}", encoding="utf-8")


def test_static_export_embeds_snapshot_telemetry_and_small_files(tmp_path: Path) -> None:
    make_mini_fixture(tmp_path)
    build(tmp_path, [ORCH_ID], log=lambda *_: None)
    _fake_dist(tmp_path)

    folder, archive = build_export(ORCH_ID, repo_root=tmp_path, build_app=False, log=lambda *_: None)

    html = (folder / "index.html").read_text(encoding="utf-8")
    assert 'id="fd-embedded-index"' in html
    assert f'id="fd-embedded-snapshot-{ORCH_ID}"' in html
    assert f'id="fd-embedded-telemetry-{ORCH_ID}-d01"' in html
    assert f'id="fd-embedded-files-{ORCH_ID}"' in html
    assert archive.is_file()
    # file:// blocks external module scripts, so the bundle must be inlined.
    assert '<script type="module">const closer = "</head>"; export {};</script>' in html
    assert "<style>body{}</style>" in html
    assert 'src="./assets/' not in html
    assert not (folder / "assets").exists()
    # Embedded payloads land in the real head, after the inlined bundle.
    assert html.index('id="fd-embedded-index"') > html.index("export {};")


def test_large_file_is_marked_not_included(tmp_path: Path) -> None:
    make_mini_fixture(tmp_path)
    build(tmp_path, [ORCH_ID], log=lambda *_: None)
    _fake_dist(tmp_path)
    snapshot_dir = tmp_path / "flightdeck" / "snapshots" / ORCH_ID
    large = snapshot_dir / "files" / "large.log"
    large.write_text("x" * MAX_INLINE_FILE_BYTES, encoding="utf-8")
    snapshot_path = snapshot_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["files"].append({"path": "large.log", "kind": "log", "bytes": MAX_INLINE_FILE_BYTES, "truncated": False, "source": "fixture"})
    snapshot_path.write_text(json.dumps(snapshot))

    folder, _ = build_export(ORCH_ID, repo_root=tmp_path, build_app=False, log=lambda *_: None)
    html = (folder / "index.html").read_text(encoding="utf-8")
    assert '"path":"large.log","kind":"log","bytes":524288,"truncated":false,"source":"fixture","included_in_export":false' in html


def test_script_payload_cannot_close_script_element() -> None:
    assert "</script>" not in _json_for_script({"body": "</script>"})

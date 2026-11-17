"""Builds a self-contained copy of the dashboard: the published artifact's
HTML template with dashboard_data_v2.json's contents inlined directly
(replacing the fetch() call), so it opens and renders correctly from disk
in any browser -- no server, no dependency on the artifact link staying
alive.

The HTML template itself lives wherever you're actively iterating on the
dashboard (e.g. the scratchpad while building it in a session) -- this
script just needs its path. Re-run after any dashboard HTML or data change.
"""

import pathlib
import sys

TEMPLATE_FETCH_BLOCK = """async function main(){
  let data;
  try {
    const res = await fetch('dashboard_data_v2.json');
    data = await res.json();
  } catch (e) {
    document.querySelector('.wrap').innerHTML = '<p class="loading">Could not load dashboard_data_v2.json</p>';
    return;
  }
"""

OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "dashboard.html"
DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "dashboard_data_v2.json"


def render(template_path: pathlib.Path) -> None:
    html = template_path.read_text(encoding="utf-8")
    data_json = DATA_PATH.read_text(encoding="utf-8")

    if TEMPLATE_FETCH_BLOCK not in html:
        raise SystemExit(
            "Template's main() doesn't match the expected fetch() block -- "
            "the dashboard HTML has changed shape; update TEMPLATE_FETCH_BLOCK "
            "in this script to match before re-running."
        )

    inlined = "async function main(){\n  const data = " + data_json + ";\n"
    html = html.replace(TEMPLATE_FETCH_BLOCK, inlined)

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python render_dashboard.py <path-to-dashboard-html-template>")
    render(pathlib.Path(sys.argv[1]))

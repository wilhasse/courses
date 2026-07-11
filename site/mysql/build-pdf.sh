#!/bin/bash
# Build one PDF per course series from the markdown chapters.
#
# Pipeline: concatenate chapters -> render mermaid diagrams to SVG
# (@mermaid-js/mermaid-cli using the local Chrome) -> pandoc to standalone
# HTML with TOC -> Chrome headless print-to-pdf. No LaTeX required.
#
# Usage: ./build-pdf.sh [workdir]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${1:-$(mktemp -d)}"
CHROME="${CHROME:-google-chrome}"
SERIES=(innodb-architecture server-architecture)
declare -A TITLES=(
  [innodb-architecture]="InnoDB Architecture Deep-Dive"
  [server-architecture]="MySQL Server Architecture Deep-Dive"
)
# Output PDF filenames (kept descriptive for standalone downloads)
declare -A OUTNAMES=(
  [innodb-architecture]="innodb-architecture"
  [server-architecture]="mysql-server-architecture"
)
SITE_BASE="https://wilhasse.github.io/courses/mysql"

command -v pandoc >/dev/null || { echo "pandoc not found"; exit 1; }
command -v "$CHROME" >/dev/null || { echo "chrome not found (set CHROME=)"; exit 1; }
mkdir -p "$WORKDIR"

cat > "$WORKDIR/puppeteer-config.json" <<EOF
{ "executablePath": "$(command -v "$CHROME")", "args": ["--no-sandbox"] }
EOF

for series in "${SERIES[@]}"; do
  src="$SCRIPT_DIR/$series"
  work="$WORKDIR/$series"
  mkdir -p "$work"
  combined="$work/combined.md"
  echo "== $series =="

  # 1. Concatenate README + chapters in order
  : > "$combined"
  for f in "$src/README.md" "$src"/[0-9][0-9]-*.md; do
    cat "$f" >> "$combined"
    printf '\n\n' >> "$combined"
  done

  # 2. Link fixes for a single-document PDF:
  #    - chapter links (./NN-xxx.md) -> internal anchors (#nn-xxx heading ids
  #      don't match filenames reliably, so point at the doc's TOC-friendly
  #      chapter title anchor is fragile; strip to plain text instead)
  #    - cross-series / PDF links -> absolute site URLs
  sed -i -E \
    -e 's|\((\./)?([0-9]{2}-[a-z0-9-]+)\.md\)|(#\2)|g' \
    -e 's|\(\.\./([a-z-]+)/README\.md\)|('"$SITE_BASE"'/\1/)|g' \
    -e 's|\(\./README\.md\)|('"$SITE_BASE"'/'"$series"'/)|g' \
    "$combined"
  # Give each chapter an anchor matching the rewritten links: insert an
  # explicit anchor line before each file's first heading using the filename.
  # (Chapters all start with "# Chapter ..." so we tag by scanning titles.)
  python3 - "$combined" "$src" <<'PYEOF'
import re, sys, pathlib
combined, src = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
text = combined.read_text()
for f in sorted(src.glob("[0-9][0-9]-*.md")):
    first = f.read_text().splitlines()[0].strip()
    anchor = f'<a id="{f.stem}"></a>\n\n'
    text = text.replace(first + "\n", anchor + first + "\n", 1)
combined.write_text(text)
PYEOF

  # 3. Render mermaid blocks to SVG images
  (cd "$work" && npx -y @mermaid-js/mermaid-cli \
      -p "$WORKDIR/puppeteer-config.json" \
      -i combined.md -o rendered.md \
      -b transparent --width 900 2>&1 | grep -v "^$" | tail -3)

  # 4. Standalone HTML with TOC + print CSS
  # --embed-resources is pandoc >= 2.19; older Debian pandoc uses --self-contained
  if pandoc --help 2>/dev/null | grep -q embed-resources; then
    EMBED="--embed-resources"
  else
    EMBED="--self-contained"
  fi
  pandoc "$work/rendered.md" -f gfm -s --toc --toc-depth=2 \
    $EMBED --resource-path="$work" \
    --metadata title="${TITLES[$series]}" \
    --metadata author="wilhasse/courses" \
    -c "$SCRIPT_DIR/pdf.css" \
    -o "$work/$series.html"

  # 5. Print to PDF via Chrome headless
  "$CHROME" --headless --disable-gpu --no-sandbox \
    --no-pdf-header-footer \
    --print-to-pdf="$SCRIPT_DIR/$series/${OUTNAMES[$series]}.pdf" \
    "file://$work/$series.html" 2>/dev/null
  echo "   -> $series/${OUTNAMES[$series]}.pdf"
done

echo "Done. Workdir: $WORKDIR"

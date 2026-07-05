#!/usr/bin/env bash
# Download and extract the O*NET 30.3 database (tab-delimited text).
# O*NET is licensed CC BY 4.0 by the U.S. Department of Labor, ETA.
# The raw dump (~110 MB) is intentionally NOT committed; this script fetches it.
set -euo pipefail

DEST="data/taxonomies/onet"
URL="https://www.onetcenter.org/dl_files/database/db_30_3_text.zip"

mkdir -p "$DEST"
echo "Downloading O*NET 30.3 ..."
curl -sSL -o "$DEST/db_30_3_text.zip" "$URL"

echo "Extracting ..."
unzip -oq "$DEST/db_30_3_text.zip" -d "$DEST"

echo "Done. Files in $DEST/db_30_3_text/"

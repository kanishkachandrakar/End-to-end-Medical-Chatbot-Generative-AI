#!/usr/bin/env bash
#
# Push the runtime files to a Hugging Face Space.
#
#   ./scripts/deploy_space.sh <hf-username>/<space-name>
#
# Only the files the container needs are copied. Data/Medical_book.pdf is
# deliberately left behind: the app never reads it (only store_index.py does,
# and that runs locally), and at 15MB it would need Git LFS on the Hub.
set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
    echo "usage: $(basename "$0") <hf-username>/<space-name>" >&2
    exit 2
fi

PAYLOAD=(
    Dockerfile
    README.md
    requirements.txt
    setup.py
    app.py
    src
    templates
    static
)

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

for item in "${PAYLOAD[@]}"; do
    if [ ! -e "$repo_root/$item" ]; then
        echo "missing from the repo: $item" >&2
        exit 1
    fi
done

echo "==> cloning https://huggingface.co/spaces/$SPACE"
git clone --quiet "https://huggingface.co/spaces/$SPACE" "$work/space"

echo "==> copying $(( ${#PAYLOAD[@]} )) entries"
for item in "${PAYLOAD[@]}"; do
    cp -R "$repo_root/$item" "$work/space/"
done

cd "$work/space"
git add -A

if git diff --cached --quiet; then
    echo "==> the Space already matches this checkout, nothing to push"
    exit 0
fi

source_rev="$(git -C "$repo_root" rev-parse --short HEAD)"
git commit --quiet -m "Deploy $source_rev"
git push --quiet

echo "==> pushed $source_rev"
echo "    build log: https://huggingface.co/spaces/$SPACE?logs=build"

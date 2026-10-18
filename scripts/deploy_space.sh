#!/usr/bin/env bash
#
# Push the runtime files to a Hugging Face Space.
#
#   ./scripts/deploy_space.sh <hf-username>/<space-name>
#
# SPACES_BASE overrides where the Space is cloned from. It exists so the script
# can be run against a local repository in a test; there is no other reason to
# set it.
#
# Only the files the container needs are copied. Data/Medical_book.pdf is
# deliberately left behind: the app never reads it (only store_index.py does,
# and that runs locally), and at 15MB it would need Git LFS on the Hub.
set -euo pipefail

SPACES_BASE="${SPACES_BASE:-https://huggingface.co/spaces}"

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
    echo "usage: $(basename "$0") <hf-username>/<space-name>" >&2
    exit 2
fi

PAYLOAD=(
    Dockerfile
    gunicorn.conf.py
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

echo "==> cloning $SPACES_BASE/$SPACE"
git clone --quiet "$SPACES_BASE/$SPACE" "$work/space"

echo "==> copying $(( ${#PAYLOAD[@]} )) entries"
for item in "${PAYLOAD[@]}"; do
    cp -R "$repo_root/$item" "$work/space/"
done

cd "$work/space"

# Stamp the revision into the Dockerfile before comparing. The Space builds the
# image itself, so there is no --build-arg to pass; baking the default in is the
# only way the running app can report which commit it came from. Doing it before
# the diff also means deploying a new commit always registers as a change.
source_rev="$(git -C "$repo_root" rev-parse --short HEAD)"
sed -i.bak "s/^ARG APP_REVISION=.*/ARG APP_REVISION=$source_rev/" Dockerfile
rm -f Dockerfile.bak

git add -A

if git diff --cached --quiet; then
    echo "==> the Space already matches this checkout, nothing to push"
    exit 0
fi

git commit --quiet -m "Deploy $source_rev"
git push --quiet

echo "==> pushed $source_rev"
echo "    build log: $SPACES_BASE/$SPACE?logs=build"

# The Space serves on a different host from the repository it is pushed to,
# derived from the owner and name. /healthz reports the revision it was built
# from, which is the one question worth asking after a deploy: did this one
# take effect, or is the old build still answering?
app_host="$(printf '%s' "$SPACE" | tr '/' '-' | sed 's/\.git$//')"
echo "    verify:    curl -s https://${app_host}.hf.space/healthz"
echo "               expect it to name $source_rev once the build finishes"

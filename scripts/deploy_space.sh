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

# The Space builds the image from its own copy of these files, so this list is
# the Space's build context. CI's image job is the only thing proving that
# build works, and it runs against this repository -- so anything here that is
# missing from the Space makes CI's proof cover a context the Space does not
# have. pyproject.toml is the one that was: nothing in the build reads it today
# (setup.py carries the package metadata, and the pip layer runs before
# `COPY . .` anyway), but dropping setup.py for PEP 621 metadata is the obvious
# next move, and it would leave CI green and the Space unable to build.
PAYLOAD=(
    Dockerfile
    gunicorn.conf.py
    README.md
    requirements.txt
    setup.py
    pyproject.toml
    app.py
    src
    templates
    static
)

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The revision that is about to be deployed, and -- see the archive below --
# the revision whose contents are deployed.
source_rev="$(git -C "$repo_root" rev-parse --short HEAD)"

# Tracked at HEAD, not merely present on disk: that is what gets shipped.
for item in "${PAYLOAD[@]}"; do
    if ! git -C "$repo_root" cat-file -e "HEAD:$item" 2>/dev/null; then
        echo "not tracked at HEAD: $item" >&2
        exit 1
    fi
done

echo "==> cloning $SPACES_BASE/$SPACE"
git clone --quiet "$SPACES_BASE/$SPACE" "$work/space"

# Clear each entry before writing it, so the Space is what this repository
# contains rather than the union of every deploy ever made. `cp -R src dest/`
# merges into an existing directory, so a module deleted here stayed there --
# and got baked into the Space's image, since the Space's build context has no
# .dockerignore to exclude anything.
echo "==> copying ${#PAYLOAD[@]} entries at $source_rev"
for item in "${PAYLOAD[@]}"; do
    rm -rf "$work/space/$item"
done

# `git archive` rather than cp: it writes what the commit contains, so the
# working tree's untracked leftovers stay out. cp took them -- a local
# src/__pycache__ put 18 .pyc files, built by two different interpreters, into
# the Space on every run. They also defeated the "nothing to push" check below,
# because running the suite between two deploys is enough to change them.
#
# It also makes the deploy consistent with its own stamp: the revision written
# into the Dockerfile is HEAD, so the contents shipped beside it should be
# HEAD's too, not whatever the tree happens to hold.
git -C "$repo_root" archive --format=tar HEAD -- "${PAYLOAD[@]}" \
    | tar -x -C "$work/space"

cd "$work/space"

# Stamp the revision into the Dockerfile before comparing. The Space builds the
# image itself, so there is no --build-arg to pass; baking the default in is the
# only way the running app can report which commit it came from. Doing it before
# the diff also means deploying a new commit always registers as a change.
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

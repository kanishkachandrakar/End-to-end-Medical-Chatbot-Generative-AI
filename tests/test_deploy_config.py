"""Three files have to agree on the port, or the Space serves nothing.

Spaces routes traffic to README.md's `app_port`. If gunicorn binds elsewhere
the build succeeds and the page never loads, with nothing in the build log to
explain it.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
DOCKERFILE = (ROOT / "Dockerfile").read_text()
GUNICORN = (ROOT / "gunicorn.conf.py").read_text()


def _frontmatter():
    block = re.match(r"^---\n(.*?)\n---\n", README, re.DOTALL)
    assert block, "README.md must open with the Space's YAML frontmatter"
    return dict(
        (k.strip(), v.strip())
        for k, v in (line.split(":", 1) for line in block.group(1).splitlines())
    )


def test_the_space_declares_the_docker_sdk():
    assert _frontmatter()["sdk"] == "docker"


def test_the_declared_port_matches_the_gunicorn_default():
    declared = _frontmatter()["app_port"]
    pattern = r"os\.environ\.get\(\s*['\"]PORT['\"]\s*,\s*['\"](\d+)['\"]"
    fallback = re.search(pattern, GUNICORN)
    assert fallback, "gunicorn.conf.py should fall back to a literal port"
    assert fallback.group(1) == declared, (
        f"README declares app_port {declared} but gunicorn binds {fallback.group(1)}"
    )


def test_the_dockerfile_exposes_the_same_port():
    exposed = re.search(r"^EXPOSE (\d+)", DOCKERFILE, re.MULTILINE)
    assert exposed and exposed.group(1) == _frontmatter()["app_port"]


def test_the_healthcheck_uses_the_same_fallback():
    fallback = re.search(r"os\.environ\.get\('PORT','(\d+)'\)", DOCKERFILE)
    assert fallback and fallback.group(1) == _frontmatter()["app_port"]


def test_the_container_runs_the_app_module_gunicorn_expects():
    assert "app:app" in DOCKERFILE
    app_py = (ROOT / "app.py").read_text()
    assert re.search(r"^app = create_app\(", app_py, re.MULTILINE)


def test_gunicorn_logs_to_stdout():
    """Anything else and a deployed Space has no request log at all."""
    assert re.search(r'accesslog\s*=\s*"-"', GUNICORN)
    assert re.search(r'errorlog\s*=\s*"-"', GUNICORN)


def test_a_single_worker_is_configured():
    """Two workers mean two copies of the embedding model on a 16GB box."""
    assert re.search(r"^workers\s*=\s*1$", GUNICORN, re.MULTILINE)

#!/usr/bin/env bash
# Deploy origin/main to the production checkout (default javelin:~/cfb-ratings).
# Run from a local checkout: deploy/deploy.sh   (override with DEPLOY_HOST / DEPLOY_DIR)
#
# The script sends itself over ssh, so the copy in your checkout runs, not the host's.
# On the host: refuse a dirty tree, hold the `cfbr weekly` lock, fast-forward to
# origin/main, uv sync, run the tests, rebuild out/web, restart cfbr-web and check it.
# Any failure after the fast-forward rolls the code and out/web back and restarts.
set -Eeuo pipefail

host="${DEPLOY_HOST:-javelin}"
app_dir="${DEPLOY_DIR:-cfb-ratings}"
base_url="http://127.0.0.1:8000"
web_log="data/web.log"

die() {
  echo "deploy: $*" >&2
  exit 1
}

restart_and_check() {
  local log_lines tries=0 path code failed=0
  log_lines="$(wc -l < "${web_log}" 2>/dev/null || echo 0)"
  systemctl --user restart cfbr-web
  until curl -fs -o /dev/null "${base_url}/"; do
    (( ++tries < 30 )) || { echo "deploy: cfbr-web did not come up" >&2; return 1; }
    sleep 1
  done
  for path in / /accuracy /bets /api/bets; do
    code="$(curl -s -o /dev/null -w '%{http_code}' "${base_url}${path}")"
    echo "${path} ${code}"
    [[ "${code}" == "200" ]] || failed=1
  done
  # grep reads to EOF (no -q): an early exit would SIGPIPE tail and pipefail would hide the match.
  if tail -n "+$(( log_lines + 1 ))" "${web_log}" | grep Traceback >&2; then
    echo "deploy: new traceback in ${web_log}" >&2
    failed=1
  fi
  return "${failed}"
}

rollback() {
  local prev="$1"
  trap - ERR
  echo "deploy: failed; rolling back to ${prev:0:7}" >&2
  git reset -q --hard "${prev}"
  uv sync --frozen -q
  uv run cfbr web build
  restart_and_check
  exit 1
}

remote() {
  cd "${HOME}/$1"
  export PATH="${HOME}/.local/bin:${PATH}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short >&2
    die "working tree is dirty; code reaches this host only through GitHub"
  fi

  # Same lock as `cfbr weekly` (cli.py): never swap code under a run, and none starts mid-deploy.
  exec 9> data/.weekly.lock
  flock -n 9 || die "cfbr weekly holds the lock; retry after the run"

  local prev new
  prev="$(git rev-parse HEAD)"
  git fetch -q origin
  git merge -q --ff-only origin/main || die "HEAD can't fast-forward to origin/main"
  new="$(git rev-parse HEAD)"
  if [[ "${new}" == "${prev}" ]]; then
    echo "deploy: already at ${new:0:7}, nothing to do"
    return 0
  fi
  git log --oneline "${prev}..${new}"

  trap 'rollback "${prev}"' ERR
  uv sync --frozen -q
  uv run pytest -q
  uv run cfbr web build
  restart_and_check
  trap - ERR
  echo "deploy: ${prev:0:7} -> ${new:0:7}"
}

# On the host the script itself is stdin; remote() gets /dev/null so nothing it starts
# (uv, git, pytest) can read from that pipe.
if [[ "${1:-}" == "--remote" ]]; then
  remote "$2" < /dev/null
else
  ssh "${host}" bash -s -- --remote "${app_dir}" < "${BASH_SOURCE[0]}"
fi

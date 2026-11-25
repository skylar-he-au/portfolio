#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

docker_config=$(mktemp -d)
trap 'rm -rf -- "$docker_config"' EXIT
printf '%s' '{"auths": {}}' > "$docker_config/config.json"

if [[ -d "$HOME/.docker/cli-plugins" ]]; then
  ln -s "$HOME/.docker/cli-plugins" "$docker_config/cli-plugins"
else
  printf '%s\n' 'Warning: ~/.docker/cli-plugins is missing; Docker CLI plugins must be available elsewhere.' >&2
fi

if [[ ! -f .env ]]; then
  printf '%s\n' 'Hint: cp .env.example .env' >&2
fi

case "${1:-}" in
  --down)
    # Only this checkout's app services; db keeps running, since other
    # sessions may be using it.
    DOCKER_CONFIG="$docker_config" docker compose rm --stop --force web api migrate
    printf '%s\n' 'Stopped web, api and migrate; db is still running. Stop everything with: ./scripts/dev-up.sh --down-all'
    ;;
  --down-all)
    # Everything in this compose project, db included; add -v to also delete the data volume.
    DOCKER_CONFIG="$docker_config" docker compose down "${@:2}"
    printf '%s\n' 'Stopped all services; without -v, the data volume is kept.'
    ;;
  *)
    DOCKER_CONFIG="$docker_config" docker compose up --build -d "$@"
    # web installs its packages (npm ci) only when its container starts, and `up` leaves a
    # running container alone when its compose config did not change; so a pull that changed
    # web's dependencies left the old packages in place (a page 500 on a missing package).
    # Restart web when web/package.json or package-lock.json is newer than its start.
    web_id=$(DOCKER_CONFIG="$docker_config" docker compose ps -q web)
    if [[ -n "$web_id" ]]; then
      started=$(docker inspect -f '{{.State.StartedAt}}' "$web_id")
      started_s=$(date -u -d "$started" +%s 2>/dev/null \
        || date -j -u -f '%Y-%m-%dT%H:%M:%S' "${started%%.*}" +%s)
      for file in web/package.json web/package-lock.json; do
        changed_s=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file")
        if (( changed_s > started_s )); then
          printf '%s\n' "$file changed after web started: restarting web (it runs npm ci, about a minute)"
          DOCKER_CONFIG="$docker_config" docker compose restart web
          break
        fi
      done
    fi
    printf 'Web: http://localhost:%s | API: http://localhost:%s/api/ping\n' "${WEB_PORT:-4321}" "${API_PORT:-8000}"
    printf '%s\n' 'Stop the app with: ./scripts/dev-up.sh --down (db keeps running); everything: --down-all'
    ;;
esac

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

if [[ "${1:-}" == --down ]]; then
  DOCKER_CONFIG="$docker_config" docker compose down "${@:2}"
else
  DOCKER_CONFIG="$docker_config" docker compose up --build -d "$@"
  printf 'Web: http://localhost:%s | API: http://localhost:%s/api/ping\n' "${WEB_PORT:-4321}" "${API_PORT:-8000}"
fi

printf '%s\n' 'docker compose down can also hit the credential helper; without -v, the data volume is kept.'
printf '%s\n' 'Stop with: ./scripts/dev-up.sh --down'

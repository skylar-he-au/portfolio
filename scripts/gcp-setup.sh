#!/usr/bin/env bash
# One-time Google Cloud setup for the production backend (SPEC "Service
# accounts", step 8). Skylar runs it after `gcloud auth login`; it never runs
# in CI. Safe to re-run: every step checks first and skips what already exists,
# and a secret that already has a version keeps it.
#
#   scripts/gcp-setup.sh --project portfolio-prod --domain example.com
#   scripts/gcp-setup.sh --project portfolio-prod --domain example.com --dry-run
#   scripts/gcp-setup.sh --project portfolio-prod --rotate origin-secret
#
# --dry-run calls no gcloud command at all (gcloud need not be installed); it
# prints every command in order as if nothing existed yet, secret values as
# <hidden>.
#
# gcloud references checked 2026-11-20 (cloud.google.com redirects to
# docs.cloud.google.com):
#   services enable          https://cloud.google.com/sdk/gcloud/reference/services/enable
#   WIF pool / provider      https://cloud.google.com/sdk/gcloud/reference/iam/workload-identity-pools/create
#                            https://cloud.google.com/sdk/gcloud/reference/iam/workload-identity-pools/providers/create-oidc
#                            https://cloud.google.com/sdk/gcloud/reference/iam/workload-identity-pools/undelete
#   WIF for GitHub, APIs,    https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines
#     principalSet format
#   AR writer role           https://cloud.google.com/artifact-registry/docs/access-control
#   AR cleanup policies      https://cloud.google.com/artifact-registry/docs/repositories/cleanup-policy
#                            https://cloud.google.com/sdk/gcloud/reference/artifacts/repositories/set-cleanup-policies
#   Secret Manager           https://cloud.google.com/sdk/gcloud/reference/secrets/create
#                            https://cloud.google.com/sdk/gcloud/reference/secrets/versions/add
#                            https://cloud.google.com/sdk/gcloud/reference/secrets/add-iam-policy-binding
#   Cloud Run deploy         https://cloud.google.com/sdk/gcloud/reference/run/deploy
#                            https://cloud.google.com/sdk/gcloud/reference/run/services/add-iam-policy-binding
#   Scheduler (HTTP target,  https://cloud.google.com/sdk/gcloud/reference/scheduler/jobs/create/http
#     no App Engine needed)  https://cloud.google.com/scheduler/docs/setup
#   Log exclusion            https://cloud.google.com/sdk/gcloud/reference/logging/sinks/update
set -euo pipefail

readonly REGION=australia-southeast1
readonly AR_REPO=portfolio
readonly SERVICE=portfolio-api
readonly PLACEHOLDER_IMAGE=us-docker.pkg.dev/cloudrun/container/hello
readonly WIF_POOL=github
readonly WIF_PROVIDER=portfolio
readonly GITHUB_REPO=skylar-he-au/portfolio
readonly SCHEDULER_JOB=portfolio-cleanup
readonly LOG_EXCLUSION=exclude-run-requests

readonly APIS=(
  run.googleapis.com
  artifactregistry.googleapis.com
  secretmanager.googleapis.com
  cloudscheduler.googleapis.com
  iamcredentials.googleapis.com
  sts.googleapis.com
  logging.googleapis.com
  # The WIF page also lists these two; service accounts and pools need them.
  iam.googleapis.com
  cloudresourcemanager.googleapis.com
)

# Secret Manager names (step 8 brief, agreement 1). Random ones are generated
# here; the rest are typed by Skylar.
readonly SECRETS=(
  anthropic-api-key gemini-api-key openai-api-key
  database-url database-url-direct
  origin-secret session-secret turnstile-secret patrol-secret
)
readonly RANDOM_SECRETS=(origin-secret session-secret patrol-secret)
# What the Cloud Run service reads (everything but the direct Neon URL).
readonly RUN_SECRETS=(
  anthropic-api-key gemini-api-key openai-api-key database-url
  origin-secret session-secret turnstile-secret patrol-secret
)
# What the deploy job fetches: migrations, Worker secret, patrol, ingest.
readonly DEPLOY_SECRETS=(database-url-direct origin-secret patrol-secret openai-api-key)

usage() {
  cat <<'EOF'
Usage:
  scripts/gcp-setup.sh --project PROJECT_ID --domain DOMAIN [--dry-run]
  scripts/gcp-setup.sh --project PROJECT_ID --rotate SECRET_NAME [--dry-run]

  --project   Google Cloud project ID (created by hand, DEPLOY.md A).
  --domain    The site's domain, e.g. example.com (printed as SITE_DOMAIN).
  --dry-run   Print the commands in order without calling gcloud.
  --rotate    Add a new version to one secret and print its number.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
have() { say "  exists: $*"; }

PROJECT=""
DOMAIN=""
ROTATE=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) [[ $# -ge 2 ]] || die "--project needs a value"; PROJECT=$2; shift 2 ;;
    --domain) [[ $# -ge 2 ]] || die "--domain needs a value"; DOMAIN=$2; shift 2 ;;
    --rotate) [[ $# -ge 2 ]] || die "--rotate needs a secret name"; ROTATE=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ -n "$PROJECT" ]] || { usage >&2; die "--project is required"; }
[[ "$PROJECT" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "not a project ID: $PROJECT"
if [[ -z "$ROTATE" ]]; then
  [[ -n "$DOMAIN" ]] || { usage >&2; die "--domain is required (except with --rotate)"; }
  [[ "$DOMAIN" =~ ^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$ ]] || die "not a bare domain: $DOMAIN"
fi

readonly RUN_SA="portfolio-run@${PROJECT}.iam.gserviceaccount.com"
readonly DEPLOY_SA="portfolio-deploy@${PROJECT}.iam.gserviceaccount.com"
readonly CLEANUP_SA="portfolio-cleanup@${PROJECT}.iam.gserviceaccount.com"

contains() {
  local needle=$1 item
  shift
  for item in "$@"; do [[ "$item" == "$needle" ]] && return 0; done
  return 1
}

# Print a command the way a shell would read it back.
show() {
  local out="" arg
  for arg in "$@"; do
    if [[ "$arg" == '|' || "$arg" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
      out+="$arg "
    elif [[ "$arg" != *"'"* ]]; then
      out+="'$arg' "
    else
      local esc=$arg
      esc=${esc//\\/\\\\}
      esc=${esc//\"/\\\"}
      esc=${esc//\$/\\\$}
      esc=${esc//\`/\\\`}
      out+="\"$esc\" "
    fi
  done
  printf '  + %s\n' "${out% }"
}

# A gcloud command that changes something: printed always, run unless --dry-run.
change() {
  show gcloud "$@" "--project=$PROJECT"
  if [[ $DRY_RUN -eq 0 ]]; then
    gcloud "$@" "--project=$PROJECT" --quiet
  fi
}

# Same, retried: IAM bindings on a service account created seconds ago can fail
# until the account has propagated.
change_retry() {
  show gcloud "$@" "--project=$PROJECT"
  [[ $DRY_RUN -eq 1 ]] && return 0
  local try
  for try in 1 2 3 4 5 6; do
    if gcloud "$@" "--project=$PROJECT" --quiet >/dev/null; then
      return 0
    fi
    say "  (retry $try in 10s)"
    sleep 10
  done
  die "failed: gcloud $*"
}

# A read-only gcloud command. Never called in a dry run.
look() {
  gcloud "$@" "--project=$PROJECT" 2>/dev/null
}

# Does ROLE for MEMBER already exist on a resource?
#   has_binding ROLE MEMBER "secrets" NAME [extra flags...]
has_binding() {
  local role=$1 member=$2 group=$3 resource=$4
  shift 4
  [[ $DRY_RUN -eq 1 ]] && return 1
  local words
  read -ra words <<<"$group"
  [[ -n "$(look "${words[@]}" get-iam-policy "$resource" "$@" \
    --flatten='bindings[].members' \
    --filter="bindings.role=$role AND bindings.members=$member" \
    --format='value(bindings.role)')" ]]
}

ensure_binding() {
  local role=$1 member=$2 group=$3 resource=$4
  shift 4
  local words
  read -ra words <<<"$group"
  if has_binding "$role" "$member" "$group" "$resource" "$@"; then
    have "$role for $member on $resource"
  else
    change_retry "${words[@]}" add-iam-policy-binding "$resource" "$@" \
      "--member=$member" "--role=$role"
  fi
}

env_name() {
  local upper
  upper=$(printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_')
  printf 'SECRET_VERSION_%s' "$upper"
}

latest_version() {
  look secrets versions list "$1" --filter='state:ENABLED' --sort-by='~createTime' \
    --limit=1 --format='value(name.basename())'
}

# Ask for a secret value without echoing it. Sets REPLY_SECRET.
prompt_secret() {
  local name=$1 value answer
  while true; do
    printf 'Value for %s (input hidden): ' "$name" >/dev/tty
    IFS= read -rs value </dev/tty
    printf '\n' >/dev/tty
    if [[ -z "$value" || "$value" =~ [[:space:]] ]]; then
      say "  empty or contains whitespace; type it again (Ctrl-C to stop)."
      continue
    fi
    local problem=""
    case "$name" in
      anthropic-api-key) [[ "$value" == sk-ant-* ]] || problem="does not start with sk-ant-" ;;
      openai-api-key) [[ "$value" == sk-* ]] || problem="does not start with sk-" ;;
      gemini-api-key) [[ "$value" == AIza* ]] || problem="does not start with AIza" ;;
      database-url)
        [[ "$value" =~ ^postgres(ql)?:// ]] || problem="is not a postgresql:// URL"
        [[ "$value" == *-pooler* ]] || problem="has no -pooler host (use Neon's pooled connection string)"
        ;;
      database-url-direct)
        [[ "$value" =~ ^postgres(ql)?:// ]] || problem="is not a postgresql:// URL"
        [[ "$value" != *-pooler* ]] || problem="has a -pooler host (use Neon's direct connection string)"
        ;;
    esac
    if [[ -n "$problem" ]]; then
      printf '  %s (%d characters) %s. Use it anyway? [y/N] ' "$name" "${#value}" "$problem" >/dev/tty
      IFS= read -r answer </dev/tty
      [[ "$answer" == [yY] ]] || continue
    fi
    say "  got ${#value} characters"
    REPLY_SECRET=$value
    return 0
  done
}

# Add one version to a secret; prints the new version number on stdout.
add_version() {
  local name=$1
  if contains "$name" "${RANDOM_SECRETS[@]}"; then
    show openssl rand -hex 32 '|' gcloud secrets versions add "$name" --data-file=- "--project=$PROJECT" >&2
    [[ $DRY_RUN -eq 1 ]] && { printf '<new-version>'; return 0; }
    openssl rand -hex 32 | tr -d '\n' |
      gcloud secrets versions add "$name" --data-file=- "--project=$PROJECT" \
        --format='value(name.basename())'
  else
    if [[ $DRY_RUN -eq 1 ]]; then
      say "  (read -s: prompt for $name)" >&2
      show printf %s '<hidden>' '|' gcloud secrets versions add "$name" --data-file=- "--project=$PROJECT" >&2
      printf '<new-version>'
      return 0
    fi
    REPLY_SECRET=""
    prompt_secret "$name" >&2
    printf '%s' "$REPLY_SECRET" |
      gcloud secrets versions add "$name" --data-file=- "--project=$PROJECT" \
        --format='value(name.basename())'
    REPLY_SECRET=""
  fi
}

preflight() {
  if [[ $DRY_RUN -eq 1 ]]; then
    say "Dry run: no gcloud command is called; every step is printed as if nothing existed yet."
    PROJECT_NUMBER="<project-number>"
    return 0
  fi
  command -v gcloud >/dev/null || die "gcloud is not installed (brew install --cask google-cloud-sdk)"
  command -v openssl >/dev/null || die "openssl is not installed"
  command -v python3 >/dev/null || die "python3 is not installed (xcode-select --install)"
  local account
  account=$(gcloud auth list --filter='status:ACTIVE' --format='value(account)' 2>/dev/null || true)
  [[ -n "$account" ]] || die "not logged in; run: gcloud auth login"
  PROJECT_NUMBER=$(look projects describe "$PROJECT" --format='value(projectNumber)' || true)
  [[ -n "$PROJECT_NUMBER" ]] || die "project $PROJECT not found, or $account cannot see it"
  say "Project $PROJECT ($PROJECT_NUMBER) as $account"
}

# ---------------------------------------------------------------- --rotate
rotate() {
  contains "$ROTATE" "${SECRETS[@]}" || die "--rotate takes one of: ${SECRETS[*]}"
  preflight
  local old=""
  if [[ $DRY_RUN -eq 0 ]]; then
    look secrets describe "$ROTATE" --format='value(name)' >/dev/null ||
      die "secret $ROTATE does not exist yet; run the full setup first"
    old=$(latest_version "$ROTATE")
  fi
  step "Rotate $ROTATE"
  local new
  new=$(add_version "$ROTATE")
  [[ -n "$new" ]] || die "gcloud did not return the new version number"
  printf '\nNew version of %s: %s\n' "$ROTATE" "$new"
  printf 'Set GitHub variable %s = %s (environment production), then run the deploy workflow.\n' \
    "$(env_name "$ROTATE")" "$new"
  if [[ "$ROTATE" == origin-secret ]]; then
    say "The deploy also hands the new ORIGIN_SECRET to the Worker."
  fi
  if [[ "$ROTATE" == session-secret ]]; then
    say "Every visitor's session ends; the site gets them a new one on their next question."
  fi
  if [[ -n "$old" ]]; then
    say "After the deploy is green, retire the old version (each enabled version is billed):"
    say "  gcloud secrets versions destroy $old --secret=$ROTATE --project=$PROJECT"
  fi
}

# ---------------------------------------------------------------- setup steps
enable_apis() {
  step "APIs"
  local enabled="" missing=() api
  if [[ $DRY_RUN -eq 0 ]]; then
    enabled=$(look services list --enabled --format='value(config.name)' || true)
  fi
  for api in "${APIS[@]}"; do
    if grep -qx "$api" <<<"$enabled"; then have "$api"; else missing+=("$api"); fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    change services enable "${missing[@]}"
  fi
}

service_accounts() {
  step "Service accounts"
  local id email
  for id in portfolio-run portfolio-deploy portfolio-cleanup; do
    email="${id}@${PROJECT}.iam.gserviceaccount.com"
    if [[ $DRY_RUN -eq 0 ]] && look iam service-accounts describe "$email" --format='value(email)' >/dev/null; then
      have "$email"
    else
      change iam service-accounts create "$id" "--display-name=$id"
    fi
  done
}

artifact_registry() {
  step "Artifact Registry"
  if [[ $DRY_RUN -eq 0 ]] && look artifacts repositories describe "$AR_REPO" "--location=$REGION" --format='value(name)' >/dev/null; then
    have "repository $AR_REPO"
  else
    change artifacts repositories create "$AR_REPO" --repository-format=docker "--location=$REGION" \
      '--description=Portfolio API images'
  fi

  # Keep the 3 newest versions of each image; delete the rest. A keep policy
  # only exempts, so the delete policy is what removes anything.
  local policies="" dry_mode=""
  if [[ $DRY_RUN -eq 0 ]]; then
    policies=$(look artifacts repositories list-cleanup-policies "$AR_REPO" "--location=$REGION" --format=json || true)
    dry_mode=$(look artifacts repositories describe "$AR_REPO" "--location=$REGION" --format='value(cleanupPolicyDryRun)' || true)
  fi
  local policy_file
  policy_file=$(mktemp "${TMPDIR:-/tmp}/ar-cleanup-policy.XXXXXX")
  cat >"$policy_file" <<'EOF'
[
  {"name": "keep-recent-3", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
EOF
  if [[ "$policies" == *keep-recent-3* ]]; then
    have "cleanup policies"
  else
    # Dry-run mode first: the API validates the policy without deleting
    # anything. (Its log of what would go appears only after about a day; the
    # repository is empty here, so there is nothing to preview.)
    change artifacts repositories set-cleanup-policies "$AR_REPO" "--location=$REGION" \
      "--policy=$policy_file" --dry-run
    dry_mode=True
  fi
  case "$dry_mode" in
    True | true)
      change artifacts repositories set-cleanup-policies "$AR_REPO" "--location=$REGION" \
        "--policy=$policy_file" --no-dry-run
      ;;
    *) have "cleanup policies active (not dry run)" ;;
  esac
  rm -f -- "$policy_file"
}

# Version of each secret, same order as SECRETS (bash 3.2 has no maps).
VERSIONS=()

secrets() {
  step "Secrets"
  local name version
  for name in "${SECRETS[@]}"; do
    if [[ $DRY_RUN -eq 0 ]] && look secrets describe "$name" --format='value(name)' >/dev/null; then
      have "secret $name"
    else
      change secrets create "$name" --replication-policy=user-managed "--locations=$REGION"
    fi
    version=""
    [[ $DRY_RUN -eq 0 ]] && version=$(latest_version "$name" || true)
    if [[ -n "$version" ]]; then
      have "$name version $version (kept)"
    else
      version=$(add_version "$name")
      [[ -n "$version" ]] || die "gcloud did not return a version for $name"
      say "  $name: version $version"
    fi
    VERSIONS+=("$version")
  done
}

roles() {
  step "Roles"
  local name
  for name in "${RUN_SECRETS[@]}"; do
    ensure_binding roles/secretmanager.secretAccessor "serviceAccount:$RUN_SA" secrets "$name"
  done
  for name in "${DEPLOY_SECRETS[@]}"; do
    ensure_binding roles/secretmanager.secretAccessor "serviceAccount:$DEPLOY_SA" secrets "$name"
  done
  # Deploy can run revisions as portfolio-run, and push images to this repository.
  ensure_binding roles/iam.serviceAccountUser "serviceAccount:$DEPLOY_SA" \
    "iam service-accounts" "$RUN_SA"
  ensure_binding roles/artifactregistry.writer "serviceAccount:$DEPLOY_SA" \
    "artifacts repositories" "$AR_REPO" "--location=$REGION"
  # roles/run.developer is granted on the service itself once it exists (below).
  # portfolio-cleanup needs no role: the service is public and the backend
  # checks the OIDC token's audience and email.
}

github_ids() {
  local json
  if ! json=$(curl -fsS "https://api.github.com/repos/$GITHUB_REPO"); then
    [[ $DRY_RUN -eq 1 ]] || die "could not read https://api.github.com/repos/$GITHUB_REPO"
    REPO_ID="<repository_id>" OWNER_ID="<repository_owner_id>"
    return 0
  fi
  REPO_ID=$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  OWNER_ID=$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["owner"]["id"])')
  [[ "$REPO_ID" =~ ^[0-9]+$ && "$OWNER_ID" =~ ^[0-9]+$ ]] || die "unexpected ids from GitHub: $REPO_ID $OWNER_ID"
}

wif() {
  step "Workload Identity Federation"
  github_ids
  say "  $GITHUB_REPO: repository_id $REPO_ID, repository_owner_id $OWNER_ID"
  local state=""
  [[ $DRY_RUN -eq 0 ]] && state=$(look iam workload-identity-pools describe "$WIF_POOL" --location=global --format='value(state)' || true)
  case "$state" in
    "") change iam workload-identity-pools create "$WIF_POOL" --location=global '--display-name=GitHub Actions' ;;
    DELETED) die "pool $WIF_POOL is soft-deleted; run: gcloud iam workload-identity-pools undelete $WIF_POOL --location=global --project=$PROJECT" ;;
    *) have "pool $WIF_POOL ($state)" ;;
  esac

  state=""
  [[ $DRY_RUN -eq 0 ]] && state=$(look iam workload-identity-pools providers describe "$WIF_PROVIDER" \
    --location=global "--workload-identity-pool=$WIF_POOL" --format='value(state)' || true)
  case "$state" in
    "")
      change iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
        --location=global "--workload-identity-pool=$WIF_POOL" \
        '--display-name=portfolio repo, main' \
        --issuer-uri=https://token.actions.githubusercontent.com \
        --attribute-mapping=google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.repository_owner_id=assertion.repository_owner_id,attribute.ref=assertion.ref \
        "--attribute-condition=assertion.repository_owner_id == '$OWNER_ID' && assertion.repository_id == '$REPO_ID' && assertion.ref == 'refs/heads/main'"
      ;;
    DELETED) die "provider $WIF_PROVIDER is soft-deleted; run: gcloud iam workload-identity-pools providers undelete $WIF_PROVIDER --workload-identity-pool=$WIF_POOL --location=global --project=$PROJECT" ;;
    *) have "provider $WIF_PROVIDER ($state)" ;;
  esac

  # Tokens from this repository may act as portfolio-deploy (the provider's
  # condition already limits them to main).
  ensure_binding roles/iam.workloadIdentityUser \
    "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository_id/$REPO_ID" \
    "iam service-accounts" "$DEPLOY_SA"
}

cloud_run() {
  step "Cloud Run"
  SERVICE_URL=""
  [[ $DRY_RUN -eq 0 ]] && SERVICE_URL=$(look run services describe "$SERVICE" "--region=$REGION" --format='value(status.url)' || true)
  if [[ -n "$SERVICE_URL" ]]; then
    have "service $SERVICE ($SERVICE_URL)"
  else
    # Placeholder image; the deploy workflow replaces it. Public service: the
    # Worker's secret header guards the paid endpoints.
    change run deploy "$SERVICE" "--image=$PLACEHOLDER_IMAGE" "--region=$REGION" \
      "--service-account=$RUN_SA" --ingress=all --no-invoker-iam-check \
      --timeout=150 --min-instances=0 --max-instances=2
    if [[ $DRY_RUN -eq 1 ]]; then
      SERVICE_URL="<service-url>"
    else
      SERVICE_URL=$(look run services describe "$SERVICE" "--region=$REGION" --format='value(status.url)')
    fi
  fi
  SERVICE_URL=${SERVICE_URL%/}
  [[ -n "$SERVICE_URL" ]] || die "could not read the service URL"
  ensure_binding roles/run.developer "serviceAccount:$DEPLOY_SA" \
    "run services" "$SERVICE" "--region=$REGION"
}

scheduler() {
  step "Cloud Scheduler"
  local audience="$SERVICE_URL/api/cleanup"
  if [[ $DRY_RUN -eq 0 ]] && look scheduler jobs describe "$SCHEDULER_JOB" "--location=$REGION" --format='value(name)' >/dev/null; then
    have "job $SCHEDULER_JOB"
  else
    change scheduler jobs create http "$SCHEDULER_JOB" "--location=$REGION" \
      '--schedule=0 3 * * *' --time-zone=Australia/Sydney \
      --http-method=POST "--uri=$audience" \
      "--oidc-service-account-email=$CLEANUP_SA" "--oidc-token-audience=$audience"
  fi
}

log_exclusion() {
  step "Logging"
  local sink=""
  [[ $DRY_RUN -eq 0 ]] && sink=$(look logging sinks describe _Default --format=json || true)
  if [[ "$sink" == *"\"$LOG_EXCLUSION\""* ]]; then
    have "exclusion $LOG_EXCLUSION on _Default"
  else
    # Request logs carry visitor IPs and paths; the app's own stdout logs stay.
    change logging sinks update _Default \
      "--add-exclusion=name=$LOG_EXCLUSION,filter=log_id(\"run.googleapis.com/requests\")"
  fi
}

print_variables() {
  local name
  step "GitHub: $GITHUB_REPO -> Settings -> Environments -> production -> Variables"
  printf '  %-36s %s\n' \
    SITE_DOMAIN "$DOMAIN" \
    GCP_WIF_PROVIDER "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/providers/$WIF_PROVIDER" \
    GCP_DEPLOY_SA "$DEPLOY_SA" \
    ORIGIN_URL "$SERVICE_URL" \
    CLEANUP_AUDIENCE "$SERVICE_URL/api/cleanup" \
    CLEANUP_INVOKER_EMAIL "$CLEANUP_SA"
  local i
  for i in "${!SECRETS[@]}"; do
    name=${SECRETS[$i]}
    if [[ $DRY_RUN -eq 1 ]]; then
      printf '  %-36s %s\n' "$(env_name "$name")" "<version of $name>"
    else
      printf '  %-36s %s\n' "$(env_name "$name")" "${VERSIONS[$i]}"
    fi
  done
  say ""
  say "  Not from this script (DEPLOY.md A):"
  printf '  %-36s %s\n' \
    CLOUDFLARE_ACCOUNT_ID "Cloudflare dashboard, Account ID" \
    PUBLIC_TURNSTILE_SITEKEY "Turnstile widget Site Key"
  say "  Environment secrets: CLOUDFLARE_API_TOKEN, PATROL_GEMINI_API_KEY"
}

main() {
  if [[ -n "$ROTATE" ]]; then
    rotate
    return 0
  fi
  preflight
  enable_apis
  service_accounts
  artifact_registry
  secrets
  roles
  wif
  cloud_run
  scheduler
  log_exclusion
  print_variables
  say ""
  if [[ $DRY_RUN -eq 1 ]]; then
    say "Dry run done; nothing was changed."
  else
    say "Done. Re-running is safe: existing resources and secret versions are left alone."
  fi
}

main

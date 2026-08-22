#!/usr/bin/env bash
# Verifies a deployed fdml-api on Cloud Run is healthy, ready, and serving.
#
# Usage:
#   scripts/verify_deploy.sh <base_url>
#     base_url       Cloud Run URL, e.g. https://fdml-api-<hash>.run.app
#   TRANSACTION_ID=<id>  override the sample TransactionID used (default 3538759)
#
# Prints one line per check; exits non-zero on the first failure.

set -euo pipefail

BASE_URL="${1:?usage: verify_deploy.sh <base_url>}"
TRANSACTION_ID="${TRANSACTION_ID:-3538759}"

say() { printf '%s\n' "$*"; }
fail() { printf '✗ %s\n' "$*" >&2; exit 1; }

get() { curl -sS --fail "$BASE_URL$1"; }
jget() { python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

say "== Checking $BASE_URL =="

# 1) Health (probe the container is up)
status=$(get /v1/health/ | jget status)
[ "$status" = "ok" ] || fail "/v1/health/ status=$status"
say "✓ /v1/health/ ok"

# 2) Readiness — the model must actually be loaded (secrets DAGSHUB_* + registry).
#    Since the fix, /v1/ready/ returns 503 with model_loaded=false when the
#    model failed to load, so treat both a non-200 and a false flag as fatal.
ready_code=$(curl -sS -o /tmp/fdml_ready.json -w "%{http_code}" "$BASE_URL/v1/ready/")
ready_loaded=$(jget model_loaded < /tmp/fdml_ready.json)
if [ "$ready_code" != "200" ] || [ "$ready_loaded" != "True" ]; then
    fail "/v1/ready/ http=$ready_code model_loaded=$ready_loaded — revisa secrets DAGSHUB_USERNAME/TOKEN/REPO"
fi
rm -f /tmp/fdml_ready.json
say "✓ /v1/ready/ model_loaded=true"

# 3) Transaction selector — the page needs it to populate the dropdown.
total=$(get "/v1/transactions/?limit=1" | jget total)
[ "$total" -ge 1 ] 2>/dev/null || fail "/v1/transactions/ total=$total"
say "✓ /v1/transactions/ ($total transacciones)"

# 4) Prediction with SHAP — the main call from the page.
predict=$(curl -sS --fail -X POST "$BASE_URL/v1/predict/" \
    -H 'Content-Type: application/json' \
    -d "{\"transaction_id\": $TRANSACTION_ID, \"include_shap\": true}")
proba=$(printf '%s' "$predict" | jget probability)
n_shap=$(printf '%s' "$predict" | python3 -c \
    'import json,sys; print(len(json.load(sys.stdin)["explanation"]["top_features"]))')
say "✓ /v1/predict/ proba=$proba shap_top=$n_shap"

# 5) Model leaderboard — the page shows the models/versions comparison.
n_models=$(get /v1/models/ | python3 -c \
    'import json,sys; print(len(json.load(sys.stdin)["models"]))')
say "✓ /v1/models/ ($n_models registrados)"

say "== Todo OK — lista para conectar la webpage =="

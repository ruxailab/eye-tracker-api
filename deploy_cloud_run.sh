#!/usr/bin/env bash
set -euo pipefail

# deploy_cloud_run.sh
# Build the Docker image, push to Artifact Registry, and deploy to Cloud Run.
# Usage: ./deploy_cloud_run.sh -p PROJECT_ID -s SERVICE_NAME [-r REGION] [-i IMAGE_NAME] [-a]

REGION=""
ALLOW_UNAUTH=""
IMAGE_NAME=""
PROJECT_ID=""
SERVICE_NAME=""
ENVIRONMENT="dev"
REPO_NAME="cloud-run-images" # Artifact Registry repository name

# Load defaults from an env file without overwriting flags already set.
load_env_defaults() {
  local file="$1"
  if [[ -f "$file" ]]; then
    # read non-empty, non-comment lines
    while IFS='=' read -r key val; do
      key=$(echo "$key" | xargs)
      val=$(echo "$val" | sed -e 's/^ *//;s/ *$//')
      if [[ -z "$key" ]]; then
        continue
      fi
      # only set variable if not already provided via flag
      if [[ -z "${!key:-}" ]]; then
        eval "$key=\"$val\""
      fi
    done < <(grep -vE '^\s*#|^\s*$' "$file")
    echo "Loaded defaults from $file"
  else
    echo "Env file not found: $file"
  fi
}

usage() {
  cat <<EOF
Usage: $0 -p PROJECT_ID -s SERVICE_NAME [-r REGION] [-i IMAGE_NAME] [-a]

Options:
  -p PROJECT_ID   GCP project ID (required unless env preset supplies it)
  -s SERVICE_NAME Cloud Run service name (required unless env preset supplies it)
  -r REGION       GCP region (default: us-central1)
  -i IMAGE_NAME   Image name (default: same as service)
  -a               Allow unauthenticated (public) access
  -e ENV          Environment preset: dev (default) or prod
  -h               Show this help
EOF
}

while getopts ":p:s:r:i:ae:h" opt; do
  case ${opt} in
    p) PROJECT_ID="$OPTARG" ;;
    s) SERVICE_NAME="$OPTARG" ;;
    r) REGION="$OPTARG" ;;
    i) IMAGE_NAME="$OPTARG" ;;
    a) ALLOW_UNAUTH=true ;;
    e) ENVIRONMENT="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing arg for -$OPTARG"; usage; exit 1 ;;
    *) echo "Invalid option -$OPTARG"; usage; exit 1 ;;
  esac
done

# Load env defaults from file corresponding to the selected environment
# Use `.env` for local/dev workflows and `.env.production` for production Cloud Run
if [[ "$ENVIRONMENT" == "dev" ]]; then
  load_env_defaults ".env"
else
  load_env_defaults ".env.production"
fi

# Fill defaults for any remaining unset values
REGION="${REGION:-us-central1}"
IMAGE_NAME="${IMAGE_NAME:-$SERVICE_NAME}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"
REPO_NAME="${REPO_NAME:-cloud-run-images}"

if [[ -z "$PROJECT_ID" || -z "$SERVICE_NAME" ]]; then
  echo "Error: PROJECT_ID and SERVICE_NAME must be set (flags, preset or environment vars)."
  usage
  exit 1
fi

TIMESTAMP=$(date -u +%Y%m%d%H%M%S)

# UPDATED: New Artifact Registry image naming convention
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TIMESTAMP}"

command -v gcloud >/dev/null 2>&1 || { echo "gcloud CLI not found. Install and authenticate first."; exit 1; }

# show what we're deploying
echo "Environment: ${ENVIRONMENT}"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPO_NAME}"
echo "Service: ${SERVICE_NAME}"
echo "Image: ${IMAGE_TAG}"
echo "Allow unauthenticated: ${ALLOW_UNAUTH}"

# UPDATED: Ensure the Artifact Registry repository exists before building
echo "Checking if Artifact Registry repository '${REPO_NAME}' exists..."
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Repository not found. Creating '${REPO_NAME}' in ${REGION}..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="Docker repository for Cloud Run images"
fi

echo "Building container and pushing to ${IMAGE_TAG}..."
# Cloud Build seamlessly pushes to Artifact Registry URLs
gcloud builds submit --tag "${IMAGE_TAG}" --project "${PROJECT_ID}" .

echo "Deploying to Cloud Run service ${SERVICE_NAME} in ${REGION}..."
# REMOVED: Deprecated --platform managed flag (it is now the standard default behavior)
DEPLOY_CMD=(gcloud run deploy "${SERVICE_NAME}" --image "${IMAGE_TAG}" --region "${REGION}" --project "${PROJECT_ID}")

if [ "$ALLOW_UNAUTH" = true ]; then
  DEPLOY_CMD+=(--allow-unauthenticated)
fi

"${DEPLOY_CMD[@]}"

echo "Deployment finished. Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format="value(status.url)"

echo "Done."

#!/bin/bash
# Co-Investigator Deployment Script for GCP Cloud Run
# Bash script for Linux/Mac

set -e

# Configuration
PROJECT_ID="${1:-queryquest-1771952465}"
REGION="${2:-us-central1}"
SERVICE_NAME="coinvestigator"
IMAGE_NAME="coinvestigator-streamlit"
SA_NAME="${SERVICE_NAME}-sa"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${CYAN}$1${NC}"; }
success() { echo -e "${GREEN}$1${NC}"; }
error() { echo -e "${RED}$1${NC}"; exit 1; }
warning() { echo -e "${YELLOW}$1${NC}"; }

# Banner
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}  Co-Investigator - Cloud Run Deployment Script${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check prerequisites
info "🔍 Checking prerequisites..."

if ! command -v gcloud &> /dev/null; then
    error "✗ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
fi
success "✓ gcloud CLI installed"

if ! command -v docker &> /dev/null; then
    error "✗ Docker not found. Install: https://docs.docker.com/get-docker/"
fi
success "✓ Docker installed"

# Set project
info "Setting GCP project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID" --quiet

# Enable required APIs
info "🔧 Enabling required GCP APIs..."
APIS=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "containerregistry.googleapis.com"
    "aiplatform.googleapis.com"
    "firestore.googleapis.com"
    "bigquery.googleapis.com"
    "storage.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo -n "  Enabling $api..."
    gcloud services enable "$api" --quiet 2>/dev/null
    success " ✓"
done

# Check/Create service account
info "🔐 Checking service account..."
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
    warning "Service account not found. Creating..."
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="Co-Investigator Service Account" \
        --quiet
    
    # Grant roles
    ROLES=(
        "roles/aiplatform.user"
        "roles/firestore.user"
        "roles/bigquery.user"
        "roles/storage.objectViewer"
    )
    
    for role in "${ROLES[@]}"; do
        echo -n "  Granting $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role" \
            --quiet 2>/dev/null
        success " ✓"
    done
else
    success "✓ Service account exists: $SA_EMAIL"
fi

# Build Docker image
info "🐳 Building Docker image..."
IMAGE_TAG="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

if ! docker build -t "$IMAGE_TAG" .; then
    error "✗ Docker build failed"
fi
success "✓ Docker image built: $IMAGE_TAG"

# Push to GCR
info "📤 Pushing image to Google Container Registry..."
gcloud auth configure-docker --quiet 2>/dev/null

if ! docker push "$IMAGE_TAG"; then
    error "✗ Docker push failed"
fi
success "✓ Image pushed to GCR"

# Deploy to Cloud Run
info "🚀 Deploying to Cloud Run..."

gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_QUOTA_PROJECT=${PROJECT_ID}" \
    --service-account "$SA_EMAIL" \
    --quiet

if [ $? -ne 0 ]; then
    error "✗ Deployment failed"
fi

success "✓ Deployment completed successfully!"
echo ""

# Get service URL
info "📍 Getting service URL..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format "value(status.url)")

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ DEPLOYMENT SUCCESSFUL${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Service URL: ${CYAN}${SERVICE_URL}${NC}"
echo ""
echo -e "${YELLOW}You can now access your application at the URL above.${NC}"
echo ""

# View logs command
echo -e "${NC}To view logs, run:${NC}"
echo -e "  ${NC}gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}'${NC}"
echo ""

# Test health endpoint
info "🏥 Testing health endpoint..."
HEALTH_URL="${SERVICE_URL}/_stcore/health"

if curl -Is "$HEALTH_URL" | head -n 1 | grep -q "200"; then
    success "✓ Health check passed"
else
    warning "⚠ Health check endpoint not responding yet (may take a minute to warm up)"
fi

echo ""
success "🎉 Deployment complete!"
echo ""

# Additional info
echo -e "${CYAN}Useful commands:${NC}"
echo "  Update service:    gcloud run deploy $SERVICE_NAME --image $IMAGE_TAG --region $REGION"
echo "  View logs:         gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME'"
echo "  Delete service:    gcloud run services delete $SERVICE_NAME --region $REGION"
echo "  Describe service:  gcloud run services describe $SERVICE_NAME --region $REGION"
echo ""

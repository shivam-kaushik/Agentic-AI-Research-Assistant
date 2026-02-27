# Co-Investigator Deployment Script for GCP Cloud Run
# PowerShell script for Windows

param(
    [string]$ProjectId = "queryquest-1771952465",
    [string]$Region = "us-central1",
    [string]$ServiceName = "coinvestigator",
    [string]$ImageName = "coinvestigator-streamlit",
    [switch]$SkipBuild,
    [switch]$SkipTests,
    [switch]$Verbose
)

# Color functions
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }

# Banner
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  Co-Investigator - Cloud Run Deployment Script" -ForegroundColor Magenta
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""

# Check prerequisites
Write-Info "🔍 Checking prerequisites..."

# Check gcloud
try {
    $gcloudVersion = gcloud version --format="value(version)"
    Write-Success "✓ gcloud CLI installed: $gcloudVersion"
} catch {
    Write-Error "✗ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Success "✓ Docker installed: $dockerVersion"
} catch {
    Write-Error "✗ Docker not found. Please install: https://docs.docker.com/get-docker/"
    exit 1
}

# Set project
Write-Info "Setting GCP project to $ProjectId..."
gcloud config set project $ProjectId

# Enable required APIs
Write-Info "🔧 Enabling required GCP APIs..."
$apis = @(
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "firestore.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com"
)

foreach ($api in $apis) {
    Write-Host "  Enabling $api..." -NoNewline
    gcloud services enable $api --quiet 2>$null
    Write-Success " ✓"
}

# Check/Create service account
Write-Info "🔐 Checking service account..."
$saEmail = "$ServiceName-sa@$ProjectId.iam.gserviceaccount.com"

$saExists = gcloud iam service-accounts describe $saEmail 2>$null
if (-not $saExists) {
    Write-Warning "Service account not found. Creating..."
    gcloud iam service-accounts create "$ServiceName-sa" `
        --display-name="Co-Investigator Service Account" `
        --quiet
    
    # Grant roles
    $roles = @(
        "roles/aiplatform.user",
        "roles/firestore.user",
        "roles/bigquery.user",
        "roles/storage.objectViewer"
    )
    
    foreach ($role in $roles) {
        Write-Host "  Granting $role..." -NoNewline
        gcloud projects add-iam-policy-binding $ProjectId `
            --member="serviceAccount:$saEmail" `
            --role="$role" `
            --quiet 2>$null
        Write-Success " ✓"
    }
} else {
    Write-Success "✓ Service account exists: $saEmail"
}

# Build Docker image
if (-not $SkipBuild) {
    Write-Info "🐳 Building Docker image..."
    $imageTag = "gcr.io/$ProjectId/${ImageName}:latest"
    
    docker build -t $imageTag .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "✗ Docker build failed"
        exit 1
    }
    Write-Success "✓ Docker image built: $imageTag"
    
    # Test locally (optional)
    if (-not $SkipTests) {
        Write-Info "🧪 Testing Docker image locally..."
        Write-Warning "Starting container on port 8501. Press Ctrl+C to stop after verification."
        Write-Host ""
        
        docker run -p 8501:8501 `
            -e GOOGLE_CLOUD_PROJECT=$ProjectId `
            -e GOOGLE_CLOUD_QUOTA_PROJECT=$ProjectId `
            $imageTag
        
        Write-Info "Test completed. Continuing with deployment..."
    }
    
    # Push to GCR
    Write-Info "📤 Pushing image to Google Container Registry..."
    docker push $imageTag
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "✗ Docker push failed"
        exit 1
    }
    Write-Success "✓ Image pushed to GCR"
} else {
    Write-Warning "⚠ Skipping build (using existing image)"
}

# Deploy to Cloud Run
Write-Info "🚀 Deploying to Cloud Run..."
$imageTag = "gcr.io/$ProjectId/${ImageName}:latest"

gcloud run deploy $ServiceName `
    --image $imageTag `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --max-instances 10 `
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_QUOTA_PROJECT=$ProjectId" `
    --service-account $saEmail `
    --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "✗ Deployment failed"
    exit 1
}

Write-Success "✓ Deployment completed successfully!"
Write-Host ""

# Get service URL
Write-Info "📍 Getting service URL..."
$serviceUrl = gcloud run services describe $ServiceName `
    --region $Region `
    --format "value(status.url)"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  ✓ DEPLOYMENT SUCCESSFUL" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Service URL: " -NoNewline
Write-Host $serviceUrl -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now access your application at the URL above." -ForegroundColor Yellow
Write-Host ""

# View logs command
Write-Host "To view logs, run:" -ForegroundColor Gray
Write-Host "  gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName'" -ForegroundColor DarkGray
Write-Host ""

# Test health endpoint
Write-Info "🏥 Testing health endpoint..."
try {
    $healthUrl = "$serviceUrl/_stcore/health"
    $response = Invoke-WebRequest -Uri $healthUrl -Method Head -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Success "✓ Health check passed"
    }
} catch {
    Write-Warning "⚠ Health check endpoint not responding yet (may take a minute to warm up)"
}

Write-Host ""
Write-Success "🎉 Deployment complete!"
Write-Host ""

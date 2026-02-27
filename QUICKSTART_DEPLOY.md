# Quick Start: Deploy to GCP Cloud Run

Deploy the Co-Investigator Streamlit app to Google Cloud Platform in **3 steps**.

---

## Prerequisites

- [x] Google Cloud Project with billing enabled
- [x] [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed
- [x] [Docker](https://docs.docker.com/get-docker/) installed
- [x] Authenticated with gcloud: `gcloud auth login`

---

## Option 1: One-Command Deployment (Recommended)

### Windows (PowerShell):
```powershell
.\deploy.ps1
```

### Linux/Mac (Bash):
```bash
chmod +x deploy.sh
./deploy.sh
```

**That's it!** The script will:
1. Enable required GCP APIs
2. Create service account with permissions
3. Build Docker image
4. Push to Google Container Registry
5. Deploy to Cloud Run
6. Return your service URL

---

## Option 2: Manual Deployment (3 Commands)

### Step 1: Enable APIs
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  containerregistry.googleapis.com aiplatform.googleapis.com \
  firestore.googleapis.com bigquery.googleapis.com storage.googleapis.com
```

### Step 2: Build & Push Image
```bash
# Set project
gcloud config set project queryquest-1771952465

# Build
docker build -t gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest .

# Configure Docker for GCR
gcloud auth configure-docker

# Push
docker push gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest
```

### Step 3: Deploy to Cloud Run
```bash
gcloud run deploy coinvestigator \
  --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=queryquest-1771952465,GOOGLE_CLOUD_QUOTA_PROJECT=queryquest-1771952465"
```

---

## Option 3: Cloud Build Automated Deployment

```bash
gcloud builds submit --config cloudbuild.yaml .
```

---

## Verify Deployment

```bash
# Get service URL
gcloud run services describe coinvestigator \
  --region us-central1 \
  --format 'value(status.url)'

# Test health endpoint
curl https://your-service-url.run.app/_stcore/health
```

---

## View Logs

```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=coinvestigator"
```

---

## Update Deployment

```bash
# Rebuild and redeploy
docker build -t gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest .
docker push gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest

gcloud run deploy coinvestigator \
  --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest \
  --region us-central1
```

---

## Troubleshooting

### Build Fails
```bash
# Check Dockerfile syntax
docker build --no-cache -t test-image .

# Check requirements.txt
pip install -r requirements.txt --dry-run
```

### Deployment Fails
```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# Verify service account
gcloud iam service-accounts describe coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com
```

### Application Errors
```bash
# Stream live logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=coinvestigator" --format=json

# Check environment variables
gcloud run services describe coinvestigator --region us-central1 --format yaml
```

---

## Cost Estimates

**Cloud Run Pricing (Pay-per-use):**
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Requests**: $0.40 per million requests
- **Free Tier**: 2 million requests/month, 360,000 GiB-seconds memory, 180,000 vCPU-seconds

**Example Monthly Cost** (100 users, 10 requests/day each):
- Requests: 30,000/month (free tier)
- Memory: ~50 GiB-seconds (free tier)
- CPU: ~200 vCPU-seconds (free tier)
- **Total: ~$0-5/month** (within free tier for light usage)

---

## Production Checklist

- [ ] Enable authentication if needed
- [ ] Configure custom domain
- [ ] Set up Cloud Monitoring alerts
- [ ] Enable Cloud Armor for DDoS protection
- [ ] Configure VPC connector for private resources
- [ ] Set up Cloud CDN for static assets
- [ ] Enable Cloud Trace for performance monitoring
- [ ] Configure backup strategy for Firestore
- [ ] Set up CI/CD pipeline with Cloud Build triggers
- [ ] Review and adjust resource limits

---

## Need Help?

- **Full Guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Cloud Run Docs**: https://cloud.google.com/run/docs
- **Streamlit on Cloud Run**: https://docs.streamlit.io/deploy/streamlit-community-cloud

---

## Quick Commands Reference

```bash
# Deploy
./deploy.sh

# Get URL
gcloud run services describe coinvestigator --region us-central1 --format 'value(status.url)'

# View logs
gcloud logging tail "resource.labels.service_name=coinvestigator"

# Update
gcloud run deploy coinvestigator --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest --region us-central1

# Rollback
gcloud run services update-traffic coinvestigator --region us-central1 --to-revisions=REVISION=100

# Delete
gcloud run services delete coinvestigator --region us-central1
```

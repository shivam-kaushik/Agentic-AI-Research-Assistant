# Deployment Guide for Co-Investigator on GCP Cloud Run

This guide explains how to deploy the Co-Investigator Streamlit application to Google Cloud Platform using Cloud Run.

---

## Prerequisites

1. **Google Cloud Project**: Ensure you have a GCP project created
   - Project ID: `queryquest-1771952465`
   - Billing enabled

2. **Required APIs**: Enable the following APIs:
   ```bash
   gcloud services enable \
     run.googleapis.com \
     cloudbuild.googleapis.com \
     containerregistry.googleapis.com \
     aiplatform.googleapis.com \
     firestore.googleapis.com \
     bigquery.googleapis.com \
     storage.googleapis.com
   ```

3. **Service Account**: Create a service account with proper permissions:
   ```bash
   gcloud iam service-accounts create coinvestigator-sa \
     --display-name="Co-Investigator Service Account"
   
   # Grant necessary roles
   gcloud projects add-iam-policy-binding queryquest-1771952465 \
     --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   
   gcloud projects add-iam-policy-binding queryquest-1771952465 \
     --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
     --role="roles/firestore.user"
   
   gcloud projects add-iam-policy-binding queryquest-1771952465 \
     --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
     --role="roles/bigquery.user"
   
   gcloud projects add-iam-policy-binding queryquest-1771952465 \
     --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
     --role="roles/storage.objectViewer"
   ```

4. **gcloud CLI**: Install and configure the gcloud CLI
   ```bash
   gcloud auth login
   gcloud config set project queryquest-1771952465
   gcloud auth configure-docker
   ```

---

## Deployment Methods

### Method 1: Manual Deployment (Recommended for First Time)

#### Step 1: Build Docker Image Locally
```bash
# Navigate to project directory
cd c:\Users\Public\Documents\My_Projects\BenchSci

# Build the image
docker build -t gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest .

# Test locally (optional)
docker run -p 8501:8501 \
  -e GOOGLE_CLOUD_PROJECT=queryquest-1771952465 \
  -e GOOGLE_CLOUD_QUOTA_PROJECT=queryquest-1771952465 \
  gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest
```

#### Step 2: Push to Google Container Registry
```bash
docker push gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest
```

#### Step 3: Deploy to Cloud Run
```bash
gcloud run deploy coinvestigator \
  --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=queryquest-1771952465,GOOGLE_CLOUD_QUOTA_PROJECT=queryquest-1771952465" \
  --service-account coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com
```

**Expected Output:**
```
Deploying container to Cloud Run service [coinvestigator] in project [queryquest-1771952465] region [us-central1]
✓ Deploying new service... Done.
  ✓ Creating Revision...
  ✓ Routing traffic...
Done.
Service [coinvestigator] revision [coinvestigator-00001-xyz] has been deployed and is serving 100 percent of traffic.
Service URL: https://coinvestigator-abc123-uc.a.run.app
```

#### Step 4: Verify Deployment
```bash
# Get service URL
gcloud run services describe coinvestigator --region us-central1 --format 'value(status.url)'

# Test the endpoint
curl -I https://coinvestigator-abc123-uc.a.run.app/_stcore/health
```

---

### Method 2: Automated Deployment with Cloud Build

#### Step 1: Setup Cloud Build Trigger
```bash
# Connect your repository (if using GitHub/GitLab)
# Or use the cloudbuild.yaml directly

# Submit build manually
gcloud builds submit --config cloudbuild.yaml .
```

#### Step 2: Create Automated Trigger (Optional)
```bash
# For GitHub repository
gcloud builds triggers create github \
  --name="coinvestigator-deploy" \
  --repo-name="BenchSci" \
  --repo-owner="your-github-username" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

#### Step 3: Monitor Build
```bash
# List recent builds
gcloud builds list --limit 5

# View build logs
gcloud builds log <BUILD_ID>
```

---

### Method 3: One-Command Deployment Script

Use the provided PowerShell script:

```powershell
# Run deployment script
.\deploy.ps1
```

Or use the bash script:
```bash
# Run deployment script
./deploy.sh
```

---

## Configuration

### Environment Variables

The following environment variables are set automatically:

| Variable | Value | Purpose |
|----------|-------|---------|
| `GOOGLE_CLOUD_PROJECT` | queryquest-1771952465 | GCP project ID |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | queryquest-1771952465 | Quota project for API calls |
| `STREAMLIT_SERVER_PORT` | 8501 | Port for Streamlit |
| `STREAMLIT_SERVER_ADDRESS` | 0.0.0.0 | Bind to all interfaces |
| `STREAMLIT_SERVER_HEADLESS` | true | Run without browser |

### Resource Limits

| Resource | Default | Adjustable |
|----------|---------|------------|
| Memory | 2Gi | Up to 32Gi |
| CPU | 2 | Up to 8 |
| Timeout | 300s | Up to 3600s |
| Max Instances | 10 | Up to 1000 |

To adjust resources:
```bash
gcloud run services update coinvestigator \
  --region us-central1 \
  --memory 4Gi \
  --cpu 4
```

---

## Post-Deployment

### 1. Set Up Custom Domain (Optional)
```bash
# Map custom domain
gcloud run domain-mappings create \
  --service coinvestigator \
  --domain coinvestigator.yourdomain.com \
  --region us-central1
```

### 2. Enable Authentication (Optional)
```bash
# Require authentication
gcloud run services update coinvestigator \
  --region us-central1 \
  --no-allow-unauthenticated
```

### 3. Configure Monitoring
```bash
# Enable Cloud Monitoring alerts
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="CoInvestigator High Error Rate" \
  --condition-threshold-value=5 \
  --condition-threshold-duration=60s
```

---

## Troubleshooting

### Issue: Container fails to start
**Solution:**
```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=coinvestigator" \
  --limit 50 \
  --format json

# Check container health
gcloud run services describe coinvestigator --region us-central1
```

### Issue: Authentication errors
**Solution:**
```bash
# Verify service account permissions
gcloud projects get-iam-policy queryquest-1771952465 \
  --flatten="bindings[].members" \
  --filter="bindings.members:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com"

# Re-grant permissions if needed
gcloud projects add-iam-policy-binding queryquest-1771952465 \
  --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### Issue: Out of memory
**Solution:**
```bash
# Increase memory allocation
gcloud run services update coinvestigator \
  --region us-central1 \
  --memory 4Gi
```

### Issue: API quota exceeded
**Solution:**
```bash
# Check quota usage
gcloud services quota list --service=aiplatform.googleapis.com

# Request quota increase in Cloud Console
```

---

## Updating the Deployment

### Quick Update (Same Image)
```bash
# Rebuild and redeploy
docker build -t gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest .
docker push gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest

# Trigger new revision
gcloud run deploy coinvestigator \
  --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest \
  --region us-central1
```

### Rollback to Previous Revision
```bash
# List revisions
gcloud run revisions list --service coinvestigator --region us-central1

# Rollback
gcloud run services update-traffic coinvestigator \
  --region us-central1 \
  --to-revisions=coinvestigator-00001-xyz=100
```

---

## Cost Optimization

### 1. Set Min Instances (Reduces Cold Starts)
```bash
gcloud run services update coinvestigator \
  --region us-central1 \
  --min-instances 0  # Or 1 for faster response
```

### 2. Enable CPU Throttling (Reduces Costs)
```bash
gcloud run services update coinvestigator \
  --region us-central1 \
  --cpu-throttling
```

### 3. Monitor Usage
```bash
# View request metrics
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"'
```

---

## Security Best Practices

1. **Use Secret Manager for Sensitive Data**
   ```bash
   # Create secret
   echo -n "your-api-key" | gcloud secrets create api-key --data-file=-
   
   # Grant access to service account
   gcloud secrets add-iam-policy-binding api-key \
     --member="serviceAccount:coinvestigator-sa@queryquest-1771952465.iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   
   # Update service to use secret
   gcloud run services update coinvestigator \
     --region us-central1 \
     --set-secrets="API_KEY=api-key:latest"
   ```

2. **Enable VPC Connector (Optional)**
   ```bash
   # Create VPC connector
   gcloud compute networks vpc-access connectors create coinvestigator-connector \
     --region us-central1 \
     --range 10.8.0.0/28
   
   # Attach to Cloud Run
   gcloud run services update coinvestigator \
     --region us-central1 \
     --vpc-connector coinvestigator-connector
   ```

3. **Enable Binary Authorization (Optional)**
   ```bash
   gcloud container binauthz policy import policy.yaml
   ```

---

## Maintenance

### Scheduled Updates
```bash
# Create Cloud Scheduler job to wake up service
gcloud scheduler jobs create http keep-warm \
  --schedule "*/5 * * * *" \
  --uri "https://coinvestigator-abc123-uc.a.run.app/_stcore/health" \
  --http-method GET
```

### Backup and Restore
```bash
# Export Firestore data
gcloud firestore export gs://queryquest-1771952465-backups/$(date +%Y%m%d)

# Import Firestore data
gcloud firestore import gs://queryquest-1771952465-backups/20260227
```

---

## Support

- **Cloud Run Documentation**: https://cloud.google.com/run/docs
- **Streamlit Documentation**: https://docs.streamlit.io
- **Project Repository**: https://github.com/your-org/BenchSci

---

## Quick Reference

```bash
# Deploy
gcloud run deploy coinvestigator --image gcr.io/queryquest-1771952465/coinvestigator-streamlit:latest --region us-central1

# View logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=coinvestigator"

# Get URL
gcloud run services describe coinvestigator --region us-central1 --format 'value(status.url)'

# Delete service
gcloud run services delete coinvestigator --region us-central1
```

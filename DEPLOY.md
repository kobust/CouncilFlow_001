# Deploy Attleboro Council Agent to Google Cloud Run

This guide walks through deploying the Streamlit app to **Google Cloud Run** (not GCS storage). Cloud Run runs your app as a container and gives you a public HTTPS URL.

## Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and logged in
- A Google Cloud project with [billing enabled](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled)
- Your **Gemini API key** and **Google Cloud Service Account** JSON (for Drive access)

---

## 1. Set up the project

```bash
# Create or select a project
gcloud projects create YOUR_PROJECT_ID   # or use existing
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

---

## 2. Store secrets in Secret Manager

The app needs `GEMINI_API_KEY` and the full **Service Account JSON** (for Drive). Store them as secrets:

```bash
# Create GEMINI_API_KEY secret (paste your key when prompted)
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=-

# Create GCP_SERVICE_ACCOUNT_JSON secret (use your SA JSON file path)
gcloud secrets create GCP_SERVICE_ACCOUNT_JSON --data-file=path/to/your-service-account.json
```

Use the same service account JSON you use locally (e.g. from `.streamlit/secrets.toml` or `test_sa.json`). The app expects the **entire JSON object** as a single string.

---

## 3. Grant Cloud Build access to deploy

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')

# Allow Cloud Build to deploy to Cloud Run
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

---

## 4. Deploy from source (recommended)

From the **project root** (where `Dockerfile` and `app.py` live):

```bash
gcloud run deploy council-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=secret:projects/YOUR_PROJECT_ID/secrets/GEMINI_API_KEY/versions/latest,GCP_SERVICE_ACCOUNT_JSON=secret:projects/YOUR_PROJECT_ID/secrets/GCP_SERVICE_ACCOUNT_JSON/versions/latest"
```

Replace `YOUR_PROJECT_ID` with your project ID. When prompted:

- **Service name**: accept `council-agent` or change it
- **Allow unauthenticated**: typically `y` so you can open the app in a browser

Cloud Run will **build the image** from the `Dockerfile` and **inject** the secrets as environment variables. The `docker_secrets.py` entrypoint writes `.streamlit/secrets.toml` from those env vars before starting Streamlit.

---

## 5. Deploy with a pre-built image (alternative)

If you prefer to build and push the image yourself:

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker YOUR_REGION-docker.pkg.dev

# Build and push
IMAGE=YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/cloud-run-source-deploy/council-agent
docker build -t $IMAGE .
docker push $IMAGE

# Deploy
gcloud run deploy council-agent \
  --image $IMAGE \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=secret:projects/YOUR_PROJECT_ID/secrets/GEMINI_API_KEY/versions/latest,GCP_SERVICE_ACCOUNT_JSON=secret:projects/YOUR_PROJECT_ID/secrets/GCP_SERVICE_ACCOUNT_JSON/versions/latest"
```

Use the same `--set-env-vars` and secret references as in step 4.

---

## 6. Grant the Cloud Run service access to secrets

The service identity that runs Cloud Run must be able to **read** the secrets:

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding GCP_SERVICE_ACCOUNT_JSON \
  --member="serviceAccount:${SA}" --role="roles/secretmanager.secretAccessor"
```

---

## 7. Open the app

After a successful deploy, `gcloud` prints the **service URL**. Open it in a browser. You should see the login page (admin / user from `config.yaml`).

---

## Important notes

### `config.yaml` (auth)

The image includes the **default** `config.yaml` (admin/user passwords). For production:

1. Regenerate password hashes:  
   `python -c "import streamlit_authenticator as stauth; print(stauth.Hasher.hash('YOUR_PASSWORD'))"`
2. Either **build your own image** with an updated `config.yaml`, or **override** it by mounting a different file (e.g. from Secret Manager or a GCS bucket) at `/app/config.yaml` when deploying.

### SQLite and `.rag_cache`

- **`council.db`**: Stored on the container filesystem. It is **ephemeral**—data is lost when the service scales to zero or restarts. For production, consider Cloud SQL or another managed DB and switching the app to use it.
- **`.rag_cache`**: RAG indexes are also ephemeral. The app will rebuild them on cold start, which can take several minutes for large libraries.

### Drive folder

The app uses a **hard-coded** Drive folder ID. Ensure the **service account** in `GCP_SERVICE_ACCOUNT_JSON` has **viewer access** to that folder (share the folder with the SA’s email).

### Region

Use a region close to your users and, if relevant, to other GCP resources (e.g. Secret Manager, Drive). Examples: `us-central1`, `europe-west1`.

---

## Troubleshooting

- **“GEMINI_API_KEY env var is required”**: Secrets are not reaching the container. Check IAM (step 6) and that `--set-env-vars` use the `secret:projects/...` form.
- **“config.yaml not found”**: The Dockerfile copies `config.yaml` into the image. If you use a custom build, ensure it’s included.
- **403 on Drive**: Share the Drive folder with the service account email from your JSON.
- **Slow first load**: RAG indexing runs on cold start; later requests are faster until the instance scales down.

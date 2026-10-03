# Moderation API — Render Deployment

## Quick Start

1. Copy `moderation_model.ftz` into this directory
2. Build and run:
   ```bash
   docker build -t moderation-api .
   docker run -p 8000:8000 moderation-api
   ```

## Render Free Deployment

1. Push this folder to a GitHub repository
2. Create a new Web Service on Render
3. Set Build Command: `pip install -r requirements.txt`
4. Set Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Upload `moderation_model.ftz` or include it in the repo

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./moderation_model.ftz` | Path to model file |
| `CONFIDENCE_WARN` | `0.5` | Threshold for warn action |
| `CONFIDENCE_DELETE` | `0.8` | Threshold for delete action |
| `CONFIDENCE_BLOCK` | `0.9` | Threshold for block action |

## API

### POST /moderate

```json
{
  "text": "user message"
}
```

Response:
```json
{
  "allowed": true,
  "label": "safe",
  "confidence": 0.98,
  "action": "allow",
  "reason_code": null
}
```

### GET /health

Returns `{"status": "ok", "model_loaded": true}`

## Architecture

- **Model** → classifies text as safe / abusive / severe_abusive
- **Policy** → decides action (allow / warn / delete / block / review)
- The model NEVER decides permanent bans. That is a policy/database decision.

# Image Captioning (CNN + LSTM) — Flickr8k

A complete image captioning system: a frozen ResNet-50 CNN encoder + an LSTM decoder,
trained on Flickr8k, served via FastAPI, and packaged in Docker.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the Flickr8k dataset as:
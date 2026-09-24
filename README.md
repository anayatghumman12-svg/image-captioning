
Images and captions are split **80% / 10% / 10%** into train/val/test **at the image level**
(all 5 captions of an image stay in the same split) using a fixed random seed for reproducibility.

## 🚀 Training Pipeline

```powershell
python -m src.feature_extraction   # cache CNN features once (Step 3)
python -m src.train                 # train the LSTM decoder (Step 4-5)
python -m src.evaluate              # BLEU scoring + examples (Step 6-7)
```

Training saves only the **best checkpoint** (lowest validation loss) — in this run, validation
loss stopped improving after epoch 7 and ticked up slightly at epoch 8, a clear sign of
overfitting, so epoch 7's weights were kept.

---

## 📈 Results

### BLEU Scores (test set, 810 images)

| Decoding | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Greedy | 0.601 | 0.422 | 0.283 | 0.189 |
| Beam Search (k=3) | *see reports/evaluation_report.json* | | | |

### Greedy vs. Beam Search

Greedy decoding picks the single highest-probability word at every step. Beam search instead
tracks the `k` (=3) most probable partial sequences at once and returns the best-scoring complete
sequence — it can recover from an early suboptimal word choice that greedy decoding gets stuck
with, at the cost of more computation per caption.

### Example Captions

| Image | Reference | Generated (Greedy) |
|---|---|---|
| `3449170348_34dac4a380.jpg` | *A girl dances on a sidewalk.* | a young girl wearing a pink shirt and pink pants is running through a grassy area |
| `3626964430_cb5c7e5acc.jpg` | *People playing cricket in the park.* | two men are playing soccer in a field |
| `1000268201_693b08cb0e.jpg` | *(girl climbing a tree)* | a little girl in a pink dress is climbing a tree |
| `3153067758_53f003b1df.jpg` | *A person holding a paper bag above a baggage cart.* | a man is sitting on a bed with a `<unk>` — ❌ failure case (object confusion) |

![Training vs Validation Loss](reports/loss_curve.png)

---

## 🌐 API Usage

Start the server locally:

```powershell
uvicorn app.main:app --reload --port 8000
```

Interactive docs: **http://127.0.0.1:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/caption` | Upload an image, receive a generated caption |

**Example request:**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/caption" -F "file=@data/Images/example.jpg"
```

**Example response:**
```json
{
  "caption": "a little girl in a pink dress is climbing a tree"
}
```

---

## 🐳 Docker

```powershell
docker build -t image-captioning-api .
docker run -d -p 8000:8000 --name captioning-container image-captioning-api
curl.exe -X POST "http://127.0.0.1:8000/caption" -F "file=@data/Images/example.jpg"
```

> `--host 0.0.0.0` in the Dockerfile's `CMD` is required — uvicorn's default host
> (`127.0.0.1`) only accepts connections from inside the container itself.

---

## 🧠 Design Decisions

- **Frozen ResNet-50 encoder:** Flickr8k (8,091 images) is far too small to fine-tune a
  25M-parameter CNN without overfitting. Freezing it also allows features to be **cached once**
  instead of recomputed every training epoch — a major speed-up.
- **LSTM decoder (init-hidden-state design):** the image feature is projected into the LSTM's
  initial `(h0, c0)` state; training uses **teacher forcing**, inference uses **greedy** or
  **beam search** decoding.
- **Best-checkpoint saving:** only the checkpoint with the lowest validation loss is kept,
  guarding against the model overfitting past its best generalization point.
- **Shared preprocessing:** the exact same image transform is used during feature caching
  (training) and inside the FastAPI service (serving), preventing a train/serve mismatch bug.

## 🔭 Limitations & Future Work

- Trained for only 8 epochs on CPU — more epochs / a GPU would likely improve BLEU scores further.
- No attention mechanism — the decoder conditions on a single pooled image vector rather than
  spatial feature maps, which limits its ability to focus on specific image regions.
- Vocabulary limited to words appearing ≥5 times in training captions; rare/unseen words become
  `<unk>`.
- Given more time, adding **visual attention** (Show, Attend and Tell-style) would likely be the
  single highest-impact improvement.

--- 📄 License

This project was built as part of the Nextbridge Summer Internship 2026, AI/ML Track — Task 4.
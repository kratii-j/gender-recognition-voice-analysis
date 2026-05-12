# Gender Recognition Using Voice

This project is now a real deep-learning voice-classification app:

- one FastAPI server
- one static frontend served by the same app
- live microphone recording and WAV upload
- a trained PyTorch multilayer perceptron (MLP)
- persistent prediction history in SQLite

## Dataset

The neural network is trained on the public `voice.csv` dataset:

- dataset name: `Gender Recognition by Voice and Speech Analysis`
- source used in this workspace: `data/datasets/voice.csv`
- samples: 3,168
- classes: balanced `male` and `female`
- features: 20 acoustic features plus one label column

## Train

```powershell
cd 'C:\Gender Recognition Using Voice'
C:\Users\welcome\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe train_model.py
```

Training writes:

- model weights to `artifacts/voice_gender_mlp.pt`
- evaluation metrics to `artifacts/voice_gender_metrics.json`

## Run

```powershell
cd 'C:\Gender Recognition Using Voice'
C:\Users\welcome\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- The app extracts acoustic features from uploaded WAV audio and feeds them into the trained MLP.
- If the trained model file is missing, the backend falls back to a simple heuristic baseline.
- All predictions are saved in `data/app.db`.
- Recorded audio files are saved in `data/audio/`.

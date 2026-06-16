# Engagement Level Estimation

Streamlit application for estimating engagement in real time using the camera. The system combines two signals:

- Emotional engagement: detects facial emotion with a model trained on FER2013 and groups it into engagement levels.
- Cognitive engagement: calculates visual attention from facial landmarks, blinking, gaze, and head position with MediaPipe.

At the end of a session, the app can generate a PDF report with emotional, cognitive, and system performance metrics.

## Execution

From the `FER2013` project root:

```bash
venv\Scripts\activate
streamlit run app/streamlit.py
```

The main screen asks for `User name` and `Session ID`, allows capture to be started/stopped, and generates a downloadable PDF once session data exists.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The application expects a `.env` file with at least the model path and architecture type:

```env
DEVICE_TYPE=auto
MODEL_TYPE=regularized
MODEL_PATH=artifacts/saved_models/regularized_best_model_20260412_224038.pth
CONFIDENCE_THRESHOLD=0.5
```

## Current Structure

```text
FER2013/
|-- app/
|   |-- streamlit.py
|   `-- generate_report.py
|-- models/
|   |-- emotion_model.py
|   |-- cognitive_eng.py
|   |-- train.py
|   |-- prepare_dataset.py
|   `-- detect_emo_cog.py
|-- artifacts/
|   |-- saved_models/
|   `-- logs/
|-- data/
|   `-- fer2013/
|-- config.py
|-- requirements.txt
`-- README.md
```

## Training

Training is not executed from the app. To train a new model, use:

```bash
python -m models.train
```

That script reads `DATA_DIR` and `CSV_FILE` from `.env`, defaults to `data/fer2013/fer2013.csv`, and saves models in `artifacts/saved_models/` along with logs in `artifacts/logs/`.

## PDF Report

The report includes:

- final emotional engagement level;
- final cognitive engagement level;
- dominant emotion;
- emotional changes per minute;
- engagement time-series curves;
- performance metrics: FPS, latency, CPU, RAM, and failures.

The PDF is generated in memory and delivered from Streamlit's `Download PDF` button.

## Maintenance Notes

- `venv/` is inside the local project, but it should not be considered source code.
- `data/fer2013/fer2013.csv` and `.pth` checkpoints can be large; they are usually not versioned.

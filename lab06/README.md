# Lab 06: Diabetes Predictor as a FastAPI Service

Serves the Logistic Regression model from Lab 5 as a JSON API, with a simple web page to try it.

## Files
| File | Purpose |
|---|---|
| `main.py` | FastAPI app (loads the model once at start-up) |
| `diabetes_logistic_model.pkl` | Trained model from Lab 5 |
| `index.html` | Front end that calls the API |
| `requirements.txt` | Dependencies (scikit-learn pinned to the training version) |
| `Dockerfile` | Container image |

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status |
| GET | `/metadata` | Model type, features and coefficients |
| POST | `/predict` | Returns the predicted label and probability |

Invalid input returns **422** (validation error). A failed prediction returns **400**, and a missing model returns **503**. Interactive docs are at `/docs`.

## Run locally
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```
Then open `http://localhost:8000/docs`, or open `index.html` in a browser.

## Run with Docker
```bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

## Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Glucose":150,"BloodPressure":70,"Insulin":150,"BMI":33,"DiabetesPedigreeFunction":0.6,"Age":45}'
```
Response:
```json
{"predicted_label": 1, "prediction": "Diabetic", "probability": 0.671}
```

## Stack
Python, FastAPI, Uvicorn, Pydantic, scikit-learn, Docker

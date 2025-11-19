Real-Time-Yoga-Pose-Detection-with-Correction-Analysis
------------------------------------------------------

Brief description
-----------------

This is a lightweight web application for recognizing and demonstrating five yoga poses using a trained model. The app provides a simple web UI (in `templates/` and `static/`) and a Flask backend (`app.py`) that loads a pre-trained model (`yoga_small_model.h5`) and sample input sequences (`train_sequences_small.npz`, `test_sequences_small.npz`).

Dataset Link
------------
https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

Dataset structure
--TRAIN
    downdog/
    goddess/
    plank/
    tree/
    warrior2/
--TEST
    downdog/
    goddess/
    plank/
    tree/
    warrior2/

Features
--------
- Web UI served from `templates/index.html` to try pose classification with example inputs in `static/examples/`.
- Flask backend in `app.py` which loads the Keras model and exposes routes for predictions and demo pages.
- Small dataset samples and model provided for quick local testing: `DATASET/`, `train_sequences_small.npz`, `test_sequences_small.npz`.

Repository structure (key files)
-------------------------------
- `app.py`: Main Flask application — run this to start the web server.
- `requirements.txt`: Python dependencies. Install with `pip install -r requirements.txt`.
- `yoga_small_model.h5`: Pre-trained Keras model used by the app.
- `yoga-5-classes.ipynb`: Notebook used for model training / exploration.
- `train_sequences_small.npz`, `test_sequences_small.npz`: small sample sequence data used for quick evaluations.
- `DATASET/`: full dataset folders (`TRAIN/`, `TEST/`) organized by pose.
- `templates/`: HTML templates (UI pages).
- `static/`: static assets and `examples/` used in the demo. This folder needs to be created under the main folder to store the example images.

Quick Start (local)
-------------------
1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```


2. Create (or activate) a Python environment (recommended Python 3.8+).

   PowerShell example:

   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

3. Run the app:

   ```powershell
   python app.py
   ```

4. Open a browser and visit `http://127.0.0.1:5000` (or the address printed by Flask).

Usage notes
-----------
- The server loads `yoga_small_model.h5` on startup. If you want to use a different model, update the path in `app.py`.
- The UI sends pose data or example selections to the backend for prediction. If your browser cannot reach the app, check the console where `app.py` is running for errors.
- To test with the provided small test sequences, see `static/examples/` and `test_sequences_small.npz` or use the notebook `yoga-5-classes.ipynb` for programmatic evaluation.

Model and data
--------------
- `yoga_small_model.h5` is a Keras HDF5 model included for demo purposes. The model expects input in the shape used by the training scripts (see the notebook for preprocessing details).
- The `DATASET/` folder contains per-pose subfolders under `TRAIN/` and `TEST/`. Use these for re-training or expanding the model.

Development
-----------
- To retrain or refine the model, open `yoga-5-classes.ipynb` or create a training script that loads images/sequences from `DATASET/` and saves a new model to the repo (update `app.py` to point to the new model).
- Add tests or a small script to validate that `yoga_small_model.h5` produces expected labels on `test_sequences_small.npz`.

Troubleshooting
---------------
- If `ImportError` appears on start, ensure `requirements.txt` is installed and the environment is active.
- If the model fails to load, confirm `yoga_small_model.h5` exists in the repo root and is not corrupted.
- If the UI doesn't display correctly, check that `templates/index.html` and the `static` folder are present and accessible.





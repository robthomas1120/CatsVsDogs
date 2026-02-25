# Cats vs Dogs Real-Time Detector

This project uses a pre-trained MobileNetV2 model to classify cats and dogs in real-time using your computer's webcam.

## Prerequisites

- Python 3.8 or higher
- A webcam

## Setup Instructions

Follow these steps to set up the project on your local machine.

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage your dependencies.

**On macOS/Linux:**
```bash
python3 -m venv venv
```

**On Windows:**
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

## How to Use

To start the real-time detector, run the following command:

```bash
python camera_app.py
```

### Controls:
- The app will open a window showing your webcam feed.
- It will display a label ("CAT" or "DOG") and the confidence percentage.
- Press **'q'** on your keyboard to quit the application.

## Project Structure

- `camera_app.py`: The main Python script for real-time detection.
- `cat_dog_model.keras`: The trained Keras model weights.
- `catsanddogs.ipynb`: Jupyter notebook used for training the model.
- `requirements.txt`: List of Python dependencies.

## Troubleshooting

### Using the Wrong Camera (macOS)
If the application is using your iPhone camera instead of your desktop camera (Continuity Camera), you can switch it manually in the code:

1. Open `camera_app.py` in a text editor.
2. Find the line: `CAMERA_INDEX = 0`.
3. Change it to: `CAMERA_INDEX = 1`.
4. Save the file and run it again.

If `1` doesn't work, you can try other numbers (like `2`, `3`) or temporarily disable **Continuity Camera** in your iPhone.

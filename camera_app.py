import cv2
import numpy as np
import tensorflow as tf
import os

# --- 1. MODEL RECONSTRUCTION ---
# This builds the "skeleton" manually to ensure compatibility with your Mac's Keras version
def build_and_load_model(model_path):
    print("Building model architecture...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None  # We load your specific trained weights next
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_weights(model_path)
        print("Model loaded successfully!")
    else:
        print(f"ERROR: Could not find {model_path} in the current folder.")
        exit()
        
    return model

# --- 2. INITIALIZATION ---
# Make sure your file is named exactly 'cat_dog_model.keras' or change this string:
MODEL_NAME = 'cat_dog_model.keras' 
model = build_and_load_model(MODEL_NAME)

# Open the webcam (0 is usually the built-in FaceTime camera)
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

# --- 3. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for a "mirror" effect (optional)
    frame = cv2.flip(frame, 1)

    # Pre-processing for MobileNetV2
    # Resize to 224x224 and scale pixels to [0, 1]
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Run Inference
    prediction = model.predict(img, verbose=0)[0][0]
    
    # Logic: < 0.5 is Cat (Class 0), > 0.5 is Dog (Class 1)
    if prediction > 0.5:
        label = f"DOG ({prediction*100:.1f}%)"
        color = (0, 255, 0) # Green
    else:
        label = f"CAT ({(1-prediction)*100:.1f}%)"
        color = (255, 0, 0) # Blue

    # Draw the label on the camera feed
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the resulting frame
    cv2.imshow('Cat vs Dog Real-Time Detector', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
# Add a small delay for macOS to close the window properly
cv2.waitKey(1)
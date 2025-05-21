import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load trained model ===
model = load_model("num_cnn_model_improved.h5")

# === Step 2: Folder with test images ===
folder_path = "numbers"
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

# === Step 3: Process each image ===
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
correct = 0
total = 0

for filename in sorted(image_files):
    img_path = os.path.join(folder_path, filename)

    # Load image, convert to grayscale and resize
    img = Image.open(img_path).convert("L").resize((28, 28))
    img = np.invert(img)

    # Normalize and reshape
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_label = np.argmax(prediction)

    # Try to extract the true label from the filename (e.g., "3.png")
    true_label = None
    try:
        true_label = int(os.path.splitext(filename)[0])
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
    except ValueError:
        pass  # skip files like "note.png"

    total += 1

    # Show image and prediction
    plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
    title = f"{filename} → Predicted: {predicted_label}"
    if true_label is not None:
        title += f" | True: {true_label} {'✅' if is_correct else '❌'}"
    plt.title(title)
    plt.axis("off")
    plt.show()

# === Step 4: Show overall accuracy ===
print(f"\n📊 Total tested: {total}")
print(f"✅ Correct predictions: {correct}")
if total > 0:
    print(f"🎯 Accuracy: {correct / total * 100:.2f}%")

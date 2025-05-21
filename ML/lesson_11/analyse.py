import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# === Load test data ===
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test_cat = to_categorical(y_test, 10)

# === Load trained model ===
model = load_model("num_cnn_model_improved.h5")

# === Optional: Show model summary ===
model.summary()

# === Evaluate on test set ===
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n🧪 Final Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# === Predict and plot examples ===
predictions = model.predict(x_test[:10])
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])} | True: {y_test[i]}")
    plt.axis("off")
    plt.show()

import tensorflow as tf
import numpy as np
import os

from QuantumLeNet_test import Quanv3x3LayerClass, quantum_layer
from tensorflow.keras.datasets import mnist

SELECTED_CLASSES = [0, 2, 3, 5, 8]
NUM_CLASSES = len(SELECTED_CLASSES)

def load_mnist_test_14x14_selected():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    mask = np.isin(y_test, SELECTED_CLASSES)
    x, y = x_test[mask], y_test[mask]

    x = x.astype("float32") / 255.0
    x = x[..., None]

    x = tf.nn.avg_pool2d(x,
                         ksize=2,
                         strides=2,
                         padding="VALID").numpy()

    remap = {c: i for i, c in enumerate(SELECTED_CLASSES)}
    y = np.vectorize(remap.get)(y)
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    return x, y

def evaluate_and_plot(model_path: str, save_dir: str) -> None:
    x_test, y_test = load_mnist_test_14x14_selected()
    
    custom_objects = {
        "Quanv3x3LayerClass": Quanv3x3LayerClass,
        "quantum_layer":      quantum_layer,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    preds   = model.predict(x_test, batch_size=128, verbose=0)
    y_true  = y_test.argmax(axis=1)
    y_pred  = preds.argmax(axis=1)

    per_class_acc = []
    for cls_idx in range(NUM_CLASSES):
        cls_mask = y_true == cls_idx
        acc = (y_pred[cls_mask] == y_true[cls_mask]).mean()
        per_class_acc.append(acc)

    overall_acc = (y_pred == y_true).mean()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(NUM_CLASSES), per_class_acc, tick_label=SELECTED_CLASSES)
    plt.ylim(0, 1)
    plt.xlabel("Digit class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-class accuracy (overall = {overall_acc:.4f})")

    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.2%}",
                 ha='center', va='bottom', fontsize=9)

    plt.axhline(overall_acc, color='red', linestyle='--', label=f"overall {overall_acc:.2%}")
    plt.legend()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    out_path = os.path.join(save_dir, 'all_test_set_accuracy.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    
    print(f"\nOverall accuracy on selected classes : {overall_acc:.4%}")
    for cls, acc in zip(SELECTED_CLASSES, per_class_acc):
        print(f"  class {cls} -> {acc:.4%}")
    print("plot saved to :", out_path)

if __name__=='__main__':
    evaluate_and_plot("FQLeNet_Data/fq_lenet_model.keras", "FQLeNet_Data")
    
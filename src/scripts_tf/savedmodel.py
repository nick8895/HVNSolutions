import tensorflow as tf

# Definiere die Modellarchitektur (muss der ursprünglichen entsprechen!)
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Pfad zu deinem Checkpoint-Verzeichnis
checkpoint_path = "/home/jovyan/data/models/tn_model"

# Modell erstellen und Gewichte aus Checkpoint laden
model = build_model()
model.load_weights(checkpoint_path)

# Optional: Überprüfen, ob die Gewichte korrekt geladen wurden
model.summary()

# Konvertiere und speichere das Modell im SavedModel-Format
saved_model_path = "/home/jovyan/data/models/saved_model_tn_model"
model.save(saved_model_path)

print(f"Modell erfolgreich im SavedModel-Format gespeichert unter: {saved_model_path}")

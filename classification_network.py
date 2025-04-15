import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


# Constants
IMG_SIZE = 64
NUM_CHANNELS = 3         # Zmienione z 1 (grayscale) na 3 (RGB)
NUM_CLASSES = 2          # Dla 2 klas: dog i cat
BATCH_SIZE = 64
EPOCHS = 100
LEARN_RATE = 0.000005

def load_dataset(base_path):
    print("Loading dataset from:", base_path)
    try:
        folder_contents = os.listdir(base_path)
        print("Contents of base folder:", folder_contents)
    except Exception as e:
        print("Error listing base folder:", e)
        return np.array([]), np.array([])
    
    images = []
    labels = []
    animal_map = {
         'dogs': 0, 'cats': 1
    }

    for animal in animal_map.keys():
        path = os.path.join(base_path, animal)
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist")
            continue

        print(f"Loading {animal} images from {path}")
        try:
            sub_contents = os.listdir(path)
            print(f"Contents of {path}:", sub_contents)
        except Exception as e:
            print(f"Error listing folder {path}: {e}")
            continue

        counter = 0
        limit = 3000  
        # Jeśli ostatni folder to 'test', ustaw mniejszy limit
        if os.path.basename(os.path.normpath(base_path)).lower() == 'test':
            limit = 950

        for img_path in os.listdir(path):
            counter += 1
            if counter >= limit:
                break

            full_path = os.path.join(path, img_path)
            # Wczytanie obrazu w trybie RGB
            try:
                img = tf.keras.preprocessing.image.load_img(
                    full_path,
                    color_mode='rgb',
                    target_size=(IMG_SIZE, IMG_SIZE)
                )
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                continue

            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(animal_map[animal])
    
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    """
    Normalize images and convert labels to one-hot encoding.
    """
    images = images.astype('float32') / 255.0
    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    return images, labels

def create_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    
    x = layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x) 

    x = layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(160, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE), name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(125, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE), name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(80, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(LEARN_RATE), name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

def get_hyperparameters(params):
    learning_rate = params.get('learning_rate', 0.002)
    dropout_rate = params.get('dropout_rate', 0.3)
    dense1_units = params.get('dense1_units', 160)
    dense2_units = params.get('dense2_units', 125)
    batch_size = params.get('batch_size', 64)
    epochs = params.get('epochs', 100)
    return learning_rate, dropout_rate, dense1_units, dense2_units, batch_size, epochs

def train_model(db_path, params=None):
    learning_rate, dropout_rate, dense1_units, dense2_units, batch_size, epochs = get_hyperparameters(params or {})
    print("Loading training data...")
    train_images, train_labels = load_dataset(os.path.join(db_path, 'train'))
    print(f"Loaded {len(train_images)} training images")

    print("\nLoading test data...")
    test_images, test_labels = load_dataset(os.path.join(db_path, 'test'))
    print(f"Loaded {len(test_images)} test images")

    # Preprocess data
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    print("\nCreating model...")
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.004, weight_decay=0.004),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=4,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    print("\nTraining model...")
    history = model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_images, val_labels),
        callbacks=callbacks
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nGenerating confusion matrix on test set:")
    plot_confusion_matrix(model, test_images, test_labels)

    return model, history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['Train', 'Validation'])

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

def predict_animal(model, image_path):
    """
    Predict animal type for a single image.
    """
    animal_map = {0: 'dog', 1: 'cat'}
    
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        color_mode='rgb',
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return animal_map[predicted_class], confidence

def plot_confusion_matrix(model, test_images, test_labels):
    """
    Create and display confusion matrix.
    """
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(true_labels, pred_labels)
    class_names = ['dog', 'cat']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    plt.show()

def test_sample_images(model_path, test_dir, samples_per_class=10):
    model = tf.keras.models.load_model(model_path)
    
    for animal in ['dogs', 'cats']:
        path = os.path.join(test_dir, animal)
        images_list = os.listdir(path)[:samples_per_class]
        
        plt.figure(figsize=(20, 4))
        for idx, img_name in enumerate(images_list):
            img_path = os.path.join(path, img_name)
            
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                color_mode='rgb',
                target_size=(IMG_SIZE, IMG_SIZE)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            predicted_animal = 'dog' if predicted_class == 0 else 'cat'
            
            plt.subplot(1, samples_per_class, idx + 1)
            display_img = cv2.imread(img_path)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            plt.imshow(display_img)
            color = 'green' if predicted_animal == animal else 'red'
            plt.title(f'Real: {animal}\nPred: {predicted_animal}\nConf: {confidence:.2f}', 
                      color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Upewnij się, że folder bazowy wskazuje na poprawną ścieżkę (np. './animals')
    db_path = './zwierzeta'
    model, history = train_model(db_path)

    plot_training_history(history)
    model.save('animal_recognition_model.keras')

    # Uzyskanie accuracy na zbiorze testowym
    test_images, test_labels = load_dataset(os.path.join(db_path, 'test'))
    test_images, test_labels = preprocess_data(test_images, test_labels)
    _, accuracy = model.evaluate(test_images, test_labels)
    
    # Zapis modelu z accuracy w nazwie
    model_name = f'model_{accuracy:.2f}.keras'
    model.save(model_name)
    print(f"Model saved as: {model_name}")

    # Test pre-trained model
    test_sample_images(model_name, os.path.join(db_path, 'test'))

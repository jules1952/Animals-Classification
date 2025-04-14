import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Parametry zgodne z Twoim modelem
IMG_SIZE = 64
NUM_CHANNELS = 3
model_path = "model_0.83.keras"  # Zmień na właściwą nazwę modelu
embedding_layer_name = "bn_2"  # Warstwa, z której wyciągasz embedding
base_dir = "teste"  # Folder, w którym masz pliki cat.4003.jpg, dog.4014.jpg, itp.

# Funkcja do wczytania i przetworzenia obrazu
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Model pomocniczy do wyciągania embeddingów
def create_embedding_model(model, layer_name):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Funkcja do obliczenia cosine similarity dla dwóch wektorów
def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

if __name__ == "__main__":
    # 1. Ładujemy wytrenowany model
    print(f"Ładowanie modelu z pliku: {model_path}")
    if not os.path.exists(model_path):
        print(f"Plik {model_path} nie istnieje.")
        exit(1)
    model = load_model(model_path)
    print("Model załadowany.")

    # 2. Tworzymy model pomocniczy
    print(f"Tworzenie modelu pomocniczego (warstwa: {embedding_layer_name})")
    embedding_model = create_embedding_model(model, embedding_layer_name)
    print("Model pomocniczy został utworzony.\n")

    # 3. Iterujemy po plikach w folderze base_dir
    if not os.path.isdir(base_dir):
        print(f"Folder {base_dir} nie istnieje.")
        exit(1)

    # Słownik, w którym będziemy przechowywać embeddingi według etykiet
    embeddings_by_label = {}

    file_list = os.listdir(base_dir)
    for filename in file_list:
        image_path = os.path.join(base_dir, filename)
        if not os.path.isfile(image_path):
            continue  # Pomijamy podfoldery, jeśli istnieją

        # 4. Na podstawie nazwy pliku wnioskujemy etykietę
        #    Tutaj przykład: jeśli nazwa zawiera 'dog' albo 'pies' -> dog,
        #    jeśli nazwa zawiera 'cat' -> cat, w przeciwnym wypadku pomijamy.
        #    Możesz dodać kolejne warunki, jeśli używasz innych oznaczeń w nazwach.
        lower_name = filename.lower()
        if "dog" in lower_name or "pies" in lower_name:
            label = "dog"
        elif "cat" in lower_name:
            label = "cat"
        else:
            # Jeśli plik nie pasuje do żadnej etykiety
            print(f"Nie rozpoznano etykiety w nazwie: {filename} – pomijam.")
            continue

        # 5. Wczytanie obrazu i wyciągnięcie embeddingu
        processed_img = load_and_preprocess_image(image_path)
        emb = embedding_model.predict(processed_img)

        # Zapisujemy embedding w słowniku
        if label not in embeddings_by_label:
            embeddings_by_label[label] = []
        embeddings_by_label[label].append(emb)

    # 6. Obliczamy cosine similarity w obrębie każdej etykiety
    for label, emb_list in embeddings_by_label.items():
        emb_array = np.vstack(emb_list)  # kształt (liczba_obrazów, wymiar_embeddingu)
        print(f"\n[INFO] Porównanie w obrębie klasy: {label} – {len(emb_list)} obraz(y).")
        for i in range(emb_array.shape[0]):
            for j in range(i+1, emb_array.shape[0]):
                sim = cosine_similarity(emb_array[i], emb_array[j])
                print(f"  {label} {i} vs {label} {j}: {sim:.4f}")

    # 7. Porównania między etykietami (np. dog vs cat) – tylko jeśli mamy co najmniej 2 etykiety
    labels = list(embeddings_by_label.keys())
    if len(labels) > 1:
        print("\n[INFO] Porównania między różnymi etykietami:")
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                label_i = labels[i]
                label_j = labels[j]
                emb_array_i = np.vstack(embeddings_by_label[label_i])
                emb_array_j = np.vstack(embeddings_by_label[label_j])
                for idx_i in range(emb_array_i.shape[0]):
                    for idx_j in range(emb_array_j.shape[0]):
                        sim = cosine_similarity(emb_array_i[idx_i], emb_array_j[idx_j])
                        print(f"  {label_i} {idx_i} vs {label_j} {idx_j}: {sim:.4f}")
    else:
        print("\n[INFO] Tylko jedna etykieta wykryta – brak porównań między klasami.")

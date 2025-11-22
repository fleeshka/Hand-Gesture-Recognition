import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import json
import os
from sklearn.utils.class_weight import compute_class_weight


class AdvancedGestureModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.history = None

    def create_advanced_model(self, input_shape, dropout_rate=0.5, l2_reg=0.001):
        """Создание улучшенной модели с регуляризацией"""
        model = keras.Sequential(
            [
                layers.Input(shape=input_shape),
                # Первый блок с сильной регуляризацией
                layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                # Второй блок
                layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate),
                # Третий блок
                layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                ),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate * 0.8),
                # Четвертый блок
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate * 0.6),
                # ВЫХОДНОЙ СЛОЙ ИСПРАВЛЕН - должно быть num_classes
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        return model

    def load_and_preprocess_data(self, csv_files, test_size=0.2, val_size=0.2):
        """Загрузка и предобработка данных из нескольких файлов"""
        all_data = []
        all_labels = []

        for file_path in csv_files:
            print(f"📂 Загружаем {file_path}...")
            df = pd.read_csv(file_path)

            # Извлекаем features
            feature_columns = (
                [f"x{i}" for i in range(21)]
                + [f"y{i}" for i in range(21)]
                + [f"z{i}" for i in range(21)]
            )
            landmarks = df[feature_columns].values
            is_right_hand = df["is_right_hand"].values.reshape(-1, 1)
            features_with_hand = np.hstack([landmarks, is_right_hand])
            labels = df["gesture"].values

            all_data.append(features_with_hand)
            all_labels.append(labels)

        # Объединяем данные
        X = np.vstack(all_data)
        y = np.hstack(all_labels)

        print(f"📊 Загружено {len(X)} samples")
        print(f"🎯 Классы: {np.unique(y)}")

        # Анализ распределения классов
        self._analyze_class_distribution(y)

        # Разделение на train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_temp,
        )

        print(f"📈 Разделение данных:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _analyze_class_distribution(self, y):
        """Анализ распределения классов"""
        unique, counts = np.unique(y, return_counts=True)
        print("📊 Распределение классов:")
        for cls, count in zip(unique, counts):
            percentage = count / len(y) * 100
            print(f"   {cls}: {count} samples ({percentage:.1f}%)")

    def prepare_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Подготовка данных: кодирование и нормализация"""
        # Кодируем метки
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Преобразуем в one-hot encoding
        y_train_categorical = keras.utils.to_categorical(
            y_train_encoded, self.num_classes
        )
        y_val_categorical = keras.utils.to_categorical(y_val_encoded, self.num_classes)
        y_test_categorical = keras.utils.to_categorical(
            y_test_encoded, self.num_classes
        )

        # Нормализуем features
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_val_normalized = self.scaler.transform(X_val)
        X_test_normalized = self.scaler.transform(X_test)

        return (
            X_train_normalized,
            X_val_normalized,
            X_test_normalized,
            y_train_categorical,
            y_val_categorical,
            y_test_categorical,
        )

    def compute_class_weights(self, y):
        """Вычисление весов классов для несбалансированных данных"""
        y_encoded = self.label_encoder.transform(y)
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_encoded), y=y_encoded
        )
        return dict(enumerate(class_weights))

    def train(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        epochs=100,
        batch_size=32,
        use_class_weights=True,
    ):
        """Обучение модели с улучшенными callback'ами"""

        # Подготавливаем данные
        X_train_proc, X_val_proc, _, y_train_proc, y_val_proc, _ = self.prepare_data(
            X_train, X_val, X_val, y_train, y_val, y_val
        )

        # Вычисляем веса классов если нужно
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train)
            print("⚖️  Используются веса классов:", class_weights)

        # Callbacks
        callbacks = [
            # Early Stopping
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            ),
            # ReduceLROnPlateau
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
            ),
            # ModelCheckpoint
            keras.callbacks.ModelCheckpoint(
                "best_gesture_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

        print("🚀 Начинаем обучение...")

        # Обучение
        history = self.model.fit(
            X_train_proc,
            y_train_proc,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_proc, y_val_proc),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            shuffle=True,
        )

        self.history = history
        return history

    def evaluate(self, X_test, y_test):
        """Оценка модели на тестовых данных"""
        X_test_proc = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_test_categorical = keras.utils.to_categorical(
            y_test_encoded, self.num_classes
        )

        results = self.model.evaluate(X_test_proc, y_test_categorical, verbose=0)

        print(f"📊 Результаты на тестовых данных:")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   Accuracy: {results[1]:.4f}")
        if len(results) > 2:
            print(f"   Precision: {results[2]:.4f}")
            print(f"   Recall: {results[3]:.4f}")

        return results

    def predict(self, X):
        """Предсказание для новых данных"""
        X_processed = self.scaler.transform(X)
        predictions = self.model.predict(X_processed, verbose=0)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    def predict_proba(self, X):
        """Вероятности предсказаний"""
        X_processed = self.scaler.transform(X)
        return self.model.predict(X_processed, verbose=0)

    def plot_training_history(self, save_path=None):
        """Визуализация процесса обучения"""
        if self.history is None:
            print("❌ История обучения не найдена")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history["loss"], label="Training Loss")
        ax2.plot(self.history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # Precision
        if "precision" in self.history.history:
            ax3.plot(self.history.history["precision"], label="Training Precision")
            ax3.plot(
                self.history.history["val_precision"], label="Validation Precision"
            )
            ax3.set_title("Model Precision")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Precision")
            ax3.legend()
            ax3.grid(True)

        # Recall
        if "recall" in self.history.history:
            ax4.plot(self.history.history["recall"], label="Training Recall")
            ax4.plot(self.history.history["val_recall"], label="Validation Recall")
            ax4.set_title("Model Recall")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Recall")
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Графики сохранены: {save_path}")

        plt.show()

    def confusion_matrix_analysis(self, X_test, y_test):
        """Анализ confusion matrix"""
        from sklearn.metrics import confusion_matrix, classification_report

        y_pred = self.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_
        )

        print("📈 Confusion Matrix Analysis:")
        print(cm)
        print("\n📊 Classification Report:")
        print(report)

        # Визуализация confusion matrix
        plt.figure(figsize=(10, 8))
        import seaborn as sns

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

        return cm, report

    def save_model(
        self, model_path="gesture_model.keras", metadata_path="model_metadata.json"
    ):
        """Сохранение модели и метаданных"""
        self.model.save(model_path)

        metadata = {
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "num_classes": self.num_classes,
            "input_shape": self.model.input_shape[1:],
            "feature_description": "63 landmarks (x0,y0,z0,...,x20,y20,z20) + is_right_hand",
            "model_type": "AdvancedGestureModel",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Сохраняем параметры scaler'а
        np.save("scaler_mean.npy", self.scaler.mean_)
        np.save("scaler_scale.npy", self.scaler.scale_)

        print(f"💾 Модель сохранена: {model_path}")
        print(f"📄 Метаданные сохранены: {metadata_path}")

    def load_model(
        self, model_path="gesture_model.keras", metadata_path="model_metadata.json"
    ):
        """Загрузка модели и метаданных"""
        self.model = keras.models.load_model(model_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.label_encoder.classes_ = np.array(metadata["label_encoder_classes"])
        self.num_classes = metadata["num_classes"]

        # Загружаем параметры scaler'а
        self.scaler.mean_ = np.load("scaler_mean.npy")
        self.scaler.scale_ = np.load("scaler_scale.npy")

        print(f"📥 Модель загружена: {model_path}")


def main():
    """Основная функция обучения"""
    # Ваши файлы с данными
    csv_files = ["mafia.csv", "if.csv", "don.csv"]

    # Определяем количество классов
    all_data = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    num_classes = len(combined_df["gesture"].unique())

    print(f"🎯 Обнаружено {num_classes} классов: {combined_df['gesture'].unique()}")

    # Создаем и обучаем модель
    gesture_model = AdvancedGestureModel(num_classes=num_classes)

    # Загружаем данные
    X_train, X_val, X_test, y_train, y_val, y_test = (
        gesture_model.load_and_preprocess_data(csv_files)
    )

    # Создаем модель
    print("🛠️ Создаем модель...")
    model = gesture_model.create_advanced_model(
        input_shape=(64,)
    )  # 63 landmarks + hand info

    # Показываем архитектуру
    model.summary()

    # Обучаем модель
    history = gesture_model.train(
        X_train, X_val, y_train, y_val, epochs=100, batch_size=32
    )

    # Оцениваем на тестовых данных
    print("\n🧪 Тестируем модель...")
    test_results = gesture_model.evaluate(X_test, y_test)

    # Анализ confusion matrix
    print("\n📊 Детальный анализ...")
    gesture_model.confusion_matrix_analysis(X_test, y_test)

    # Визуализация обучения
    gesture_model.plot_training_history("training_history.png")

    # Сохраняем модель
    gesture_model.save_model()

    # Проверяем на нескольких примерах
    print("\n🔍 Проверка на случайных примерах:")
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    correct = 0

    for idx in sample_indices:
        sample = X_test[idx : idx + 1]
        true_label = y_test[idx]
        pred_label = gesture_model.predict(sample)[0]
        confidence = np.max(gesture_model.predict_proba(sample))
        is_correct = true_label == pred_label
        correct += is_correct

        print(
            f"   True: {true_label:12} Pred: {pred_label:12} "
            f"Conf: {confidence:.3f} {'✓' if is_correct else '✗'}"
        )

    print(f"   Sample accuracy: {correct/len(sample_indices):.2f}")


if __name__ == "__main__":
    main()

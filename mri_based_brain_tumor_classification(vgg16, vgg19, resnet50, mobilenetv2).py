# -*- coding: utf-8 -*-


# # !pip install tensorflow
# print(f"TensorFlow Version: {tf.__version__}")


import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 15



# Dataset path
dataset_path = 'Dataset'
train_dir = os.path.join(dataset_path, 'Training')
test_dir = os.path.join(dataset_path, 'Testing')

print(f"Loading data from: {os.path.abspath(dataset_path)}")



# Data load an d preprocess

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, labels='inferred', label_mode='int', image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=123, shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, labels='inferred', label_mode='int',image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=123, shuffle=False
)

CLASS_NAMES = train_ds.class_names
print(f"\nTarget Classes: {CLASS_NAMES}")





# Prefetch and cache for better and fast performance
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)



# callbacks for monitoring
def get_callbacks(model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    filepath = f'{model_name}_best_model.keras'
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)

    return [early_stopping, checkpoint, lr_reducer]






#creating functon to create model
def create_transfer_model(base_model_class, input_shape, num_classes, optimizer, activation_fn='relu', dropout_rate=0.5, l2_reg=1e-4, trainable_layers=0):


    inputs = keras.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1./255)(x)

    #loading based model
    base_model = base_model_class(weights='imagenet', include_top=False, input_tensor=x)

    # Freezing / unfreezing policy
    base_model.trainable = True
    if trainable_layers > 0:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        print(f"Base model set to train last {trainable_layers} layers.")
    else:
        base_model.trainable = False
        print("Base model feature extractor frozen.")

    # Classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, kernel_regularizer=l2(l2_reg))(x)
    x = layers.Activation(activation_fn)(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model




#Evaluation Function

def evaluate_model(model, test_dataset, model_name, class_names):
    print(f"\nEvaluating {model_name}")

    # Prediction
    y_true = []
    y_pred_classes = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        predicted_classes = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred_classes.extend(predicted_classes)

    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

    # Metrics extraction
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    overall_accuracy = report['accuracy']


    print("\n" + "-"*50)
    print(f"| Model: {model_name.upper():<41}|")
    print("-"*50)
    print(f"| Overall Accuracy: {overall_accuracy:.4f}")
    print(f"| Macro-Averaged Recall: {macro_recall:.4f}")
    print(f"| Macro-Averaged F1-Score: {macro_f1:.4f} ")
    print("-"*50)


    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    return {
        'model_name': model_name,
        'accuracy': overall_accuracy,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'report': report
    }



INPUT_SHAPE = IMAGE_SIZE + (3,)
results = []
comparison_table = []



# MD 1: VGG16 Baseline (Adam, ReLU, L2)
model_name_1 = "VGG16_Adam_L2"
print(f"\n\n\nStarting {model_name_1}")

model_1 = create_transfer_model(VGG16, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-3), l2_reg=1e-4, trainable_layers=0)

model_1.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_1), verbose=2)


res_1 = evaluate_model(model_1, test_ds, model_name_1, CLASS_NAMES)
comparison_table.append(res_1)



#MD 2: VGG16 Fine-Tuning (Adam, ReLU, L2, Fine-Tuning LR)
model_name_2 = "VGG16_FineTune_LR1e-5"
print(f"\n\n\nStarting {model_name_2}")

model_2 = create_transfer_model(VGG16, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-5), l2_reg=1e-4, trainable_layers=4 )

model_2.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_2), verbose=2)


res_2 = evaluate_model(model_2, test_ds, model_name_2, CLASS_NAMES)
comparison_table.append(res_2)



#MD 3: MobileNetV2 (Adam, ReLU, Dropout 0.4)
model_name_3 = "MobileNetV2_Adam_Drop0.4"
print(f"\n\n\nStarting {model_name_3}...")

model_3 = create_transfer_model( MobileNetV2, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-3), dropout_rate=0.4, trainable_layers=0)
model_3.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_3), verbose=2)
res_3 = evaluate_model(model_3, test_ds, model_name_3, CLASS_NAMES)
comparison_table.append(res_3)



#MD 4: ResNet50 Baseline (Adam, ReLU, L2)
model_name_4 = "ResNet50_Adam_L2"
print(f"\n\n\nStarting {model_name_4}")

model_4 = create_transfer_model(ResNet50, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-3), l2_reg=1e-4, trainable_layers=0)

model_4.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_4), verbose=2)

res_4 = evaluate_model(model_4, test_ds, model_name_4, CLASS_NAMES)
comparison_table.append(res_4)

# MD 5: ResNet50 with SGD Optimizer (Optimizer Change)
model_name_5 = "ResNet50_SGD_Optimizer"
print(f"\n\n\nStarting {model_name_5}")

model_5 = create_transfer_model(ResNet50, INPUT_SHAPE, NUM_CLASSES, optimizers.SGD(learning_rate=1e-3, momentum=0.9), l2_reg=1e-4, trainable_layers=0)

model_5.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_5), verbose=2)
res_5 = evaluate_model(model_5, test_ds, model_name_5, CLASS_NAMES)
comparison_table.append(res_5)





# MD 6: ResNet50 with GELU Activation (Activation Change)
model_name_6 = "ResNet50_GELU_Activation"
print(f"\n\n\nStarting {model_name_6}")

model_6 = create_transfer_model(ResNet50, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-3), activation_fn='gelu',l2_reg=1e-4, trainable_layers=0)

model_6.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_6), verbose=2)

res_6 = evaluate_model(model_6, test_ds, model_name_6, CLASS_NAMES)
comparison_table.append(res_6)

#MD 7: MobileNetV2 with High Dropout (Regularization Change)
model_name_7 = "MobileNetV2_HighDrop0.6"
print(f"\n\n\nStarting {model_name_7}")

model_7 = create_transfer_model(MobileNetV2, INPUT_SHAPE, NUM_CLASSES,optimizers.Adam(learning_rate=1e-3), dropout_rate=0.6,trainable_layers=0 )
model_7.fit( train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_7), verbose=2 )
res_7 = evaluate_model(model_7, test_ds, model_name_7, CLASS_NAMES)
comparison_table.append(res_7)

# # MD 8: VGG19 (Alternative Depth/Architecture)
# model_name_8 = "VGG19_Adam_L2"
# print(f"Starting {model_name_8}")

# model_8 = create_transfer_model(VGG19, INPUT_SHAPE, NUM_CLASSES, optimizers.Adam(learning_rate=1e-3), l2_reg=1e-4, trainable_layers=0)

# model_8.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks(model_name_8), verbose=2)
# res_8 = evaluate_model(model_8, test_ds, model_name_8, CLASS_NAMES)
# comparison_table.append(res_8)



print("\n\nFINAL MODEL COMPARISON TABLE \n")

# Convert results to pandas DataFrame for display
results_df = pd.DataFrame([
    {'Setup': r['model_name'],
     'Accuracy': f"{r['accuracy']:.4f}",
     'Macro-Recall (CRITICAL)': f"{r['macro_recall']:.4f}",
     'Macro-F1': f"{r['macro_f1']:.4f}",
     'Model Type': r['model_name'].split('_')[0]} for r in comparison_table
])

print(results_df.to_markdown(index=False))

#best model based on Macro-Recall
best_model = results_df.loc[results_df['Macro-Recall (CRITICAL)'].astype(float).idxmax()]

print("\n\nBEST MODEL (Highest Macro-Recall) \n")

print(best_model.to_markdown(index=True))










import cv2
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import pandas as pd
import keras
from keras.applications import MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("🍇 TREINAMENTO MULTICLASSE: UVA vs UVA_RUIM vs NÃO-UVA vs NADA")
print("=" * 60)
print("Nova estratégia:")
print("  - Treino com 4 classes")
print("  - UVA (0), UVA_RUIM (1), NÃO-UVA (2), NADA (3)")
print("  - Leitura direta de resize_teste/uva, /uva_ruim, /nao_uva e /nada")
print("  - Sem pré-tratamento no treino (imagem crua lida do disco)")
print("=" * 60 + "\n")

BASE_VIDEO_DIR = 'resize_teste'
CLASS_NAMES = ['UVA', 'UVA_RUIM', 'NAO_UVA', 'NADA']
CLASS_FOLDERS = ['uva', 'uva_ruim', 'nao_uva', 'nada']
INPUT_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)

def criar_modelo_mobilenet_multiclasse():
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(INPUT_SIZE[0], INPUT_SIZE[1], 3))
    x = inputs
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def getImagemComId():
    objetos = []
    ids = []
    extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp')

    for class_id, (class_name, class_folder) in enumerate(zip(CLASS_NAMES, CLASS_FOLDERS)):
        pasta_classe = os.path.join(BASE_VIDEO_DIR, class_folder)
        if not os.path.exists(pasta_classe):
            raise FileNotFoundError(f"Pasta '{pasta_classe}' não encontrada!")

        caminhos_imagens = [
            os.path.join(pasta_classe, f)
            for f in os.listdir(pasta_classe)
            if f.lower().endswith(extensoes_validas)
        ]
        print(f"✅ Encontradas {len(caminhos_imagens)} imagens de {class_name}")

        for caminhoImagem in caminhos_imagens:
            imagemobjeto = cv2.imread(caminhoImagem)
            if imagemobjeto is None:
                continue

            # Apenas valida formato esperado para evitar erro de shape no modelo.
            h, w = imagemobjeto.shape[:2]
            if (w, h) != INPUT_SIZE:
                print(f"⚠️ Ignorando {os.path.basename(caminhoImagem)}: esperado {INPUT_SIZE[0]}x{INPUT_SIZE[1]}, recebido {w}x{h}")
                continue

            if len(imagemobjeto.shape) != 3 or imagemobjeto.shape[2] != 3:
                print(f"⚠️ Ignorando {os.path.basename(caminhoImagem)}: imagem sem 3 canais")
                continue

            ids.append(class_id)
            objetos.append(imagemobjeto)
    
    return np.array(ids), np.array(objetos)

print("\n📂 Carregando imagens...\n")
ids, objetos = getImagemComId()

labels = ids

print(f"\n📊 ESTATÍSTICAS DO DATASET:")
print(f"  Total de imagens: {len(objetos)}")
for class_id, class_name in enumerate(CLASS_NAMES):
    qtd = np.sum(labels == class_id)
    pct = (qtd / len(labels) * 100) if len(labels) > 0 else 0
    print(f"  {class_name} ({class_id}):{' ' * (8 - len(str(class_id)))} {qtd} ({pct:.1f}%)")

if len(objetos) < 20:
    raise ValueError("Dataset muito pequeno! Adicione mais imagens.")

for class_id, class_name in enumerate(CLASS_NAMES):
    if np.sum(labels == class_id) == 0:
        raise ValueError(f"Faltam imagens da classe {class_name} ({class_id})!")

labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val = train_test_split(
    objetos, labels_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)

print(f"\n📚 DIVISÃO DOS DADOS:")
print(f"  Treino:     {len(X_train)} imagens")
print(f"  Validação:  {len(X_val)} imagens")

classes = np.unique(labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=labels
)
class_weight_dict = dict(zip(classes, class_weights))

print(f"\n⚖️  PESOS DAS CLASSES:")
for class_id, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name} ({class_id}): {class_weight_dict[class_id]:.4f}")

print("\n🏗️  Criando modelo MobileNet MULTICLASSE...")
mobilenet_uva = criar_modelo_mobilenet_multiclasse()

mobilenet_uva.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n📐 ARQUITETURA DO MODELO:")
print("=" * 60)
mobilenet_uva.summary()
print("=" * 60)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 INICIANDO TREINAMENTO...")
print("=" * 60)

history = mobilenet_uva.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

print("=" * 60)

os.makedirs('Classificadores_de_Treino', exist_ok=True)

mobilenet_uva.save('Classificadores_de_Treino/classificadorMobileNet.keras')
print("\n💾 Modelo salvo: Classificadores_de_Treino/classificadorMobileNet.keras")

historico_df = pd.DataFrame(history.history)
historico_df.to_csv('Classificadores_de_Treino/historico_treinamento_binario.csv', index=False)
print("💾 Histórico salvo: Classificadores_de_Treino/historico_treinamento_binario.csv")

print("\n" + "=" * 60)
print("📊 AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
print("=" * 60)

results = mobilenet_uva.evaluate(X_val, y_val, verbose=0)

print(f"Loss:      {results[0]:.4f}")
print(f"Accuracy:  {results[1] * 100:.2f}%")
print(f"Precision: {results[2] * 100:.2f}%")
print(f"Recall:    {results[3] * 100:.2f}%")

if results[2] + results[3] > 0:
    f1_score = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"F1-Score:  {f1_score * 100:.2f}%")

print("=" * 60)
print("\n✅ TREINAMENTO CONCLUÍDO!")
print("\n📋 RESUMO:")
print("  • Modelo MULTICLASSE treinado")
print("  • Label 0 = UVA")
print("  • Label 1 = UVA_RUIM")
print("  • Label 2 = NÃO-UVA")
print("  • Label 3 = NADA")
print("  • Use reconhecedorV8_deteccao.py para reconhecimento")
print("=" * 60)

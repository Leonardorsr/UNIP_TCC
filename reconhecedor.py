import cv2
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import keras
import serial
import time
import threading

# ============================================================
# CONFIGURAÇÃO ARDUINO/SERVO
# ============================================================
try:
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("✅ Arduino conectado com sucesso!")
    arduino_conectado = True
except:
    print("⚠️  AVISO: Arduino não conectado! Continuando sem controle do servo...")
    arduino_conectado = False

# ============================================================
# VARIÁVEIS DE CONTROLE DO SERVO
# ============================================================
ultimo_comando_enviado = None
tempo_ultimo_comando = 0
intervalo_minimo_comandos = 3.0
contador_comandos_uva = 0
contador_comandos_uva_ruim = 0
contador_comandos_nao_uva = 0
contador_deteccoes_nada = 0
delay_centralizacao = 2.0

# ============================================================
# MÉTRICAS DE TEMPO
# ============================================================
tempo_ultimo_reconhecimento = None
soma_intervalos_reconhecimento = 0.0
contador_intervalos_reconhecimento = 0
ultima_classe_reconhecida = None

soma_latencia_envio_comando = 0.0
contador_envios_comando = 0

# ============================================================
# VARIÁVEIS DE VERIFICAÇÃO TRIPLA DE UVA
# ============================================================
contador_deteccoes_uva = 0
tempo_ultima_deteccao_uva = 0
intervalo_verificacao_uva = 0.3
deteccoes_necessarias_uva = 3

# ============================================================
# PARÂMETROS DE PRÉ-PROCESSAMENTO (ALINHADOS AO SCRIPT DE PRÉ)
# ============================================================
TAMANHO_PADRAO = 224
PADDING_COLOR = (127, 127, 127)

# ============================================================
# CONFIGURAÇÕES DE CÂMERA (ALINHADAS COM COLETA)
# ============================================================
CAMERA_ID = 0
CAMERA_BACKEND = cv2.CAP_DSHOW
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
TARGET_FPS = 30


def normalizar_imagem(imagem, tamanho=TAMANHO_PADRAO):
    h, w = imagem.shape[:2]
    if h == 0 or w == 0:
        return None

    escala = min(tamanho / w, tamanho / h)
    novo_w = max(1, int(round(w * escala)))
    novo_h = max(1, int(round(h * escala)))

    imagem_redimensionada = cv2.resize(imagem, (novo_w, novo_h), interpolation=cv2.INTER_AREA)

    pad_top = (tamanho - novo_h) // 2
    pad_bottom = tamanho - novo_h - pad_top
    pad_left = (tamanho - novo_w) // 2
    pad_right = tamanho - novo_w - pad_left

    imagem_final = cv2.copyMakeBorder(
        imagem_redimensionada,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=PADDING_COLOR,
    )

    return imagem_final

# ============================================================
# LOCK SERIAL
# ============================================================
serial_lock = threading.Lock()

def enviar_comando_arduino(comando):
    if arduino_conectado:
        with serial_lock:
            try:
                arduino.write(comando.encode() if isinstance(comando, str) else comando)
                arduino.flush()
                return True
            except Exception as e:
                print(f"❌ Erro ao enviar comando '{comando}': {e}")
                return False
    return False

def enviar_comando_arduino_cronometrado(comando, tempo_reconhecimento):
    global soma_latencia_envio_comando, contador_envios_comando

    sucesso = enviar_comando_arduino(comando)
    if sucesso and tempo_reconhecimento is not None:
        latencia_total = time.time() - tempo_reconhecimento
        soma_latencia_envio_comando += latencia_total
        contador_envios_comando += 1

        media_latencia = soma_latencia_envio_comando / contador_envios_comando
        print(
            f"    ⏱️  Latência reconhecimento→comando: {latencia_total*1000:.1f} ms "
            f"(média: {media_latencia*1000:.1f} ms)"
        )

    return sucesso

def enviar_centralizacao_atrasada():
    time.sleep(delay_centralizacao)
    if enviar_comando_arduino(b'C'):
        print(f"    ↩️  COMANDO 'C' ENVIADO (centralizar após {delay_centralizacao}s)")

# ============================================================
# PREPARAÇÃO DA CÂMERA E MODELO
# ============================================================
camera = cv2.VideoCapture(CAMERA_ID, CAMERA_BACKEND)

if not camera.isOpened():
    raise RuntimeError("❌ Não foi possível abrir a câmera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)

for _ in range(20):
    camera.read()

print("=" * 60)
print(f"Resolução configurada: {actual_width}x{actual_height}")
print(f"FPS real: {actual_fps:.2f}")
if actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
    print("✅ Resolução 224x224 ativada!")
else:
    print(f"⚠️  224x224 não disponível, usando {actual_width}x{actual_height}")
print("=" * 60)

print("📦 Carregando modelo MobileNet...")
reconhecedorMobileNet = keras.models.load_model('Classificadores_de_Treino/classificadorMobileNet.keras')
print("✅ Modelo carregado com sucesso!")

# ============================================================
# CONFIGURAÇÕES
# ============================================================
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

threshold_confianca_minima = 0.60

classes_nomes = {
    0: "UVA",
    1: "UVA_RUIM",
    2: "NAO-UVA",
    3: "NADA"
}

classes_cores = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 0, 255),
    3: (255, 200, 0)
}

# ============================================================
# INÍCIO DO RECONHECIMENTO
# ============================================================
print("\n" + "=" * 60)
print("🎥 INICIANDO RECONHECIMENTO (IMAGEM INTEIRA + RESIZE)")
print("=" * 60)
print("Pipeline:")
print("  1️⃣  Ler frame inteiro da câmera")
print("  2️⃣  Aplicar resize com padding para 224x224")
print("  3️⃣  Classificar: UVA, UVA_RUIM, NÃO-UVA ou NADA")
print("  4️⃣  Regra NADA: não enviar comando ao Arduino")
print("\nControles: Q = sair")
print("=" * 60)

# ============================================================
# LOOP PRINCIPAL
# ============================================================
print("\n🎬 Loop de reconhecimento iniciado!\n")

while True:
    conectado, imagem = camera.read()
    
    if not conectado:
        print("❌ Erro ao capturar imagem da câmera!")
        break
    
    h, w = imagem.shape[:2]
    imagem_display = imagem.copy()

    imagem_processada = normalizar_imagem(imagem, TAMANHO_PADRAO)
    if imagem_processada is None:
        cv2.putText(imagem_display, "Falha no resize/padding", (20, 40), font, 1.0, (0, 0, 255), 2)
        cv2.imshow("Reconhecedor de Uvas - Detecção + Classificação", imagem_display)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("\n🛑 Encerrando...")
            break
        continue

    roi_array = np.expand_dims(imagem_processada, axis=0)

    predicoes = reconhecedorMobileNet.predict(roi_array, verbose=0)[0]
    if len(predicoes) != 4:
        cv2.putText(imagem_display, "Saida do modelo invalida (esperado 4 classes)",
                    (20, 140), font, 0.8, (0, 0, 255), 2)
    else:
        id_resultado = int(np.argmax(predicoes))
        confianca_maxima = float(predicoes[id_resultado] * 100.0)

        prob_uva = float(predicoes[0] * 100.0)
        prob_uva_ruim = float(predicoes[1] * 100.0)
        prob_nao_uva = float(predicoes[2] * 100.0)
        prob_nada = float(predicoes[3] * 100.0)

        cv2.putText(imagem_display, f"Prob UVA: {prob_uva:.1f}%", (20, 140), font, 0.8, (0, 255, 0), 1)
        cv2.putText(imagem_display, f"Prob UVA_RUIM: {prob_uva_ruim:.1f}%", (20, 165), font, 0.8, (0, 165, 255), 1)
        cv2.putText(imagem_display, f"Prob NAO-UVA: {prob_nao_uva:.1f}%", (20, 190), font, 0.8, (0, 0, 255), 1)
        cv2.putText(imagem_display, f"Prob NADA: {prob_nada:.1f}%", (20, 215), font, 0.8, (255, 200, 0), 1)

        if predicoes[id_resultado] >= threshold_confianca_minima:
            nome = classes_nomes[id_resultado]
            cor = classes_cores[id_resultado]

            cv2.rectangle(imagem_display, (5, 5), (w - 5, h - 5), cor, 4)

            label_text = f"{nome} ({confianca_maxima:.1f}%)"
            cv2.putText(imagem_display, label_text, (20, 115), font, 1.2, cor, 2)

            tempo_atual = time.time()
            ignorar_nada_repetido = id_resultado == 3 and ultima_classe_reconhecida == 3

            if not ignorar_nada_repetido:
                if tempo_ultimo_reconhecimento is not None:
                    intervalo_reconhecimento = tempo_atual - tempo_ultimo_reconhecimento
                    soma_intervalos_reconhecimento += intervalo_reconhecimento
                    contador_intervalos_reconhecimento += 1
                tempo_ultimo_reconhecimento = tempo_atual
                ultima_classe_reconhecida = id_resultado

            if id_resultado == 0 and arduino_conectado:
                if tempo_atual - tempo_ultimo_comando >= intervalo_minimo_comandos:
                    if tempo_atual - tempo_ultima_deteccao_uva >= intervalo_verificacao_uva:
                        contador_deteccoes_uva += 1
                        tempo_ultima_deteccao_uva = tempo_atual

                        print(f"    🍇 Detecção UVA #{contador_deteccoes_uva}/3 (Conf: {confianca_maxima:.1f}%)")

                        if contador_deteccoes_uva >= deteccoes_necessarias_uva:
                            if enviar_comando_arduino_cronometrado(b'U', tempo_atual):
                                contador_comandos_uva += 1
                                print(f"\n{'='*50}")
                                print(f"🍇 COMANDO 'U' ENVIADO (UVA → 180°) 🍇")
                                print(f"{'='*50}")

                                threading.Thread(target=enviar_centralizacao_atrasada, daemon=True).start()

                                ultimo_comando_enviado = 0
                                tempo_ultimo_comando = tempo_atual
                                contador_deteccoes_uva = 0

            elif id_resultado in (1, 2) and arduino_conectado:
                if contador_deteccoes_uva > 0:
                    print(f"    ❌ {classes_nomes[id_resultado]} - RESET contador ({contador_deteccoes_uva}/3)")
                    contador_deteccoes_uva = 0

                if tempo_atual - tempo_ultimo_comando >= intervalo_minimo_comandos:
                    if enviar_comando_arduino_cronometrado(b'N', tempo_atual):
                        if id_resultado == 1:
                            contador_comandos_uva_ruim += 1
                        else:
                            contador_comandos_nao_uva += 1
                        print(f"\n{'='*50}")
                        print(f"❌ COMANDO 'N' ENVIADO ({classes_nomes[id_resultado]} → 0°) ❌")
                        print(f"{'='*50}")

                        threading.Thread(target=enviar_centralizacao_atrasada, daemon=True).start()

                        ultimo_comando_enviado = id_resultado
                        tempo_ultimo_comando = tempo_atual

            elif id_resultado == 3:
                if contador_deteccoes_uva > 0:
                    contador_deteccoes_uva = 0
                if not ignorar_nada_repetido:
                    contador_deteccoes_nada += 1
                if arduino_conectado and not ignorar_nada_repetido:
                    print(f"    ⚪ NADA detectado ({confianca_maxima:.1f}%) - nenhum comando enviado")

            if not ignorar_nada_repetido or id_resultado != 3:
                print(id_resultado)
        else:
            cv2.rectangle(imagem_display, (5, 5), (w - 5, h - 5), (128, 128, 128), 3)
            cv2.putText(imagem_display, f"INCERTO ({confianca_maxima:.1f}%)", (20, 115), font, 1, (128, 128, 128), 2)
    
    # ============================================================
    # INDICADORES NA TELA
    # ============================================================
    cv2.putText(imagem_display, f"Entrada camera: {w}x{h} | IA: 224x224", (20, 40), font, 1.0, (255, 255, 255), 2)
    cv2.putText(imagem_display, "Tecla: Q sair", (20, 70), font, 0.9, (255, 255, 255), 1)
    cv2.putText(imagem_display, "Pipeline: IMAGEM INTEIRA + RESIZE/PADDING", (20, 95), font, 0.9, (0, 255, 255), 1)
    
    if contador_deteccoes_uva > 0:
        cv2.putText(imagem_display, f"Verificacao UVA: {contador_deteccoes_uva}/3", 
                   (20, 110), font, 1.2, (0, 255, 255), 2)
        
        for i in range(3):
            cor_circulo = (0, 255, 0) if i < contador_deteccoes_uva else (100, 100, 100)
            cv2.circle(imagem_display, (280 + i*40, 105), 12, cor_circulo, -1)
    
    if arduino_conectado:
        cv2.putText(imagem_display, f"Arduino: OK | U:{contador_comandos_uva} UR:{contador_comandos_uva_ruim} N:{contador_comandos_nao_uva}", 
                   (20, h-20), font, 1, (0, 255, 0), 1)

    if contador_intervalos_reconhecimento > 0:
        media_intervalo = soma_intervalos_reconhecimento / contador_intervalos_reconhecimento
        cv2.putText(imagem_display, f"Med. intervalo reconhecimento: {media_intervalo:.2f}s",
                   (20, h-50), font, 0.8, (255, 255, 0), 1)

    if contador_envios_comando > 0:
        media_latencia_cmd = soma_latencia_envio_comando / contador_envios_comando
        cv2.putText(imagem_display, f"Med. latencia cmd Arduino: {media_latencia_cmd*1000:.1f}ms",
                   (20, h-35), font, 0.8, (0, 255, 255), 1)
    
    cv2.imshow("Reconhecedor de Uvas - Detecção + Classificação", imagem_display)
    
    tecla = cv2.waitKey(1) & 0xFF
    
    if tecla == ord('q'):
        print("\n🛑 Encerrando...")
        break

# ============================================================
# FINALIZAÇÃO
# ============================================================
print("\n🧹 Limpando recursos...")

camera.release()
cv2.destroyAllWindows()

if arduino_conectado:
    print("↩️  Centralizando servo...")
    enviar_comando_arduino(b'C')
    time.sleep(0.5)
    
    arduino.close()
    print("🔌 Arduino desconectado.")

print("\n" + "=" * 60)
print("🏁 SISTEMA ENCERRADO")
print(f"🍇 Comandos 'U' (UVA): {contador_comandos_uva}")
print(f"🟠 Comandos 'N' (UVA_RUIM): {contador_comandos_uva_ruim}")
print(f"❌ Comandos 'N' (NÃO-UVA): {contador_comandos_nao_uva}")
print(f"⚪ Detecções 'NADA' (sem comando): {contador_deteccoes_nada}")
if contador_intervalos_reconhecimento > 0:
    media_intervalo = soma_intervalos_reconhecimento / contador_intervalos_reconhecimento
    print(f"⏱️  Tempo médio entre reconhecimentos: {media_intervalo:.3f}s")
else:
    print("⏱️  Tempo médio entre reconhecimentos: sem dados suficientes")

if contador_envios_comando > 0:
    media_latencia_cmd = soma_latencia_envio_comando / contador_envios_comando
    print(f"⏱️  Tempo médio reconhecimento→comando Arduino: {media_latencia_cmd*1000:.1f}ms")
else:
    print("⏱️  Tempo médio reconhecimento→comando Arduino: sem comandos enviados")
print("=" * 60)

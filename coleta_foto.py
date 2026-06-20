import cv2
from pathlib import Path


def calcular_nitidez(frame):
    """Retorna uma métrica simples de nitidez (variância do Laplaciano)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def capturar_frame_mais_nitido(camera, tentativas=8):
    """Lê vários frames e devolve o mais nítido para reduzir foto borrada."""
    melhor_frame = None
    melhor_nitidez = -1.0

    for _ in range(tentativas):
        ret, frame = camera.read()
        if not ret:
            continue

        nitidez = calcular_nitidez(frame)
        if nitidez > melhor_nitidez:
            melhor_nitidez = nitidez
            melhor_frame = frame

    return melhor_frame, melhor_nitidez

# Configurações
CAMERA_ID = 0
BASE_DIR = Path("video")

# Resolução de captura para treino
TARGET_WIDTH = 224
TARGET_HEIGHT = 224
LIMIAR_NITIDEZ = 120.0

# Inicializa câmera (CAP_DSHOW costuma dar melhor controle no Windows)
camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Não foi possível abrir a câmera.")
    quit()

# Configura resolução alvo
camera.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verifica resolução real obtida
actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("="*60)

# Pequeno aquecimento para estabilizar exposição/foco
for _ in range(20):
    camera.read()
print(f"Resolução configurada: {actual_width}x{actual_height}")
if actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
    print("✅ Resolução 224x224 ativada!")
else:
    print(f"⚠️  224x224 não disponível, usando {actual_width}x{actual_height}")
print("="*60)

# Solicita categoria
print("\nEscolha a categoria para captura:")
print("1 - Uva")
print("2 - Uva_ruim")
print("3 - Nao_uva")
print("4 - Nada")

categoria_input = input("\nDigite o número da categoria (1/2/3/4): ").strip()

# Define categoria e configurações
if categoria_input == "1":
    categoria = "Uva"
    prefixo = "uva"
elif categoria_input == "2":
    categoria = "Uva_ruim"
    prefixo = "uva_ruim"
elif categoria_input == "3":
    categoria = "Nao_uva"
    prefixo = "nao_uva"
elif categoria_input == "4":
    categoria = "Nada"
    prefixo = "nada"
else:
    print("❌ Opção inválida! Encerrando...")
    quit()

# Cria diretório da categoria
SAVE_DIR = BASE_DIR / categoria
SAVE_DIR.mkdir(parents=True, exist_ok=True)

foto_count = 0

print(f"\n✅ Modo: {categoria}")
print(f"📁 Salvando em: {SAVE_DIR}")
print("\nPressione ESPAÇO para tirar foto | A para ligar/desligar autofoco | ESC para sair")

autofoco_ligado = True

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Mostra preview com informações
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]
    nitidez_preview = calcular_nitidez(display_frame)
    
    # Info no topo
    cv2.putText(display_frame, f"Categoria: {categoria} | Fotos: {foto_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f"{w}x{h}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Nitidez: {nitidez_preview:.1f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Autofoco: {'ON' if autofoco_ligado else 'OFF'}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow("Camera", display_frame)
    
    # Captura tecla
    key = cv2.waitKey(1) & 0xFF
    
    # ESPAÇO: salva exatamente o frame atual do preview
    if key == ord(' '):
        frame_capturado = frame.copy()
        nitidez_captura = calcular_nitidez(frame_capturado)

        filename = SAVE_DIR / f"{prefixo}-224_{foto_count:04d}.jpg"
        cv2.imwrite(str(filename), frame_capturado, [cv2.IMWRITE_JPEG_QUALITY, 98])
        foto_count += 1

        if nitidez_captura < LIMIAR_NITIDEZ:
            print(f"⚠️ Foto {foto_count} salva, mas nitidez baixa ({nitidez_captura:.1f}).")
        else:
            print(f"✅ Foto {foto_count} salva: {filename.name} | nitidez {nitidez_captura:.1f}")

    # A: alterna autofoco
    elif key == ord('a'):
        autofoco_ligado = not autofoco_ligado
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofoco_ligado else 0)
        print(f"🔧 Autofoco {'ligado' if autofoco_ligado else 'desligado'}")
    
    # ESC: sai
    elif key == 27:
        break

camera.release()
cv2.destroyAllWindows()
print(f"\n✅ Total: {foto_count} fotos salvas em {SAVE_DIR}")

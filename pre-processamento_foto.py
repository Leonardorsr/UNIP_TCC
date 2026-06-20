import cv2
from pathlib import Path


INPUT_BASE_DIR = Path("video")
OUTPUT_BASE_DIR = Path("video_pre-processados")
CLASSES = ["uva", "uva_ruim", "nao_uva", "nada"]
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_SIZE = (224, 224)
JPEG_QUALITY = 95
PADDING_COLOR = (127, 127, 127)


def resize_com_padding(imagem, target_size, padding_color=PADDING_COLOR):
    """Redimensiona preservando proporcao e completa com padding ate target_size."""
    target_w, target_h = target_size
    h, w = imagem.shape[:2]

    if h == 0 or w == 0:
        return None

    escala = min(target_w / w, target_h / h)
    novo_w = max(1, int(round(w * escala)))
    novo_h = max(1, int(round(h * escala)))

    imagem_redimensionada = cv2.resize(imagem, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    imagem_final = cv2.copyMakeBorder(
        imagem_redimensionada,
        (target_h - novo_h) // 2,
        target_h - novo_h - ((target_h - novo_h) // 2),
        (target_w - novo_w) // 2,
        target_w - novo_w - ((target_w - novo_w) // 2),
        cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return imagem_final


def redimensionar_classe(nome_classe: str) -> tuple[int, int, int]:
    """Redimensiona imagens de uma classe e retorna (lidas, salvas, falhas)."""
    pasta_entrada = INPUT_BASE_DIR / nome_classe
    pasta_saida = OUTPUT_BASE_DIR / nome_classe
    pasta_saida.mkdir(parents=True, exist_ok=True)

    if not pasta_entrada.exists():
        print(f"❌ Pasta de entrada não encontrada: {pasta_entrada}")
        return 0, 0, 0

    arquivos = [
        p for p in sorted(pasta_entrada.iterdir())
        if p.is_file() and p.suffix.lower() in EXTENSOES_VALIDAS
    ]

    # Remove arquivos antigos para substituir completamente o lote resized.
    for antigo in pasta_saida.iterdir():
        if antigo.is_file() and antigo.suffix.lower() in EXTENSOES_VALIDAS:
            antigo.unlink()

    lidas = len(arquivos)
    salvas = 0
    falhas = 0

    print(f"\n📁 Classe: {nome_classe}")
    print(f"   Entrada: {pasta_entrada}")
    print(f"   Saída:   {pasta_saida}")
    print(f"   Arquivos encontrados: {lidas}")

    for arquivo in arquivos:
        imagem = cv2.imread(str(arquivo))
        if imagem is None:
            falhas += 1
            print(f"   ⚠️ Falha ao ler: {arquivo.name}")
            continue

        imagem_redimensionada = resize_com_padding(imagem, TARGET_SIZE)
        if imagem_redimensionada is None:
            falhas += 1
            print(f"   ⚠️ Falha ao processar: {arquivo.name}")
            continue

        caminho_saida = pasta_saida / arquivo.name
        ok = cv2.imwrite(
            str(caminho_saida),
            imagem_redimensionada,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if ok:
            salvas += 1
        else:
            falhas += 1
            print(f"   ⚠️ Falha ao salvar: {arquivo.name}")

    return lidas, salvas, falhas


def main() -> None:
    print("=" * 60)
    print("🖼️  PRÉ-PROCESSAMENTO = RESIZE COM PADDING (224x224)")
    print("=" * 60)
    print(f"Origem: {INPUT_BASE_DIR}")
    print(f"Destino: {OUTPUT_BASE_DIR}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("=" * 60)

    total_lidas = 0
    total_salvas = 0
    total_falhas = 0

    for classe in CLASSES:
        lidas, salvas, falhas = redimensionar_classe(classe)
        total_lidas += lidas
        total_salvas += salvas
        total_falhas += falhas

    print("\n" + "=" * 60)
    print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"Total lidas:  {total_lidas}")
    print(f"Total salvas: {total_salvas}")
    print(f"Total falhas: {total_falhas}")
    print(f"\nPastas de saída:")
    for classe in CLASSES:
        print(f"  - {OUTPUT_BASE_DIR / classe}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    main()

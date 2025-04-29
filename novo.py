#!/usr/bin/env python3
"""
Script para gerar embeddings de imagens de desenho (Enzo e Marcelo) e comparar com imagens Canny.
Calcula a similaridade de cossenos e indica qual desenho é mais similar para cada imagem.
"""
import os
import glob

import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


def embed_image(path, processor, model):
    # Abre a imagem
    image = Image.open(path)
    # Converte desenhos de Enzo e Marcelo para preto e branco antes de embeddar
    # mantendo três canais para o modelo ViT
    
    image = image.convert("L")  # Converte para escala de cinza
    image = image.point(lambda x: 255 if x > 200 else 0, mode='1')  # Binariza
    image = image.convert("RGB")  # Converte para 3 canais para o ViT

    image.show()

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # Usa o token [CLS] como embedding
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().reshape(-1)
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def main():
    enzo_dir = "enzo_desenho"
    marcelo_dir = "marcelo_desenho"
    canny_dir = "fotos_canny"

    # Carrega modelo e processador ViT
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Lista arquivos de desenho de Enzo
    enzo_files = sorted(f for f in os.listdir(enzo_dir)
                        if os.path.isfile(os.path.join(enzo_dir, f)))

    for filename in enzo_files:
        name, _ = os.path.splitext(filename)
        enzo_path = os.path.join(enzo_dir, filename)
        marcelo_path = os.path.join(marcelo_dir, filename)

        # Encontra arquivo Canny correspondente
        pattern = os.path.join(canny_dir, f"canny_{name}.*")
        canny_list = glob.glob(pattern)

        if not canny_list:
            print(f"Nenhuma imagem Canny encontrada para '{name}', pulando.")
            continue
        
        canny_path = canny_list[0]

        # Gera embeddings
        emb_enzo = embed_image(enzo_path, processor, model)
        emb_marcelo = embed_image(marcelo_path, processor, model)
        emb_canny = embed_image(canny_path, processor, model)

        # Calcula similaridades
        sim_enzo = cosine_similarity(emb_enzo, emb_canny)
        sim_marcelo = cosine_similarity(emb_marcelo, emb_canny)

        # Exibe resultados
        print(f"Imagem: {name}")
        print(f"  Similaridade Enzo   : {sim_enzo:.4f}")
        print(f"  Similaridade Marcelo: {sim_marcelo:.4f}")
        if sim_enzo > sim_marcelo:
            vencedor = "Desenho do Enzo"
            sim_vencedor = sim_enzo
        else:
            vencedor = "Desenho do Marcelo"
            sim_vencedor = sim_marcelo
        print(f"  => {vencedor} é mais similar à imagem Canny ({sim_vencedor:.4f})")
        print()


if __name__ == "__main__":
    main()
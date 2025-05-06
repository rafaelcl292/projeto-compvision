#!/usr/bin/env python3
"""
Script para gerar uma tabela de comparação entre imagens de desenhos de múltiplos jogadores
e suas respectivas imagens Canny de referência, usando embeddings do modelo ViT.
"""
import os
import glob
from tabulate import tabulate

import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


def embed_image(path, processor, model):
    """Gera o embedding de uma imagem usando o modelo ViT."""
    image = Image.open(path)
    image = image.convert("L")  # Converte para escala de cinza
    image = image.point(lambda x: 255 if x > 200 else 0, mode='1')  # Binariza
    image = image.convert("RGB")  # Converte para 3 canais para o ViT
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().reshape(-1)
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula a similaridade de cossenos entre dois vetores."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def main():
    print("Iniciando comparação de imagens...")

    # Carrega modelo e processador ViT
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    players_dir = "players"
    canny_dir = "fotos_canny"

    # Lista jogadores
    players = sorted(d for d in os.listdir(players_dir)
                     if os.path.isdir(os.path.join(players_dir, d)))
    if not players:
        print("Nenhum jogador encontrado na pasta 'players'.")
        return

    # Obtém lista de desenhos a partir do primeiro jogador
    first_player_dir = os.path.join(players_dir, players[0])
    drawing_files = sorted(f for f in os.listdir(first_player_dir)
                          if os.path.isfile(os.path.join(first_player_dir, f)))

    # Estrutura para armazenar dados da tabela
    table_data = []
    headers = ["Desenho"] + players

    for filename in drawing_files:
        name, _ = os.path.splitext(filename)
        # Encontra arquivo Canny correspondente
        pattern = os.path.join(canny_dir, f"canny_{name}.*")
        canny_list = glob.glob(pattern)

        if not canny_list:
            print(f"Nenhuma imagem Canny encontrada para '{name}', pulando.")
            continue

        canny_path = canny_list[0]
        emb_canny = embed_image(canny_path, processor, model)

        # Calcula similaridades para cada jogador
        row = [name]
        sim_scores = {}
        for player in players:
            player_path = os.path.join(players_dir, player, filename)
            if not os.path.isfile(player_path):
                print(f"Aviso: {player} não tem o desenho '{filename}', pulando.")
                row.append("N/A")
                continue
            emb = embed_image(player_path, processor, model)
            sim = cosine_similarity(emb, emb_canny)
            sim_scores[player] = sim
            row.append(f"{sim:.4f}")

        table_data.append(row)

    # Gera e exibe a tabela
    print("\nTabela de Comparação de Similaridades com Imagens Canny:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()

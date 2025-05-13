#!/usr/bin/env python3
"""
Script para gerar embeddings de imagens de desenho de múltiplos jogadores e comparar com imagens Canny.
Calcula a similaridade de cossenos, indica o vencedor de cada desenho e quem obteve a maior pontuação.
"""

import glob
import os

import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


def embed_image(path, processor, model):
    # Abre a imagem
    image = Image.open(path)
    # Converte desenhos de Enzo e Marcelo para preto e branco antes de embeddar
    # mantendo três canais para o modelo ViT

    image = image.convert("L")  # Converte para escala de cinza
    image = image.point(lambda x: 255 if x > 200 else 0, mode="1")  # Binariza
    image = image.convert("RGB")  # Converte para 3 canais para o ViT

    # Gera embeddings para várias rotações e faz a média
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    embeddings = []
    for angle in angles:
        rotated = image.rotate(angle, expand=True)
        inputs = processor(images=rotated, return_tensors="pt")
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].detach().numpy().reshape(-1)
        embeddings.append(emb)
    embedding = np.max(embeddings, axis=0)
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def main():
    print("Iniciando comparação de imagens...")

    # Carrega modelo e processador ViT
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    players_dir = "players"
    canny_dir = "fotos_canny"

    # Lista jogadores
    players = sorted(
        d
        for d in os.listdir(players_dir)
        if os.path.isdir(os.path.join(players_dir, d))
    )
    # Inicializa contagem de vitórias
    wins_count = {player: 0 for player in players}

    # Obtém lista de desenhos a partir do primeiro jogador
    if not players:
        print("Nenhum jogador encontrado na pasta 'players'.")
        return

    first_player_dir = os.path.join(players_dir, players[0])
    drawing_files = sorted(
        f
        for f in os.listdir(first_player_dir)
        if os.path.isfile(os.path.join(first_player_dir, f))
    )

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
        sim_scores = {}
        for player in players:
            player_path = os.path.join(players_dir, player, filename)
            if not os.path.isfile(player_path):
                print(f"Aviso: {player} não tem o desenho '{filename}', pulando.")
                continue
            emb = embed_image(player_path, processor, model)
            sim = cosine_similarity(emb, emb_canny)
            sim_scores[player] = sim

        if not sim_scores:
            continue

        # Exibe resultados individuais
        print(f"Imagem: {name}")
        for player, sim in sim_scores.items():
            print(f"  Similaridade {player: <10}: {sim:.4f}")

        # Determina vencedor(es) para este desenho
        max_sim = max(sim_scores.values())
        winners = [p for p, s in sim_scores.items() if s == max_sim]
        if len(winners) == 1:
            winner = winners[0]
            wins_count[winner] += 1
            print(f"  => Vencedor: {winner} (similaridade: {max_sim:.4f})")
        else:
            for w in winners:
                wins_count[w] += 1
            winners_str = ", ".join(winners)
            print(f"  => Empate entre: {winners_str} (similaridade: {max_sim:.4f})")
        print()

    # Exibe pontuação final
    print("Pontuação final:")
    for player, count in wins_count.items():
        print(f"  {player: <10}: {count}")
    # Determina campeão
    max_wins = max(wins_count.values())
    champions = [p for p, c in wins_count.items() if c == max_wins]
    if len(champions) == 1:
        print(f"Campeão: {champions[0]} com {max_wins} ponto(s)!")
    else:
        champs_str = " e ".join(champions)
        print(f"Empate geral entre: {champs_str} com {max_wins} ponto(s) cada!")


if __name__ == "__main__":
    main()

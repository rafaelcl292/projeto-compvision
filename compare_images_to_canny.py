#!/usr/bin/env python3
"""
Script para gerar uma tabela de comparação entre imagens de desenhos de múltiplos jogadores
e suas respectivas imagens Canny de referência, usando embeddings do modelo ViT, com visualização em HTML.
"""

import glob
import os

import numpy as np
from jinja2 import Template
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


def embed_image(path, processor, model):
    """Gera o embedding de uma imagem usando o modelo ViT."""
    image = Image.open(path)
    image = image.convert("L")  # Converte para escala de cinza
    image = image.point(lambda x: 255 if x > 200 else 0, mode="1")  # Binariza
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
    if not players:
        print("Nenhum jogador encontrado na pasta 'players'.")
        return

    # Obtém lista de desenhos a partir do primeiro jogador
    first_player_dir = os.path.join(players_dir, players[0])
    drawing_files = sorted(
        f
        for f in os.listdir(first_player_dir)
        if os.path.isfile(os.path.join(first_player_dir, f))
    )

    # Estrutura para armazenar dados da tabela
    table_data = []

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
        for player in players:
            player_path = os.path.join(players_dir, player, filename)
            if not os.path.isfile(player_path):
                print(f"Aviso: {player} não tem o desenho '{filename}', pulando.")
                row.append("N/A")
                continue
            emb = embed_image(player_path, processor, model)
            sim = cosine_similarity(emb, emb_canny)
            row.append(f"{sim:.4f}")

        table_data.append(row)

    # Template HTML
    html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Tabela de Comparação de Similaridades com Imagens Canny</title>
    <style>
        table { border-collapse: collapse; }
        th, td { border: 1px solid black; padding: 8px; text-align: center; }
        img { width: 100px; height: auto; }
    </style>
</head>
<body>
    <h1>Tabela de Comparação de Similaridades com Imagens Canny</h1>
    <table>
        <tr>
            <th>Desenho</th>
            {% for player in players %}
            <th>{{ player }}</th>
            {% endfor %}
        </tr>
        {% for row in table_data %}
        <tr>
            <td>{{ row[0] }}</td>
            {% for sim in row[1:] %}
            <td>{{ sim }}</td>
            {% endfor %}
        </tr>
        <tr>
            <td><img src="{{ canny_dir }}/canny_{{ row[0] }}.png" alt="{{ row[0] }}"></td>
            {% for player in players %}
            <td><img src="{{ players_dir }}/{{ player }}/{{ row[0] }}.png" alt="{{ player }} - {{ row[0] }}"></td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
</body>
</html>
    """

    # Gera o HTML
    template = Template(html_template)
    html = template.render(
        players=players,
        table_data=table_data,
        players_dir=players_dir,
        canny_dir=canny_dir,
    )

    # Salva o HTML em um arquivo
    with open("tabela_com_imagens.html", "w") as f:
        f.write(html)

    print("Tabela com imagens gerada em 'tabela_com_imagens.html'.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script para gerar uma tabela de comparação entre imagens de desenhos do jogador Bruno
com transformações (original, 75% tamanho, 150% tamanho, 45° rotação, 90° rotação, dilatação 1x)
e suas respectivas imagens Canny de referência, usando embeddings do modelo ViT, com visualização em HTML.
"""

import glob
import os

import cv2
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


def apply_transformations(image_path, output_dir, filename):
    """Aplica transformações à imagem e salva os resultados, retornando caminhos."""
    image = Image.open(image_path)
    base_name, _ = os.path.splitext(filename)
    paths = {}

    # Original
    original_path = os.path.join(output_dir, f"{base_name}_original.png")
    image.save(original_path)
    paths["original"] = original_path

    # 75% tamanho
    size_75 = (int(image.width * 0.75), int(image.height * 0.75))
    image_75 = image.resize(size_75, Image.Resampling.LANCZOS)
    path_75 = os.path.join(output_dir, f"{base_name}_75percent.png")
    image_75.save(path_75)
    paths["75percent"] = path_75

    # 150% tamanho
    size_150 = (int(image.width * 1.5), int(image.height * 1.5))
    image_150 = image.resize(size_150, Image.Resampling.LANCZOS)
    path_150 = os.path.join(output_dir, f"{base_name}_150percent.png")
    image_150.save(path_150)
    paths["150percent"] = path_150

    # 45° rotação
    image_45 = image.rotate(45, expand=True, resample=Image.Resampling.BICUBIC)
    path_45 = os.path.join(output_dir, f"{base_name}_45rotate.png")
    image_45.save(path_45)
    paths["45rotate"] = path_45

    # 90° rotação
    image_90 = image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    path_90 = os.path.join(output_dir, f"{base_name}_90rotate.png")
    image_90.save(path_90)
    paths["90rotate"] = path_90

    # Dilatação 1x
    image_np = np.array(image.convert("L"))
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.erode(image_np, kernel, iterations=1)
    image_dilated = Image.fromarray(dilated)
    path_dilate = os.path.join(output_dir, f"{base_name}_dilate.png")
    image_dilated.save(path_dilate)
    paths["dilate"] = path_dilate

    return paths


def main():
    print("Iniciando comparação de imagens para Bruno...")

    # Carrega modelo e processador ViT
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    bruno_dir = "players/bruno"
    canny_dir = "fotos_canny"
    transformed_dir = "transformed_images"

    # Cria diretório para imagens transformadas
    os.makedirs(transformed_dir, exist_ok=True)

    # Lista desenhos do Bruno
    if not os.path.isdir(bruno_dir):
        print(f"Diretório '{bruno_dir}' não encontrado.")
        return
    drawing_files = sorted(
        f for f in os.listdir(bruno_dir) if os.path.isfile(os.path.join(bruno_dir, f))
    )
    if not drawing_files:
        print("Nenhum desenho encontrado em 'players/bruno/'.")
        return

    # Estrutura para armazenar dados da tabela
    table_data = []
    transformations = [
        "original",
        "75percent",
        "150percent",
        "45rotate",
        "90rotate",
        "dilate",
    ]
    transform_labels = [
        "Original",
        "75% Tamanho",
        "150% Tamanho",
        "45° Rotação",
        "90° Rotação",
        "Dilatação 1x",
    ]

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

        # Aplica transformações e obtém caminhos
        bruno_path = os.path.join(bruno_dir, filename)
        transformed_paths = apply_transformations(bruno_path, transformed_dir, filename)

        # Calcula similaridades para cada transformação
        row = [name]
        for transform in transformations:
            if transform not in transformed_paths:
                row.append("N/A")
                continue
            emb = embed_image(transformed_paths[transform], processor, model)
            sim = cosine_similarity(emb, emb_canny)
            row.append(f"{sim:.4f}")

        table_data.append((row, transformed_paths))

    # Template HTML
    html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Tabela de Comparação de Transformações com Imagens Canny</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid black; padding: 8px; text-align: center; }
        img { width: 100px; height: auto; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Tabela de Comparação de Transformações com Imagens Canny (Bruno)</h1>
    <table>
        <tr>
            <th>Desenho</th>
            {% for label in transform_labels %}
            <th>{{ label }}</th>
            {% endfor %}
        </tr>
        {% for row, paths in table_data %}
        <tr>
            <td>{{ row[0] }}</td>
            {% for sim in row[1:] %}
            <td>{{ sim }}</td>
            {% endfor %}
        </tr>
        <tr>
            <td><img src="{{ canny_dir }}/canny_{{ row[0] }}.png" alt="Canny {{ row[0] }}"></td>
            {% for transform in transformations %}
            <td><img src="{{ paths[transform] }}" alt="{{ transform }} - {{ row[0] }}"></td>
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
        transform_labels=transform_labels,
        table_data=table_data,
        transformations=transformations,
        canny_dir=canny_dir,
        paths=lambda: None,  # Placeholder para acessar paths dinamicamente
    )

    # Salva o HTML em um arquivo
    with open("tabela_transformacoes.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Tabela com transformações gerada em 'tabela_transformacoes.html'.")


if __name__ == "__main__":
    main()

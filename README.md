# Projeto de Visão Computacional

Este projeto é uma aplicação de visão computacional focada na análise de robustez do modelo Vision Transformer (ViT) `google/vit-base-patch16-224-in21k`. O objetivo principal é avaliar como o modelo se comporta frente a diferentes transformações de imagem, comparando sempre com a versão Canny da imagem original.

## 🎯 Objetivo

O projeto realiza uma análise estatística abrangente da robustez do modelo ViT, gerando mais de 1000 variações de cada imagem através de diferentes combinações de:

-   Redimensionamento (resize)
-   Rotação
-   Dilatação

Cada variação é comparada com a transformação Canny da imagem original, permitindo avaliar a consistência e robustez do modelo frente a diferentes transformações geométricas.

## 🚀 Funcionalidades

-   Processamento de imagens usando OpenCV
-   Detecção de bordas com algoritmo de Canny
-   Transformação e manipulação de imagens
-   Geração de comparações visuais
-   Análise estatística de resultados
-   Avaliação de robustez do modelo ViT
-   Geração de múltiplas variações de imagens
-   Comparação de similaridade usando embeddings do modelo

## 📋 Pré-requisitos

-   Python >= 3.13
-   uv (gerenciador de pacotes Python)
-   Dependências listadas em `pyproject.toml`

## 🛠️ Instalação

1. Clone o repositório
2. Instale o uv (se ainda não tiver instalado):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Crie um ambiente virtual e instale as dependências usando uv:

```bash
uv sync
```

## 📁 Estrutura do Projeto

```
.
├── fotos/                  # Diretório de imagens originais
├── fotos_canny/           # Imagens processadas com algoritmo de Canny
├── players/               # Diretório de imagens de jogadores
├── tabelas_comparacoes/   # Resultados de comparações
├── teste_estatistico/     # Análises estatísticas
│   ├── generate_variations_evaluate.py  # Script principal de geração e avaliação
│   └── transformation_results/          # Resultados das transformações e gráficos
├── utils/                 # Utilitários e scripts de processamento
│   ├── transform_to_canny.py    # Script de transformação Canny
│   └── base_comparisson.py      # Script de comparação base
└── pyproject.toml         # Configuração do projeto e dependências
```

## 🎯 Principais Componentes

### Processamento de Imagens

-   Transformação de imagens para escala de cinza
-   Detecção de bordas usando algoritmo de Canny
-   Dilatação de bordas para melhor visualização
-   Inversão de cores para melhor contraste

### Análise e Comparação

-   Geração de tabelas comparativas em HTML
-   Análise estatística dos resultados
-   Visualização de transformações
-   Avaliação de similaridade usando embeddings do ViT
-   Geração de gráficos de análise de robustez

### Transformações Aplicadas

-   Redimensionamento: variação de 50% a 150% do tamanho original
-   Rotação: ângulos de 0° a 360° em intervalos de 18°
-   Dilatação: 1 a 3 iterações com kernel 3x3

## 📊 Dependências Principais

-   OpenCV (opencv-python) >= 4.11.0.86
-   Matplotlib >= 3.10.1
-   SciPy >= 1.15.3
-   Seaborn >= 0.13.2
-   TorchVision >= 0.22.0
-   Transformers >= 4.51.3
-   Jinja2 >= 3.1.6

## 🔧 Uso

1. Coloque as imagens a serem processadas no diretório `fotos/`
2. Execute o script de transformação Canny:

```bash
python utils/transform_to_canny.py
```

3. Execute o script de geração e avaliação de variações:

```bash
python teste_estatistico/generate_variations_evaluate.py
```

4. Os resultados serão salvos em `teste_estatistico/transformation_results/`
5. Analise os gráficos e resultados gerados

## 📝 Notas

-   O projeto utiliza o algoritmo de Canny para detecção de bordas
-   As imagens são processadas em escala de cinza
-   O kernel de dilatação usado é 3x3
-   Os resultados incluem visualizações comparativas em HTML
-   A similaridade é calculada usando cosine similarity entre embeddings do modelo ViT
-   Cada imagem gera mais de 1000 variações para análise estatística

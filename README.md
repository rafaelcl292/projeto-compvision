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

## 🎯 Principais Componentes

### Geração e Avaliação de Variações (teste_estatistico/generate_variations_evaluate.py)

Este script é responsável por gerar e avaliar múltiplas variações de imagens usando o modelo ViT. Suas principais funcionalidades incluem:

-   Geração sistemática de variações de imagens combinando:
    -   Redimensionamento (50% a 150% do tamanho original)
    -   Rotação (0° a 360° em intervalos de 18°)
    -   Dilatação (1 a 3 iterações)
-   Cálculo de similaridade entre embeddings usando cosine similarity
-   Processamento de imagens individuais com interface interativa
-   Geração de resultados para cada combinação de transformações
-   Armazenamento dos resultados em arquivos de texto organizados por jogador
-   Utilização do modelo ViT (google/vit-base-patch16-224-in21k) para extração de embeddings
-   Comparação automática com versões Canny das imagens originais

O script permite selecionar interativamente quais imagens processar e gera resultados detalhados para análise posterior.

### Análise Estatística e Visualização (teste_estatistico/graphs_all_images.py)

Este script realiza uma análise estatística completa dos resultados gerados, criando visualizações e testes estatísticos. Suas principais funcionalidades incluem:

-   Análise estatística detalhada para cada conjunto de dados:
    -   Cálculo de média, mediana, desvio padrão
    -   Identificação de valores mínimos e máximos
    -   Contagem de amostras
-   Geração de visualizações comparativas:
    -   Box plots para distribuição dos resultados
    -   Gráficos de barras para p-valores dos testes estatísticos
    -   Histogramas para distribuição dos resultados
-   Realização de testes estatísticos:
    -   Testes T para comparação entre diferentes conjuntos de dados
    -   Cálculo de p-valores e significância estatística
-   Processamento de múltiplos conjuntos de dados:
    -   Análise de resultados para diferentes imagens (Estrela, Mack, Raposa, etc.)
    -   Comparação entre diferentes jogadores
    -   Inclusão de casos base para referência
-   Armazenamento automático dos resultados:
    -   Geração de gráficos em alta resolução
    -   Salvamento em diretório específico para resultados estatísticos
    -   Formatação clara dos resultados numéricos

O script gera relatórios detalhados tanto em formato visual (gráficos) quanto numérico (estatísticas), facilitando a interpretação dos resultados das transformações.

### Caso Base (teste_estatistico/base_case/base_case.py)

A pasta `base_case` contém os resultados de referência para comparação com as transformações. Suas principais características incluem:

-   Armazenamento de similaridades Canny:
    -   Resultados de comparação direta com as imagens Canny originais
    -   Arquivos de texto contendo valores de similaridade para cada imagem
    -   Formato padronizado para fácil comparação com resultados transformados
-   Uso como referência estatística:
    -   Serve como linha de base para avaliação das transformações
    -   Permite comparar o impacto das transformações em relação ao caso original
    -   Facilita a identificação de quais transformações mantêm melhor a similaridade
-   Integração com análise estatística:
    -   Utilizado nos gráficos de distribuição para comparação visual
    -   Fornece contexto para interpretação dos resultados das transformações
    -   Ajuda a estabelecer benchmarks de performance

Esta pasta é fundamental para a análise comparativa do projeto, servindo como ponto de referência para todas as transformações realizadas.

### Análise de Médias e Seleção de Jogadores (lowest_mean.py)

Este script realiza uma análise crucial para a seleção dos jogadores no jogo, identificando as menores médias de similaridade com significância estatística. Suas principais funcionalidades incluem:

-   Análise de resultados por jogador:
    -   Leitura dos resultados de transformação para cada jogador
    -   Cálculo da média de similaridade para cada participante
    -   Contagem do número de valores analisados
-   Comparação entre jogadores:
    -   Cálculo das diferenças entre médias de todos os pares de jogadores
    -   Ordenação das diferenças do menor para o maior
    -   Identificação dos três pares com menores diferenças
-   Seleção para o jogo:
    -   A menor média de similaridade, dos desenhos que obtiveram significância estatística, é selecionada como critério de desempate
    -   Caso a diferença de similaridade seja menor que o critério encontrado, haverá um empate

Resultados obtidos:

| Jogador   | Média    |
| --------- | -------- |
| raposa    | 0.007824 |
| cavalo    | 0.011552 |
| estrela   | 0.020191 |
| linus     | 0.020571 |
| luminaria | 0.020509 |
| mack      | 0.024050 |
| nike      | 0.031801 |
| gato      | 0.056632 |

Este script é fundamental para a mecânica do jogo, pois determina se haverá empate ou não.

### Estrutura de Pastas de Imagens

O projeto utiliza três pastas principais para gerenciar as imagens:

-   `fotos/`:

    -   Contém as imagens originais dos jogadores
    -   Imagens em formato PNG ou JPG
    -   Usadas como referência para todas as transformações
    -   Base para geração das versões Canny

-   `fotos_canny/`:

    -   Armazena as versões processadas com o algoritmo de Canny
    -   Geradas automaticamente a partir das imagens originais
    -   Usadas como referência para comparação de similaridade
    -   Nomeadas com prefixo "canny\_" para fácil identificação

-   `players/`:
    -   Organiza as imagens por jogador
    -   Cada jogador tem sua própria subpasta
    -   Contém as imagens originais e suas variações

## 📊 Dependências Principais

-   OpenCV (opencv-python) >= 4.11.0.86
-   Matplotlib >= 3.10.1
-   SciPy >= 1.15.3
-   Seaborn >= 0.13.2
-   TorchVision >= 0.22.0
-   Transformers >= 4.51.3
-   Jinja2 >= 3.1.6
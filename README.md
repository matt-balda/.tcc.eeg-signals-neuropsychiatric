# <Diagnóstico de transtornos mentais via EEG>

Insira aqui um resumo do projeto que será construído. Tente apresentar uma justificativa para o projeto. É desejável que também se insira um [graphical abstract](https://www.elsevier.com/authors/tools-and-resources/visual-abstract).

---
<!--

## Funcionalidades

Esse template foi inicialmente baseado no [template de ciência de dados do cookiecutter](https://drivendata.github.io/cookiecutter-data-science/), mas ao longo do tempo várias modificações foram sendo realizadas. Atualmente o template tem as seguintes características:

- Utilização do arquivo `pyproject.toml` como centralizador de dependências;
- Configuração para criação de aplicação `streamlit`;
- Utilização de [jupyter notebooks](https://jupyter.org/) para arquivos de análise;
- Documentação com o [mkdocs](https://www.mkdocs.org/) ([material design](https://squidfunk.github.io/mkdocs-material/) theme)

## Instruções

### Requisitos

Para utilizar este template, você precisará de um ambiente com os seguintes softwares:

- git
- Python 3.8
- Poetry `1.1.13` ou superior

É aconselhável o uso do `pyenv` para o gerenciamento de versões do Python.

Com o repositório clonado, você precisa navegar até a pasta local, usando o comando :

```
cd REPOSITORIO
```

Estando na pasta do repositório, basta instalar as dependências do projeto utilizando o comando:

```
poetry install
```

Ele irá instalar todas as dependências contidas no arquivo `pyproject.toml`. Depois disso basta ativar o ambiente virtual criado pelo Poetry utilizando o comando:

```
poetry shell
```

Para mais informações sobre os comandos do Poetry, visite a [documentação oficial](https://python-poetry.org/docs/). -->

### Organização de diretórios

```
.
├── data/              # Diretório contendo todos os arquivos de dados
│   ├── external/      # Arquivos de dados de fontes externas
│   ├── processed/     # Arquivos de dados processados
│   └── raw/           # Arquivos de dados originais, imutáveis
├── docs/              # Documentação gerada através da biblioteca mkdocs
├── models/            # Modelos treinados e serializados, predições ou resumos de modelos
├── notebooks/         # Diretório contendo todos os notebooks utilizados nos passos
├── references/        # Dicionários de dados
├── src/               # Código fonte utilizado nesse projeto
│   ├── data/          # Classes e funções utilizadas para download e processamento de dados
│   ├── deployment/    # Classes e funções utilizadas para implantação do modelo
│   └── model/         # Classes e funções utilizadas para modelagem
├── app.py             # Arquivo com o código da aplicação do streamlit
├── pyproject.toml     # Arquivo de dependências para reprodução do projeto
├── poetry.lock        # Arquivo com sub-dependências do projeto principal
└── README.md          # Informações gerais do projeto

```

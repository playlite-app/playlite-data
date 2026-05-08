# Pipeline de Dados – Filtragem Colaborativa

Esta pasta contém exclusivamente os **artefatos de ciência de dados** usados para gerar as recomendações colaborativas
do Playlite.

Diferente do código da aplicação, que roda no dispositivo do usuário, tudo aqui é executado **offline, em batch**,
durante o desenvolvimento. O resultado é um conjunto de arquivos prontos para consumo pelo aplicativo.

---

## 🎯 Sobre o Projeto e Datasets

O Playlite foi projetado para ser simples, rápido e local-first. Para manter essas características, qualquer
processamento pesado é **retirado do runtime da aplicação**.

Esta pasta existe para:

* Processar grandes volumes de dados de avaliações de jogos
* Extrair padrões globais de preferência entre jogadores
* Transformar esses padrões em dados estáticos
* Alimentar o sistema de recomendação do app sem custo computacional adicional

Neste projeto foi utilizado o dataset Game Recommendations on Steam (Kaggle) que contém interações reais de usuários
Steam com jogos. Ele inclui se o usuário recomendou ou não o jogo e pode ser utilizado para construir a matriz implícita
de preferências necessária para a filtragem colaborativa do Playlite.

---

## 🧠 O que NÃO acontece aqui

É importante destacar que esta pasta **não**:

* Processa dados do usuário do Playlite
* Executa inferência em tempo real
* Interage com o banco SQLite do aplicativo
* Depende de serviços externos em produção

Tudo aqui é isolado do usuário final.

---

## 📦 Estrutura da Pasta

```text
data/
 ├─ README.md          # Visão geral do pipeline de dados
 ├─ scripts/           # Scripts Python para processamento batch
 ├─ raw/               # Datasets brutos (não versionados)
 ├─ processed/         # Dados intermediários gerados pelos scripts
 ├─ outputs/           # Arquivos JSON finais consumidos pelo app
 └─ reports/           # Análises e estatísticas do processamento
```

---

## 🔁 Fluxo de Trabalho Esperado

O fluxo típico de uso desta pasta é:

1. Adicionar ou atualizar datasets em `datasets/`
2. Explorar e validar dados em `notebooks/`
3. Consolidar lógica em `scripts/`
4. Gerar arquivos finais em `outputs/`
5. Copiar os JSONs consolidados para o projeto principal

---

## 📊 Tipo de Processamento Realizado

Os scripts desta pasta são responsáveis por:

* Filtragem de jogos com volume mínimo de avaliações
* Conversão de avaliações em feedback implícito
* Cálculo de similaridade entre jogos
* Limitação e ordenação de vizinhos similares
* Preparação de metadados auxiliares (popularidade, categorias)

Nenhuma decisão de recomendação é tomada aqui — apenas **dados são preparados**.

---

## 📉 Estatísticas do Processamento

Os dados brutos passam por um rigoroso pipeline de limpeza para garantir que apenas interações de alta qualidade
alimentem o modelo. Abaixo estão as métricas da última execução (`2026-01-25`):

### Funil de Dados (Data Funnel)

Redução de ruído para manter apenas jogos e usuários estatisticamente relevantes:

| Métrica        | Dataset Original | Dataset Filtrado | Redução    |
|:---------------|:-----------------|:-----------------|:-----------|
| **Jogos**      | 50.872           | 14.319           | **-71,9%** |
| **Usuários**   | 14.3M            | 6.2M             | **-56,6%** |
| **Avaliações** | 41.1M            | 25.2M            | **-38,7%** |

### Critérios de Corte

Foram mantidos apenas dados que atendem aos seguintes requisitos simultâneos:

* **Jogos:** Mínimo de **100 avaliações** e **70% de aprovação** (foco em jogos estabelecidos e bem avaliados).
* **Usuários:** Mínimo de **2 reviews** e **2 horas de jogo** (removendo bots e contas inativas).

### Matriz de Interação Resultante

A matriz final utilizada para o cálculo de similaridade possui as seguintes características:

* **Dimensões:** 5.9M Usuários x 12k Jogos
* **Esparsidade:** 99,96% (Apenas 0,03% da matriz possui valores preenchidos)
* **Interações Totais:** ~25.2 Milhões

> **Nota:** A alta esparsidade confirma a necessidade do uso de algoritmos baseados em *Cosine Similarity*, que
> performam melhor neste cenário do que abordagens baseadas em densidade.

---

## 🔒 Privacidade e Ética

* Nenhum dado pessoal do usuário do Playlite é utilizado
* Nenhuma identificação individual é preservada nos outputs
* O objetivo é extrair padrões agregados, não comportamentos individuais

---

## 📊 Fonte de Dados & Créditos

Este projeto utiliza o conjunto de dados **Game Recommendations on Steam** de Anton Kozyriev,
disponibilizado através do Kaggle sob a licença CC0: Public Domain.

### Informações do Conjunto de Dados

- **Título:** Game Recommendations on Steam
- **Autor:** Anton Kozyriev
- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- **Licença:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **DOI:** [10.34740/KAGGLE/DS/2871694](https://doi.org/10.34740/KAGGLE/DS/2871694)
- **Acessado em:** Janeiro de 2026

### Citação

Se utilizar este projeto em trabalhos acadêmicos, por favor cite o conjunto de dados original:

```bibtex
@misc{anton_kozyriev_2023,
	title={Game Recommendations on Steam},
	url={https://www.kaggle.com/ds/2871694},
	DOI={10.34740/KAGGLE/DS/2871694},
	publisher={Kaggle},
	author={Anton Kozyriev},
	year={2023}
}
```

Ou no formato APA:
> Kozyriev, A. (2023). *Game Recommendations on Steam* [Data set]. Kaggle.
> https://doi.org/10.34740/KAGGLE/DS/2871694

---

## 📌 Observações

* Esta pasta pode evoluir conforme novas estratégias forem testadas.

Esta separação garante que o Playlite continue sendo um aplicativo **leve, rápido e previsível**, mesmo oferecendo
funcionalidades avançadas de recomendação.

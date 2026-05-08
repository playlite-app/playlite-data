"""
build_interactions.py - Construção da matriz de interações usuário-jogo

Este script calcula o valor de interação para cada par (user_id, app_id) baseado em:
- Horas jogadas (normalizado)
- Review positiva (implícito - todos os reviews aqui são positivos)

Entrada: data/processed/filtered_reviews.parquet
Saída: data/processed/interactions.parquet
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configurações
SCRIPT_DIR = Path(__file__).parent  # data/scripts
PROCESSED_DIR = SCRIPT_DIR.parent / "processed"  # data/processed
REPORTS_DIR = SCRIPT_DIR.parent / "reports"  # data/reports

# Parâmetros de normalização
MAX_HOURS_CAP = 50.0  # Cap de horas para normalização


def log_stats(stage, df, name):
    """Imprime estatísticas para debug"""
    print(f"[{stage}] {name}: {len(df):,} registros")


def normalize_hours(hours, max_cap=MAX_HOURS_CAP):
    """
    Normaliza horas jogadas para [0, 1] com cap

    Fórmula: min(hours / max_cap, 1.0)

    Exemplos:
    - 10 horas → 0.20
    - 25 horas → 0.50
    - 50 horas → 1.00
    - 100 horas → 1.00 (cap)
    """
    return np.minimum(hours / max_cap, 1.0)


def calculate_interaction_value(df):
    """
    Calcula valor de interação baseado apenas em horas jogadas

    Para o dataset Steam (offline):
    - Todos os reviews aqui são positivos (is_recommended == True)
    - Não temos favoritos ou ratings do usuário (virão do Playlite)
    - Usamos apenas horas como proxy de engajamento

    interaction_value = normalized_hours

    Nota: No Playlite, ao combinar recomendações, aplicaremos pesos adicionais:
    - w_fav × favorite(i) = 1.0 × {0,1}
    - w_rate × rating(i) = 0.6 × (stars-1)/4
    - w_hours × normalized_hours(i) = 0.4 × interaction_value
    """
    print("\n=== Calculando Valores de Interação ===")

    # Normalizar horas
    df['interaction_value'] = normalize_hours(df['hours'])

    # Estatísticas
    print(f"\nEstatísticas de Horas:")
    print(f"  Mínimo: {df['hours'].min():.2f}h")
    print(f"  Máximo: {df['hours'].max():.2f}h")
    print(f"  Média: {df['hours'].mean():.2f}h")
    print(f"  Mediana: {df['hours'].median():.2f}h")
    print(f"  Cap aplicado: {MAX_HOURS_CAP}h")

    print(f"\nEstatísticas de Interaction Value:")
    print(f"  Mínimo: {df['interaction_value'].min():.4f}")
    print(f"  Máximo: {df['interaction_value'].max():.4f}")
    print(f"  Média: {df['interaction_value'].mean():.4f}")
    print(f"  Mediana: {df['interaction_value'].median():.4f}")

    # Distribuição por faixas
    print(f"\nDistribuição de Valores:")
    print(
        f"  [0.0 - 0.2]: {(df['interaction_value'] <= 0.2).sum():,} ({(df['interaction_value'] <= 0.2).sum() / len(df) * 100:.1f}%)")
    print(
        f"  (0.2 - 0.4]: {((df['interaction_value'] > 0.2) & (df['interaction_value'] <= 0.4)).sum():,} ({((df['interaction_value'] > 0.2) & (df['interaction_value'] <= 0.4)).sum() / len(df) * 100:.1f}%)")
    print(
        f"  (0.4 - 0.6]: {((df['interaction_value'] > 0.4) & (df['interaction_value'] <= 0.6)).sum():,} ({((df['interaction_value'] > 0.4) & (df['interaction_value'] <= 0.6)).sum() / len(df) * 100:.1f}%)")
    print(
        f"  (0.6 - 0.8]: {((df['interaction_value'] > 0.6) & (df['interaction_value'] <= 0.8)).sum():,} ({((df['interaction_value'] > 0.6) & (df['interaction_value'] <= 0.8)).sum() / len(df) * 100:.1f}%)")
    print(
        f"  (0.8 - 1.0]: {(df['interaction_value'] > 0.8).sum():,} ({(df['interaction_value'] > 0.8).sum() / len(df) * 100:.1f}%)")

    return df


def build_interactions():
    """Constrói matriz de interações a partir dos reviews filtrados"""
    print("=" * 60)
    print("CONSTRUÇÃO DA MATRIZ DE INTERAÇÕES")
    print("=" * 60)

    # Carregar reviews filtrados
    print("\n=== Carregando Reviews Filtrados ===")
    df = pd.read_parquet(PROCESSED_DIR / "filtered_reviews.parquet")
    log_stats("LOADED", df, "filtered_reviews")

    print(f"\nColunas disponíveis: {list(df.columns)}")
    print(f"Tipos: {df.dtypes.to_dict()}")

    # Calcular valores de interação
    df = calculate_interaction_value(df)

    # Selecionar apenas colunas necessárias
    interactions = df[['user_id', 'app_id', 'interaction_value']].copy()

    # Converter para tipos otimizados
    interactions = interactions.astype({
        'user_id': 'int32',
        'app_id': 'int32',
        'interaction_value': 'float32'
    })

    # Verificar duplicatas (não deveria ter)
    duplicates = interactions.duplicated(subset=['user_id', 'app_id']).sum()
    if duplicates > 0:
        print(f"\nAVISO: {duplicates:,} duplicatas encontradas. Removendo...")
        # Ordenar por interaction_value (descendente) e manter o primeiro (máximo)
        interactions = interactions.sort_values('interaction_value', ascending=False)
        interactions = interactions.drop_duplicates(subset=['user_id', 'app_id'], keep='first')

    # Ordenar para melhor compressão
    interactions = interactions.sort_values(['user_id', 'app_id'])

    # Salvar
    print("\n=== Salvando Matriz de Interações ===")
    output_file = PROCESSED_DIR / "interactions.parquet"
    interactions.to_parquet(
        output_file,
        index=False,
        compression='snappy'
    )

    log_stats("SAVED", interactions, "interactions")

    # Estatísticas finais
    print("\n=== Estatísticas Finais ===")
    unique_users = interactions['user_id'].nunique()
    unique_games = interactions['app_id'].nunique()
    total_interactions = len(interactions)
    sparsity = 1 - (total_interactions / (unique_users * unique_games))

    print(f"  Usuários únicos: {unique_users:,}")
    print(f"  Jogos únicos: {unique_games:,}")
    print(f"  Total de interações: {total_interactions:,}")
    print(f"  Densidade da matriz: {(1 - sparsity) * 100:.4f}%")
    print(f"  Esparsidade: {sparsity * 100:.4f}%")
    print(f"  Média interações/usuário: {total_interactions / unique_users:.2f}")
    print(f"  Média interações/jogo: {total_interactions / unique_games:.2f}")

    # Gerar resumo
    generate_summary(interactions, unique_users, unique_games)

    return interactions


def generate_summary(interactions, unique_users, unique_games):
    """Gera arquivo de resumo com estatísticas da matriz de interações"""
    print("\n=== Gerando Resumo ===")

    total_interactions = len(interactions)
    sparsity = 1 - (total_interactions / (unique_users * unique_games))

    summary = {
        "generated_at": datetime.now().isoformat(),
        "normalization": {
            "max_hours_cap": MAX_HOURS_CAP,
            "method": "min(hours / max_cap, 1.0)"
        },
        "matrix_stats": {
            "unique_users": unique_users,
            "unique_games": unique_games,
            "total_interactions": total_interactions,
            "matrix_density_pct": round((1 - sparsity) * 100, 4),
            "sparsity_pct": round(sparsity * 100, 4),
            "avg_interactions_per_user": round(total_interactions / unique_users, 2),
            "avg_interactions_per_game": round(total_interactions / unique_games, 2)
        },
        "interaction_value_stats": {
            "min": round(float(interactions['interaction_value'].min()), 4),
            "max": round(float(interactions['interaction_value'].max()), 4),
            "mean": round(float(interactions['interaction_value'].mean()), 4),
            "median": round(float(interactions['interaction_value'].median()), 4),
            "std": round(float(interactions['interaction_value'].std()), 4)
        },
        "hours_stats": {
            "min": round(float(interactions['interaction_value'].min() * MAX_HOURS_CAP), 2),
            "max": "capped at " + str(MAX_HOURS_CAP),
            "mean": "represented in interaction_value",
            "note": "Original hours were normalized to [0,1] range"
        },
        "note": "This interaction matrix represents Steam dataset only. User favorites and ratings from Playlite will be applied as weights during tooltips aggregation."
    }

    # Salvar
    with open(REPORTS_DIR / "interactions_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"  Resumo salvo em: {REPORTS_DIR / 'interactions_summary.json'}")


def main():
    """Executa o pipeline de construção de interações"""
    # Criar diretórios de saída se não existirem
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        interactions = build_interactions()

        print("\n" + "=" * 60)
        print("MATRIZ DE INTERAÇÕES CONSTRUÍDA COM SUCESSO!")
        print("=" * 60)
        print(f"\nArquivos gerados em: {PROCESSED_DIR}/")
        print("  - interactions.parquet")
        print("  - interactions_summary.json")

    except Exception as e:
        print(f"\nERRO: {e}")
        raise


if __name__ == "__main__":
    main()

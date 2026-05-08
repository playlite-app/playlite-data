"""
compute_similarity.py - Cálculo de similaridade item-item usando Cosine Similarity

Este script calcula a similaridade entre jogos baseado em coocorrência de usuários.
Usa estratégia eficiente de memória (não gera matriz NxN completa).

Entrada: data/processed/interactions.parquet
Saída: data/processed/similarity_raw.parquet
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configurações - usar caminho relativo baseado na localização do script
SCRIPT_DIR = Path(__file__).parent  # data/scripts
PROCESSED_DIR = SCRIPT_DIR.parent / "processed"  # data/processed
REPORTS_DIR = SCRIPT_DIR.parent / "reports"  # data/reports

# Parâmetros do algoritmo
TOP_K_SIMILAR = 20  # Manter apenas top-K jogos similares por jogo
MIN_SHARED_USERS = 20  # Mínimo de usuários em comum para considerar similaridade
MIN_SIMILARITY_SCORE = 0.1  # Score mínimo para salvar
BATCH_SIZE = 1000  # Processar usuários em batches


def log_progress(message):
    """Log com timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_interactions():
    """Carrega matriz de interações"""
    log_progress("Carregando matriz de interações...")

    df = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")

    print(f"  Total de interações: {len(df):,}")
    print(f"  Usuários únicos: {df['user_id'].nunique():,}")
    print(f"  Jogos únicos: {df['app_id'].nunique():,}")
    print(f"  Memória: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    return df


def build_user_game_dict(interactions):
    """
    Constrói dicionário: user_id -> [(app_id, interaction_value), ...]
    Mais eficiente que agrupar pandas para iteração
    """
    log_progress("Construindo índice usuário -> jogos...")

    user_games = defaultdict(list)

    for _, row in tqdm(interactions.iterrows(), total=len(interactions), desc="Indexando"):
        user_games[row['user_id']].append((row['app_id'], row['interaction_value']))

    print(f"  Usuários indexados: {len(user_games):,}")
    print(f"  Média jogos/usuário: {sum(len(games) for games in user_games.values()) / len(user_games):.2f}")

    return user_games


def compute_game_norms(interactions):
    """
    Calcula norma L2 de cada jogo (para cosseno)
    ||game|| = sqrt(sum(interaction_value^2))
    """
    log_progress("Calculando normas dos jogos...")

    norms = interactions.groupby('app_id')['interaction_value'].apply(
        lambda x: np.sqrt((x ** 2).sum())
    ).to_dict()

    print(f"  Normas calculadas para {len(norms):,} jogos")

    return norms


def compute_pairwise_similarities(user_games, game_norms):
    """
    Calcula similaridade entre pares de jogos

    Estratégia eficiente:
    1. Para cada usuário, pegar todos os pares de jogos que ele avaliou
    2. Acumular produto escalar entre esses pares
    3. Ao final, dividir pelas normas (cosseno)

    Isso evita matriz NxN completa
    """
    log_progress("Calculando similaridades par-a-par...")

    # Estruturas para acumular
    pair_dot_product = defaultdict(float)  # (app_id_1, app_id_2) -> soma de produtos
    pair_shared_users = defaultdict(int)  # (app_id_1, app_id_2) -> count de usuários

    # Processar por usuário
    total_users = len(user_games)
    processed = 0

    print(f"  Processando {total_users:,} usuários...")

    for user_id, games in tqdm(user_games.items(), desc="Computando pares", total=total_users):
        # Pegar todos os pares de jogos deste usuário
        for i in range(len(games)):
            app_id_1, value_1 = games[i]

            for j in range(i + 1, len(games)):
                app_id_2, value_2 = games[j]

                # Garantir ordem consistente (menor app_id primeiro)
                if app_id_1 > app_id_2:
                    app_id_1, app_id_2 = app_id_2, app_id_1
                    value_1, value_2 = value_2, value_1

                # Acumular produto escalar
                pair_key = (app_id_1, app_id_2)
                pair_dot_product[pair_key] += value_1 * value_2
                pair_shared_users[pair_key] += 1

        processed += 1

        # Log de progresso a cada 100k usuários
        if processed % 100000 == 0:
            print(f"    Processados: {processed:,}/{total_users:,} usuários ({processed / total_users * 100:.1f}%)")
            print(f"    Pares encontrados até agora: {len(pair_dot_product):,}")

    log_progress(f"Pares de jogos encontrados: {len(pair_dot_product):,}")

    # Filtrar pares com poucos usuários em comum
    log_progress(f"Filtrando pares com menos de {MIN_SHARED_USERS} usuários em comum...")

    filtered_pairs = {
        pair: dot_prod
        for pair, dot_prod in pair_dot_product.items()
        if pair_shared_users[pair] >= MIN_SHARED_USERS
    }

    removed = len(pair_dot_product) - len(filtered_pairs)
    print(f"  Removidos: {removed:,} pares ({removed / len(pair_dot_product) * 100:.1f}%)")
    print(f"  Restantes: {len(filtered_pairs):,} pares")

    # Calcular similaridade cosine para cada par
    log_progress("Calculando scores de similaridade cosine...")

    similarities = []

    for (app_id_1, app_id_2), dot_product in tqdm(filtered_pairs.items(), desc="Cosine"):
        norm_1 = game_norms.get(app_id_1, 1.0)
        norm_2 = game_norms.get(app_id_2, 1.0)

        # Cosine similarity
        cosine_sim = dot_product / (norm_1 * norm_2) if (norm_1 * norm_2) > 0 else 0.0

        # Shared users
        shared_users = pair_shared_users[(app_id_1, app_id_2)]

        # Confidence (baseado em usuários em comum)
        confidence = min(shared_users / 100, 1.0)

        # Só salvar se passar no threshold
        if cosine_sim >= MIN_SIMILARITY_SCORE:
            similarities.append({
                'app_id_1': app_id_1,
                'app_id_2': app_id_2,
                'cosine_similarity': cosine_sim,
                'shared_users': shared_users,
                'confidence': confidence
            })

    log_progress(f"Similaridades calculadas: {len(similarities):,}")

    return pd.DataFrame(similarities)


def apply_popularity_penalty(similarities_df, interactions):
    """
    Aplica penalização por popularidade

    Fórmula: popularity_penalty = log(shared_users + 1) / log(max_shared_users + 1)

    Isso reduz o score de pares extremamente populares
    """
    log_progress("Aplicando penalização por popularidade...")

    max_shared = similarities_df['shared_users'].max()
    print(f"  Máximo de usuários compartilhados: {max_shared:,}")

    similarities_df['popularity_penalty'] = (
            np.log(similarities_df['shared_users'] + 1) / np.log(max_shared + 1)
    )

    # Score final
    similarities_df['final_score'] = (
            similarities_df['cosine_similarity'] *
            similarities_df['popularity_penalty'] *
            similarities_df['confidence']
    )

    print(f"  Score final - Média: {similarities_df['final_score'].mean():.4f}")
    print(f"  Score final - Mediana: {similarities_df['final_score'].median():.4f}")
    print(f"  Score final - Min: {similarities_df['final_score'].min():.4f}")
    print(f"  Score final - Max: {similarities_df['final_score'].max():.4f}")

    return similarities_df


def keep_top_k_per_game(similarities_df):
    """
    Mantém apenas top-K similares para cada jogo

    Como a matriz é simétrica (A-B e B-A), precisamos:
    1. Duplicar cada par (fazer simétrico)
    2. Agrupar por jogo
    3. Pegar top-K
    """
    log_progress(f"Selecionando top-{TOP_K_SIMILAR} similares por jogo...")

    # Criar entradas simétricas
    # Para cada par (A, B), criar também (B, A)
    symmetric_rows = []

    for _, row in tqdm(similarities_df.iterrows(), total=len(similarities_df), desc="Simetrizando"):
        # Original: A -> B
        symmetric_rows.append({
            'app_id': row['app_id_1'],
            'similar_app_id': row['app_id_2'],
            'cosine_similarity': row['cosine_similarity'],
            'shared_users': row['shared_users'],
            'confidence': row['confidence'],
            'popularity_penalty': row['popularity_penalty'],
            'final_score': row['final_score']
        })

        # Simétrico: B -> A
        symmetric_rows.append({
            'app_id': row['app_id_2'],
            'similar_app_id': row['app_id_1'],
            'cosine_similarity': row['cosine_similarity'],
            'shared_users': row['shared_users'],
            'confidence': row['confidence'],
            'popularity_penalty': row['popularity_penalty'],
            'final_score': row['final_score']
        })

    symmetric_df = pd.DataFrame(symmetric_rows)

    print(f"  Entradas simétricas criadas: {len(symmetric_df):,}")

    # Ordenar por score (descendente) e pegar top-K por jogo
    symmetric_df = symmetric_df.sort_values(['app_id', 'final_score'], ascending=[True, False])

    top_k_df = symmetric_df.groupby('app_id').head(TOP_K_SIMILAR).reset_index(drop=True)

    print(f"  Após filtrar top-{TOP_K_SIMILAR}: {len(top_k_df):,} entradas")

    # Estatísticas
    games_with_similar = top_k_df['app_id'].nunique()
    avg_similar_per_game = len(top_k_df) / games_with_similar if games_with_similar > 0 else 0

    print(f"  Jogos com similares: {games_with_similar:,}")
    print(f"  Média de similares por jogo: {avg_similar_per_game:.2f}")

    return top_k_df


def save_results(similarities_df):
    """Salva resultados em formato otimizado"""
    log_progress("Salvando resultados...")

    # Converter para tipos otimizados
    similarities_df = similarities_df.astype({
        'app_id': 'int32',
        'similar_app_id': 'int32',
        'cosine_similarity': 'float32',
        'shared_users': 'int32',
        'confidence': 'float32',
        'popularity_penalty': 'float32',
        'final_score': 'float32'
    })

    output_file = PROCESSED_DIR / "similarity_raw.parquet"
    similarities_df.to_parquet(
        output_file,
        index=False,
        compression='snappy'
    )

    print(f"  Arquivo salvo: {output_file}")
    print(f"  Tamanho: {output_file.stat().st_size / 1024 ** 2:.2f} MB")

    return similarities_df


def generate_summary(similarities_df, total_pairs_computed):
    """Gera resumo com estatísticas do processamento"""
    log_progress("Gerando resumo...")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "top_k_similar": TOP_K_SIMILAR,
            "min_shared_users": MIN_SHARED_USERS,
            "min_similarity_score": MIN_SIMILARITY_SCORE
        },
        "computation_stats": {
            "unique_pairs": total_pairs_computed,
            "symmetric_entries": len(similarities_df),
            "reduction_pct": round((1 - len(similarities_df) / total_pairs_computed) * 100,
                                   2) if total_pairs_computed > 0 else 0,
            "games_with_similarities": int(similarities_df['app_id'].nunique()),
            "avg_similarities_per_game": round(len(similarities_df) / similarities_df['app_id'].nunique(), 2) if
            similarities_df['app_id'].nunique() > 0 else 0
        },
        "score_statistics": {
            "cosine_similarity": {
                "min": round(float(similarities_df['cosine_similarity'].min()), 4),
                "max": round(float(similarities_df['cosine_similarity'].max()), 4),
                "mean": round(float(similarities_df['cosine_similarity'].mean()), 4),
                "median": round(float(similarities_df['cosine_similarity'].median()), 4)
            },
            "final_score": {
                "min": round(float(similarities_df['final_score'].min()), 4),
                "max": round(float(similarities_df['final_score'].max()), 4),
                "mean": round(float(similarities_df['final_score'].mean()), 4),
                "median": round(float(similarities_df['final_score'].median()), 4)
            },
            "shared_users": {
                "min": int(similarities_df['shared_users'].min()),
                "max": int(similarities_df['shared_users'].max()),
                "mean": round(float(similarities_df['shared_users'].mean()), 2),
                "median": int(similarities_df['shared_users'].median())
            }
        }
    }

    with open(REPORTS_DIR / "similarity_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"  Resumo salvo: {REPORTS_DIR / 'similarity_summary.json'}")


def main():
    """Pipeline completo de cálculo de similaridade"""
    print("=" * 70)
    print("CÁLCULO DE SIMILARIDADE ITEM-ITEM (COSINE SIMILARITY)")
    print("=" * 70)
    print()

    # Criar diretórios de saída se não existirem
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Carregar interações
        interactions = load_interactions()
        print()

        # 2. Construir índice user -> games
        user_games = build_user_game_dict(interactions)
        print()

        # 3. Calcular normas dos jogos
        game_norms = compute_game_norms(interactions)
        print()

        # 4. Computar similaridades par-a-par
        similarities_df = compute_pairwise_similarities(user_games, game_norms)
        total_pairs = len(similarities_df)
        print()

        # 5. Aplicar penalização por popularidade
        similarities_df = apply_popularity_penalty(similarities_df, interactions)
        print()

        # 6. Manter apenas top-K por jogo
        similarities_df = keep_top_k_per_game(similarities_df)
        print()

        # 7. Salvar resultados
        similarities_df = save_results(similarities_df)
        print()

        # 8. Gerar resumo
        generate_summary(similarities_df, total_pairs)
        print()

        print("=" * 70)
        print("SIMILARIDADES CALCULADAS COM SUCESSO!")
        print("=" * 70)
        print()
        print(f"Arquivos gerados em: {PROCESSED_DIR}/")
        print("  - similarity_raw.parquet")

    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

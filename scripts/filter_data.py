"""
filter_data.py - Filtragem inicial dos dados brutos do dataset Steam

Este script aplica os filtros de qualidade definidos para:
- Jogos: user_reviews >= 100 AND positive_ratio >= 70
- Usuários: reviews >= 2
- Reviews: is_recommended == True AND hours >= 2

Entrada: data/raw/
Saída: data/processed/
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configurações - usar caminho relativo baseado na localização do script
SCRIPT_DIR = Path(__file__).parent  # data/scripts
RAW_DIR = SCRIPT_DIR.parent / "raw"  # data/raw
PROCESSED_DIR = SCRIPT_DIR.parent / "processed"  # data/processed
REPORTS_DIR = SCRIPT_DIR.parent / "reports"  # data/reports

# Filtros definidos
MIN_USER_REVIEWS = 100
MIN_POSITIVE_RATIO = 70
MIN_USER_REVIEWS_PER_USER = 2
MIN_HOURS_PLAYED = 2

# Whitelist de gêneros principais
GENRE_TAGS = {
    "Action", "Adventure", "RPG", "Strategy",
    "Simulation", "Indie", "Puzzle", "Shooter",
    "Sports", "Racing", "Horror", "Platformer",
    "Fighting", "Survival", "Sandbox", "MMORPG"
}


def log_stats(stage, df, name):
    """Imprime estatísticas para debug"""
    print(f"[{stage}] {name}: {len(df):,} registros")


def load_metadata_entries(path: Path):
    """Carrega metadata suportando JSON array ou JSONL."""
    try:
        with path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  Formato array inválido em {path.name}: {exc}; tentando JSONL...")

    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    entries.append(obj)
                else:
                    print(f"  Linha {line_no} ignorada: esperado objeto JSON, obtido {type(obj)}")
            except json.JSONDecodeError as exc:
                print(f"  Linha {line_no} ignorada: {exc}")
    return entries


def filter_games():
    """Filtra jogo baseado em volume e qualidade de reviews"""
    print("\n=== Filtrando Jogos ===")

    # Carregar apenas colunas necessárias
    columns_to_use = ['app_id', 'title', 'user_reviews', 'positive_ratio']
    df = pd.read_csv(
        RAW_DIR / "games.csv",
        usecols=columns_to_use,
        dtype={
            'app_id': 'int32',
            'user_reviews': 'int32',
            'positive_ratio': 'float32'
        }
    )
    log_stats("LOADED", df, "games")

    # Aplicar filtros
    filtered = df[
        (df['user_reviews'] >= MIN_USER_REVIEWS) &
        (df['positive_ratio'] >= MIN_POSITIVE_RATIO)
        ].copy()

    log_stats("FILTERED", filtered, "games")
    print(f"  Removidos: {len(df) - len(filtered):,} jogos ({(1 - len(filtered) / len(df)) * 100:.1f}%)")

    # Salvar
    filtered.to_parquet(
        PROCESSED_DIR / "filtered_games.parquet",
        index=False,
        compression='snappy'
    )

    return set(filtered['app_id']), len(df)


def filter_users():
    """Filtra usuários baseado em número mínimo de reviews"""
    print("\n=== Filtrando Usuários ===")

    # Carregar apenas colunas necessárias
    columns_to_use = ['user_id', 'reviews']
    df = pd.read_csv(
        RAW_DIR / "users.csv",
        usecols=columns_to_use,
        dtype={
            'user_id': 'int32',
            'reviews': 'int32'
        }
    )
    log_stats("LOADED", df, "users")

    # Aplicar filtro
    filtered = df[df['reviews'] >= MIN_USER_REVIEWS_PER_USER].copy()

    log_stats("FILTERED", filtered, "users")
    print(f"  Removidos: {len(df) - len(filtered):,} usuários ({(1 - len(filtered) / len(df)) * 100:.1f}%)")

    # Salvar
    filtered.to_parquet(
        PROCESSED_DIR / "filtered_users.parquet",
        index=False,
        compression='snappy'
    )

    return set(filtered['user_id']), len(df)


def filter_reviews(valid_games, valid_users):
    """Filtra reviews baseado em recomendação positiva e horas jogadas"""
    print("\n=== Filtrando Reviews ===")

    # Carregar apenas colunas necessárias
    columns_to_use = ['app_id', 'user_id', 'is_recommended', 'hours']

    # Ler em chunks para lidar com arquivo grande
    chunk_size = 500_000
    filtered_chunks = []
    total_loaded = 0

    print("  Processando em chunks...")
    for chunk in pd.read_csv(
            RAW_DIR / "recommendations.csv",
            usecols=columns_to_use,
            dtype={
                'app_id': 'int32',
                'user_id': 'int32',
                'is_recommended': 'bool',
                'hours': 'float32'
            },
            chunksize=chunk_size
    ):
        total_loaded += len(chunk)

        # Aplicar todos os filtros no chunk
        filtered_chunk = chunk[
            (chunk['is_recommended'] == True) &
            (chunk['hours'] >= MIN_HOURS_PLAYED) &
            (chunk['app_id'].isin(valid_games)) &
            (chunk['user_id'].isin(valid_users))
            ].copy()

        if len(filtered_chunk) > 0:
            filtered_chunks.append(filtered_chunk)

        # Progresso a cada 5M reviews
        if total_loaded % 5_000_000 == 0:
            print(f"    Processados: {total_loaded:,} reviews")

    # Concatenar todos os chunks
    filtered = pd.concat(filtered_chunks, ignore_index=True)

    log_stats("LOADED", pd.DataFrame({'dummy': range(total_loaded)}), "reviews")
    log_stats("FILTERED", filtered, "reviews")
    print(f"  Removidos: {total_loaded - len(filtered):,} reviews ({(1 - len(filtered) / total_loaded) * 100:.1f}%)")

    # Remover coluna is_recommended (todos são True agora)
    filtered = filtered.drop(columns=['is_recommended'])

    # Salvar
    filtered.to_parquet(
        PROCESSED_DIR / "filtered_reviews.parquet",
        index=False,
        compression='snappy'
    )

    return filtered, total_loaded


def filter_metadata(valid_games):
    """Filtra e processa metadata dos jogos"""
    print("\n=== Filtrando Metadata ===")

    metadata = load_metadata_entries(RAW_DIR / "games_metadata.json")

    if not metadata:
        raise ValueError(
            "Nenhuma entrada de metadata foi carregada. Verifique o formato de games_metadata.json (JSON array ou JSONL de objetos por linha).")

    original_count = len(metadata)
    log_stats("LOADED", metadata, "metadata entries")

    # Filtrar e processar
    filtered_metadata = {}

    for item in metadata:
        app_id = str(item.get('app_id'))

        # Verificar se o jogo passou nos filtros
        if int(app_id) not in valid_games:
            continue

        # Extrair tags
        tags = item.get('tags', [])

        # Extrair gêneros principais (interseção com whitelist)
        genres = [tag for tag in tags if tag in GENRE_TAGS]

        filtered_metadata[app_id] = {
            'tags': tags[:10],  # Limitar a 10 tags principais
            'genres': genres[:3]  # Limitar a 3 gêneros principais
        }

    log_stats("FILTERED", filtered_metadata, "metadata entries")
    print(
        f"  Removidos: {len(metadata) - len(filtered_metadata):,} entradas ({(1 - len(filtered_metadata) / len(metadata)) * 100:.1f}%)")

    # Salvar
    with open(PROCESSED_DIR / "filtered_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(filtered_metadata, f, indent=2, ensure_ascii=False)

    return filtered_metadata, original_count


def generate_summary(valid_games, valid_users, filtered_reviews, metadata,
                     original_games, original_users, original_reviews, original_metadata):
    """Gera arquivo de resumo com estatísticas do processamento"""
    print("\n=== Gerando Resumo ===")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "filters_applied": {
            "min_user_reviews": MIN_USER_REVIEWS,
            "min_positive_ratio": MIN_POSITIVE_RATIO,
            "min_user_reviews_per_user": MIN_USER_REVIEWS_PER_USER,
            "min_hours_played": MIN_HOURS_PLAYED
        },
        "raw_data": {
            "total_games": original_games,
            "total_users": original_users,
            "total_reviews": original_reviews,
            "total_metadata_entries": original_metadata
        },
        "filtered_data": {
            "total_games": len(valid_games),
            "total_users": len(valid_users),
            "total_reviews": len(filtered_reviews),
            "total_metadata_entries": len(metadata)
        },
        "reduction": {
            "games_removed": original_games - len(valid_games),
            "games_removed_pct": round((1 - len(valid_games) / original_games) * 100, 1),
            "users_removed": original_users - len(valid_users),
            "users_removed_pct": round((1 - len(valid_users) / original_users) * 100, 1),
            "reviews_removed": original_reviews - len(filtered_reviews),
            "reviews_removed_pct": round((1 - len(filtered_reviews) / original_reviews) * 100, 1),
            "metadata_removed": original_metadata - len(metadata),
            "metadata_removed_pct": round((1 - len(metadata) / original_metadata) * 100, 1)
        },
        "statistics": {
            "avg_reviews_per_game": round(len(filtered_reviews) / len(valid_games), 2) if valid_games else 0,
            "avg_reviews_per_user": round(len(filtered_reviews) / len(valid_users), 2) if valid_users else 0,
            "avg_hours_per_review": round(float(filtered_reviews['hours'].mean()), 2) if len(
                filtered_reviews) > 0 else 0
        }
    }

    # Salvar
    with open(REPORTS_DIR / "filter_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\nDados Originais:")
    print(f"  Jogos: {summary['raw_data']['total_games']:,}")
    print(f"  Usuários: {summary['raw_data']['total_users']:,}")
    print(f"  Reviews: {summary['raw_data']['total_reviews']:,}")
    print(f"  Metadata: {summary['raw_data']['total_metadata_entries']:,}")

    print("\nDados Filtrados:")
    print(
        f"  Jogos válidos: {summary['filtered_data']['total_games']:,} (-{summary['reduction']['games_removed_pct']}%)")
    print(
        f"  Usuários válidos: {summary['filtered_data']['total_users']:,} (-{summary['reduction']['users_removed_pct']}%)")
    print(
        f"  Reviews válidas: {summary['filtered_data']['total_reviews']:,} (-{summary['reduction']['reviews_removed_pct']}%)")
    print(
        f"  Metadata válido: {summary['filtered_data']['total_metadata_entries']:,} (-{summary['reduction']['metadata_removed_pct']}%)")

    print("\nEstatísticas:")
    print(f"  Média reviews/jogo: {summary['statistics']['avg_reviews_per_game']:.1f}")
    print(f"  Média reviews/usuário: {summary['statistics']['avg_reviews_per_user']:.1f}")
    print(f"  Média horas/review: {summary['statistics']['avg_hours_per_review']:.1f}")

    print(f"\n Resumo salvo em: {REPORTS_DIR / 'filter_sumary.json'}")


def main():
    """Executa o pipeline completo de filtragem"""
    print("=" * 60)
    print("PIPELINE DE FILTRAGEM - COLLABORATIVE FILTERING")
    print("=" * 60)

    # Criar diretório de saída se não existir
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Criar diretório de relatórios se não existir
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Executar filtragens na ordem correta e coletar contagens originais
    valid_games, original_games = filter_games()
    valid_users, original_users = filter_users()
    filtered_reviews, original_reviews = filter_reviews(valid_games, valid_users)
    metadata, original_metadata = filter_metadata(valid_games)

    # Gerar resumo com dados originais e filtrados
    generate_summary(
        valid_games, valid_users, filtered_reviews, metadata,
        original_games, original_users, original_reviews, original_metadata
    )

    print("\n" + "=" * 60)
    print("FILTRAGEM CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    print(f"\nArquivos gerados em: {PROCESSED_DIR}/")
    print("  - filtered_games.parquet")
    print("  - filtered_users.parquet")
    print("  - filtered_reviews.parquet")
    print("  - filtered_metadata.json")


if __name__ == "__main__":
    main()

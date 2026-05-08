"""
export_json.py - Exporta similaridades item-item para JSON consumível pelo backend Rust

Entrada:
- data/processed/similarity_raw.parquet

Saída:
- data/output/collaborative_index.json

Características:
- Baseado em steam_app_id
- Lookup O(1) (HashMap no Rust)
- Scores normalizados por jogo
- Limite top-K por jogo
- Sem dados do usuário
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
REPORTS_DIR = DATA_DIR / "reports"

# Configuração
TOP_K = 10  # Máx. de similares por jogo
MIN_SCORE = 0.01  # Score mínimo (proteção extra)
VERSION = "cf-item-item-v1"
SOURCE = "steam"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_similarity() -> pd.DataFrame:
    """Carrega matriz de similaridade pré-computada"""
    log("Carregando similarity_raw.parquet...")

    path = PROCESSED_DIR / "similarity_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Execute compute_similarity.py primeiro."
        )

    df = pd.read_parquet(path)

    print(f"  Entradas totais: {len(df):,}")
    print(f"  Jogos origem: {df['app_id'].nunique():,}")
    print(f"  Jogos similares: {df['similar_app_id'].nunique():,}")
    print(f"  Score médio: {df['final_score'].mean():.4f}")
    print(f"  Score mínimo: {df['final_score'].min():.4f}")
    print(f"  Score máximo: {df['final_score'].max():.4f}")

    return df


def normalize_scores(group: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza scores dentro de cada jogo origem (0–1)

    Estratégia: divide pelo score máximo do grupo
    Benefício: evita viés de jogos com muitos usuários
    """
    max_score = group["final_score"].max()

    if max_score <= 0:
        group["score_norm"] = 0.0
    else:
        group["score_norm"] = group["final_score"] / max_score

    return group


def build_index(df: pd.DataFrame) -> dict:
    """
    Constrói índice item-item otimizado para Rust

    Retorna: { "app_id": [{"app_id": X, "score": Y}, ...] }
    """
    log("Construindo índice item-item...")

    index = {}
    total_pairs = 0
    games_filtered = 0

    # Agrupar por jogo origem
    for app_id, group in df.groupby("app_id"):
        # Filtro de score mínimo
        original_size = len(group)
        group = group[group["final_score"] >= MIN_SCORE]

        if group.empty:
            games_filtered += 1
            continue

        # Ordenar por score (descendente)
        group = group.sort_values("final_score", ascending=False)

        # Normalizar scores por jogo
        group = normalize_scores(group)

        # Limitar top-K
        group = group.head(TOP_K)

        # Construir lista de similares
        items = []
        for _, row in group.iterrows():
            items.append({
                "app_id": int(row["similar_app_id"]),
                "score": round(float(row["score_norm"]), 4),
            })

        if items:
            index[str(int(app_id))] = items
            total_pairs += len(items)

    print(f"  Jogos com similares: {len(index):,}")
    print(f"  Jogos filtrados (score < {MIN_SCORE}): {games_filtered:,}")
    print(f"  Total de pares exportados: {total_pairs:,}")
    print(f"  Média de similares por jogo: {total_pairs / max(len(index), 1):.2f}")

    return index


def calculate_stats(index: dict) -> dict:
    """Calcula estatísticas do índice gerado"""
    if not index:
        return {
            "total_games": 0,
            "total_pairs": 0,
            "avg_per_game": 0,
            "min_per_game": 0,
            "max_per_game": 0
        }

    counts = [len(similars) for similars in index.values()]

    return {
        "total_games": len(index),
        "total_pairs": sum(counts),
        "avg_per_game": round(sum(counts) / len(counts), 2),
        "min_per_game": min(counts),
        "max_per_game": max(counts)
    }


def export_json(index: dict):
    """Exporta índice para JSON com metadados"""
    log("Exportando para JSON...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "collaborative_index.json"

    stats = calculate_stats(index)

    payload = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "parameters": {
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "normalization": "per_game_max"
        },
        "index": index
    }

    # Salvar JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Estatísticas do arquivo
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    log(f" Arquivo exportado: {output_path}")
    print(f"  Tamanho: {file_size_mb:.2f} MB")
    print(f"  Jogos: {stats['total_games']:,}")
    print(f"  Pares: {stats['total_pairs']:,}")
    print(f"  Média/jogo: {stats['avg_per_game']:.1f}")


def validate_index(index: dict):
    """Validação básica do índice gerado"""
    log("Validando índice...")

    issues = []

    # Verificar se há jogos
    if not index:
        issues.append("  Índice vazio!")

    # Verificar estrutura de alguns jogos
    sample_size = min(10, len(index))
    for app_id, similars in list(index.items())[:sample_size]:
        # Verificar se é lista
        if not isinstance(similars, list):
            issues.append(f"  app_id {app_id}: similares não é lista")
            continue

        # Verificar estrutura de cada similar
        for similar in similars:
            if "app_id" not in similar or "score" not in similar:
                issues.append(f"  app_id {app_id}: similar sem campos obrigatórios")
                break

            # Verificar tipos
            if not isinstance(similar["app_id"], int):
                issues.append(f"  app_id {app_id}: similar app_id não é int")
                break

            if not isinstance(similar["score"], (int, float)):
                issues.append(f"  app_id {app_id}: score não é numérico")
                break

            # Verificar range do score
            if not 0 <= similar["score"] <= 1:
                issues.append(f"  app_id {app_id}: score fora do range [0,1]")
                break

    if issues:
        print("\n".join(issues))
    else:
        print("   Estrutura válida")


def generate_summary(df: pd.DataFrame, index: dict):
    """Gera relatório detalhado da exportação"""
    log("Gerando relatório...")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Calcular estatísticas
    counts = [len(similars) for similars in index.values()]

    # Estatísticas de scores do índice exportado
    all_scores = []
    for similars in index.values():
        all_scores.extend([s["score"] for s in similars])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": "similarity_raw.parquet",
        "output_file": "collaborative_index.json",
        "version": VERSION,
        "parameters": {
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "normalization": "per_game_max"
        },
        "input_data": {
            "total_entries": len(df),
            "unique_games_origin": int(df['app_id'].nunique()),
            "unique_games_similar": int(df['similar_app_id'].nunique()),
            "score_stats": {
                "min": round(float(df['final_score'].min()), 4),
                "max": round(float(df['final_score'].max()), 4),
                "mean": round(float(df['final_score'].mean()), 4),
                "median": round(float(df['final_score'].median()), 4)
            }
        },
        "output_data": {
            "games_with_similars": len(index),
            "games_filtered": int(df['app_id'].nunique()) - len(index),
            "total_pairs_exported": sum(counts),
            "coverage_pct": round(len(index) / df['app_id'].nunique() * 100, 2) if df['app_id'].nunique() > 0 else 0,
            "similars_per_game": {
                "min": min(counts) if counts else 0,
                "max": max(counts) if counts else 0,
                "mean": round(sum(counts) / len(counts), 2) if counts else 0,
                "median": round(sorted(counts)[len(counts) // 2], 2) if counts else 0
            },
            "normalized_scores": {
                "min": round(min(all_scores), 4) if all_scores else 0,
                "max": round(max(all_scores), 4) if all_scores else 0,
                "mean": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0,
                "median": round(sorted(all_scores)[len(all_scores) // 2], 4) if all_scores else 0
            }
        },
        "file_stats": {
            "size_mb": round((OUTPUT_DIR / "collaborative_index.json").stat().st_size / (1024 * 1024), 2),
            "estimated_memory_rust_mb": round(len(index) * 0.001 + sum(counts) * 0.02, 2)
        },
        "quality_metrics": {
            "avg_similars_per_game": round(sum(counts) / len(counts), 2) if counts else 0,
            "games_with_full_topk": sum(1 for c in counts if c == TOP_K),
            "games_with_full_topk_pct": round(sum(1 for c in counts if c == TOP_K) / len(counts) * 100,
                                              2) if counts else 0
        },
        "recommendations": []
    }

    # Adicionar recomendações baseadas nos resultados
    avg_similars = summary["output_data"]["similars_per_game"]["mean"]
    coverage = summary["output_data"]["coverage_pct"]

    if avg_similars < 2:
        summary["recommendations"].append({
            "severity": "warning",
            "message": f"Média de similares muito baixa ({avg_similars:.1f}). Considere reduzir MIN_SCORE ou ajustar confidence no compute_similarity.py"
        })

    if coverage < 70:
        summary["recommendations"].append({
            "severity": "warning",
            "message": f"Cobertura baixa ({coverage:.1f}%). Muitos jogos sem similares. Considere reduzir thresholds."
        })

    if avg_similars >= 2 and coverage >= 70:
        summary["recommendations"].append({
            "severity": "success",
            "message": "Qualidade excelente! Boa cobertura e quantidade de similares."
        })

    # Salvar relatório
    report_path = REPORTS_DIR / "export_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"   Relatório salvo: {report_path}")

    # Mostrar resumo no terminal
    print(f"\n Resumo da Exportação:")
    print(f"  Cobertura: {coverage:.1f}% ({len(index):,} de {df['app_id'].nunique():,} jogos)")
    print(f"  Média similares/jogo: {avg_similars:.2f}")
    print(f"  Total de pares: {sum(counts):,}")
    print(f"  Tamanho do arquivo: {summary['file_stats']['size_mb']:.2f} MB")

    if summary["recommendations"]:
        print(f"\n Recomendações:")
        for rec in summary["recommendations"]:
            print(f"{rec['message']}")

    return summary


def main():
    """Pipeline completo de exportação"""
    print("=" * 70)
    print("EXPORTAÇÃO CF ITEM-ITEM → JSON (PLAYLITE)")
    print("=" * 70)
    print()

    try:
        # 1. Carregar similaridades
        df = load_similarity()
        print()

        # 2. Construir índice
        index = build_index(df)
        print()

        # 3. Validar
        validate_index(index)
        print()

        # 4. Exportar
        export_json(index)

        # 5. Gerar relatório
        generate_summary(df, index)

        print("\n" + "=" * 70)
        print(" EXPORTAÇÃO CONCLUÍDA COM SUCESSO")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n ERRO: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

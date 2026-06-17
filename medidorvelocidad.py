"""
benchmark_aqr.py
Compara Baseline HNSW vs AQR-HNSW (solo re-ranking stage) en tiempo y recall.

Uso:
    python benchmark_aqr.py

Requiere:
    pip install hnswlib numpy scikit-learn
    (o el fork compilado de chroma-core/hnswlib con set_aqr_params)
"""

import time
import numpy as np
import hnswlib
from sklearn.neighbors import NearestNeighbors

# ─── Configuración ───────────────────────────────────────────────────────────
DIM          = 128       # dimensión de los vectores
N_ELEMENTS   = 50_000   # vectores en el índice
N_QUERIES    = 500       # queries de benchmark
K            = 10        # top-k vecinos
EF_SEARCH    = 100        # ef para baseline (>= K)
M            = 32        # parámetro M del grafo HNSW
EF_CONSTRUCT = 400       # ef_construction

# Parámetros AQR (ajustar según el paper: Nc=45-70, tau_gap=0.01-0.03, etc.)
AQR_TAU_GAP   = 0.01
AQR_TAU_RATIO = 1.008
AQR_N_RERANK  = 28
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
data    = np.random.rand(N_ELEMENTS, DIM).astype(np.float32)
queries = np.random.rand(N_QUERIES,  DIM).astype(np.float32)

# ── Ground truth con brute-force ─────────────────────────────────────────────
print("Calculando ground truth (brute force)...")
t0 = time.perf_counter()
nn = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="l2", n_jobs=-1)
nn.fit(data)
_, gt_labels = nn.kneighbors(queries)
print(f"  Ground truth listo en {time.perf_counter()-t0:.2f}s\n")


def build_index():
    """Construye y retorna un índice HNSW con los datos."""
    idx = hnswlib.Index(space="l2", dim=DIM)
    idx.init_index(max_elements=N_ELEMENTS, ef_construction=EF_CONSTRUCT, M=M)
    idx.add_items(data)
    return idx


def recall_at_k(predicted, ground_truth, k):
    """Recall@k promedio sobre todas las queries."""
    hits = [
        len(np.intersect1d(predicted[i], ground_truth[i])) / k
        for i in range(len(predicted))
    ]
    return float(np.mean(hits))


def run_query(idx, queries, k, n_runs=3):
    """Corre knn_query n_runs veces y retorna (labels, qps_promedio)."""
    labels = None
    times  = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        labels, _ = idx.knn_query(queries, k=k)
        times.append(time.perf_counter() - t0)
    best_time = min(times)  # mejor de n_runs para reducir ruido
    qps = len(queries) / best_time
    return labels, qps


# ── Baseline HNSW ─────────────────────────────────────────────────────────────
print("=" * 50)
print("BASELINE HNSW")
idx_base = build_index()
idx_base.set_ef(100)

labels_base, qps_base = run_query(idx_base, queries, K)
recall_base = recall_at_k(labels_base, gt_labels, K)

print(f"  QPS     : {qps_base:>10.1f}")
print(f"  Recall@{K}: {recall_base:>10.4f}")


# ── AQR-HNSW (re-ranking activado) ────────────────────────────────────────────
print()
print("=" * 50)
print("AQR-HNSW (re-ranking stage)")

idx_aqr = build_index()
# ef más alto para tener más candidatos antes del re-ranking
idx_aqr.set_ef(150)

# Activar re-ranking — este método requiere el fork compilado con los cambios
try:
    idx_aqr.set_aqr_params(
        tau_gap   = AQR_TAU_GAP,
        tau_ratio = AQR_TAU_RATIO,
        n_rerank  = AQR_N_RERANK,
        enabled   = True,
    )
    aqr_available = True
except AttributeError:
    print("  [!] set_aqr_params no disponible — compilar el fork modificado primero.")
    print("      Se muestra solo el efecto de ef más alto como aproximación.\n")
    aqr_available = False

labels_aqr, qps_aqr = run_query(idx_aqr, queries, K)
recall_aqr = recall_at_k(labels_aqr, gt_labels, K)

print(f"  QPS     : {qps_aqr:>10.1f}")
print(f"  Recall@{K}: {recall_aqr:>10.4f}")


# ── Resumen ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("RESUMEN COMPARATIVO")
print(f"  {'Método':<20} {'QPS':>10} {'Recall@'+str(K):>12}")
print(f"  {'-'*44}")
print(f"  {'Baseline HNSW':<20} {qps_base:>10.1f} {recall_base:>12.4f}")
label_aqr = "AQR-HNSW" if aqr_available else "HNSW ef*2 (proxy)"
print(f"  {label_aqr:<20} {qps_aqr:>10.1f} {recall_aqr:>12.4f}")
print()
print(f"  Speedup QPS   : {qps_aqr/qps_base:.2f}x")
print(f"  Delta Recall  : {recall_aqr - recall_base:+.4f}")
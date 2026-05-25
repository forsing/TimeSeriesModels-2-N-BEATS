#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične modele time series.

2. N-BEATS: Interpretable Neural Basis Expansion


Loto 7/39 (loto7hh_4620_k41.csv):
 • Klase NBEATSBlock i NBEATS.
 • Trend (polinomna baza) i Seasonal (harmonijska baza) za interpretabilnost
   i za "backcast" residual update: residual = residual - trend.
 • Forecast glava bloka proizvodi signal dužine 39 logita (po jedan za svaki broj 1..39). 
   Sume forecast-a iz svih blokova daju ukupni multi-label izlaz.
 • Ulaz: poslednjih LOOK_BACK kola, feature-i:
     - 39 multi-hot brojeva
     - rolling frekvencije 20/50/100 za brojeve 1..39
     - gap za brojeve 1..39
     - 4 statistike kola (suma, neparni, niski, raspon)
   sve flattenovano u 1D vektor (N-BEATS očekuje [batch, input_len]).
 • Loss: BCEWithLogits + pos_weight = (39-7)/7 ≈ 4.57
 • Vremenski split train/val/back-test, bez shuffle.
 • BEST / FINAL / ENSEMBLE + back-test poslednjih 100: hits/7, AUC, LRAP.
 • SEED=39, single-thread, PyTorch deterministic.
 • Snima u 2_N-BEATS_loto_v2_predikcija.txt.

Plot plt pokazuje N-BEATS dekompoziciju iz loto signala:
Vrh: bar chart sigmoid skorova za sledeće kolo (39 brojeva), top-7 obojeni crveno.
Sredina: trend komponenta (polinomna baza) iz N-BEATS dekompozicije.
Dno: seasonal komponenta (harmonijska baza).
"""


import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


N_MIN, N_MAX = 1, 39
K = 7


class NBEATSBlock(nn.Module):
    def __init__(self, input_len, theta_dim, share_thetas=True):
        super().__init__()
        # 4-layer MLP backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Linear(input_len, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, theta_dim)
        )
        
        # Trend basis: polynomial terms
        self.trend_basis = nn.Linear(theta_dim // 2, input_len)
        
        # Seasonality basis: harmonic functions
        self.seasonality_basis = nn.Linear(theta_dim // 2, input_len)
        
        # Forecast glava: 39 logita (umesto signala dužine input_len)
        self.forecast_head = nn.Linear(theta_dim, N_MAX)
        
    def forward(self, x):
        # x: [batch, input_len]
        theta = self.backbone(x)
        theta_trend, theta_seasonal = torch.chunk(theta, 2, dim=1)
        
        # Generate interpretable components (za backcast residual)
        trend = self.trend_basis(theta_trend)
        seasonal = self.seasonality_basis(theta_seasonal)
        
        # Forecast: 39 logita po broju (multi-label izlaz za loto)
        forecast_logits = self.forecast_head(theta)
        
        return forecast_logits, trend, seasonal


class NBEATS(nn.Module):
    def __init__(self, input_len, n_stacks=8, theta_dim=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_len, theta_dim=theta_dim) for _ in range(n_stacks)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Historical values [batch, input_len]
        Returns:
            forecast_logits [batch, 39] + dekompozicija za interpretabilnost
        """
        block_forecasts = []
        block_trends = []
        block_seasonals = []
        
        # Iterative decomposition: each block refines residual
        residual = x
        for block in self.blocks:
            forecast_logits, trend, seasonal = block(residual)
            block_forecasts.append(forecast_logits)
            block_trends.append(trend)
            block_seasonals.append(seasonal)
            
            # Update residual for next block (kao u polaznom: backcast preko trend-a)
            residual = residual - trend[:, :residual.size(1)]
            
        # Sum forecast logita preko svih blokova
        total_logits = sum(block_forecasts)
        total_trend = sum(block_trends)
        total_seasonal = sum(block_seasonals)
        
        return {
            'logits': total_logits,
            'trend': total_trend,
            'seasonal': total_seasonal
        }


# =========================
# Učitavanje Loto 7/39 CSV-a
# =========================
CSV_PATH = "/loto7hh_4620_k41.csv"
OUT_TXT = Path("/2_N-BEATS_loto_v2_predikcija.txt")

LOOK_BACK = 10
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 100
BATCH = 64
LR = 1e-3
N_STACKS = 8
THETA_DIM = 64
DROPOUT = 0.10

T0 = time.time()
print()
print("START 2_N-BEATS_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N}, brojeva po kolu: {K}")
print()


# =========================
# Feature engineering
# =========================
def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, look_back):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq = make_sequences(step_features, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]

# Flatten u 1D vektor po primeru (N-BEATS očekuje [batch, input_len])
X_flat = X_seq.reshape(X_seq.shape[0], -1).astype(np.float32)

n_total = X_flat.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_train_full, Y_train_full = X_flat[:n_train], Y_seq[:n_train]
X_tr, Y_tr = X_train_full[:-VAL_N], Y_train_full[:-VAL_N]
X_val, Y_val = X_train_full[-VAL_N:], Y_train_full[-VAL_N:]
X_back, Y_back = X_flat[n_train:], Y_seq[n_train:]
X_next = step_features[-LOOK_BACK:].reshape(1, -1).astype(np.float32)

INPUT_LEN = X_flat.shape[1]
print(f"Input dim (LOOK_BACK*feat): {INPUT_LEN}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


# Train on lotto draws (Loto 7/39, 39 brojeva)
model = NBEATS(input_len=INPUT_LEN, n_stacks=N_STACKS, theta_dim=THETA_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def make_loader(X, Y, batch_size, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr, Y_tr, BATCH, shuffle=False)
best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje N-BEATS na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(xb)
        loss = criterion(output['logits'], yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        out_val = model(torch.from_numpy(X_val))
        val_loss = float(criterion(out_val['logits'], torch.from_numpy(Y_val)).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


# =========================
# Predikcija sledećeg kola + back-test
# =========================
def predict_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[start:start + BATCH])
            output = model(xb)
            out.append(torch.sigmoid(output['logits']).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, Y):
    scores = predict_scores(model, X)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, Y_back)
next_best = predict_scores(model, X_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, Y_back)
next_final = predict_scores(model, X_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("N-BEATS_best", pick_best), ("N-BEATS_final", pick_final), ("N-BEATS_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"N-BEATS_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"N-BEATS_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"N-BEATS_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<16} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'N-BEATS_best':<16} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'N-BEATS_final':<16} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'N-BEATS_ensemble':<16} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N}, epochs={EPOCHS}) ---\n")
    f.write(f"N-BEATS_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"N-BEATS_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"N-BEATS_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print()
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()


# =========================
# Plot N-BEATS dekompozicija — loto signal (polazni demo plot prilagođen)
# =========================
SHOW_PLOTS = True  # postavi False ako ne želiš da se otvori grafik
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    decomp = model(torch.from_numpy(X_next))
probs_next = torch.sigmoid(decomp['logits'])[0].cpu().numpy()
trend_next = decomp['trend'][0].cpu().numpy()
seasonal_next = decomp['seasonal'][0].cpu().numpy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
colors = ['tab:red' if (n + 1) in pick_best.tolist() else 'tab:blue' for n in range(N_MAX)]
ax1.bar(np.arange(1, N_MAX + 1), probs_next, color=colors)
ax1.set_title(f"N-BEATS — sigmoid skor za sledeće kolo (top-7 crveni): {pick_best.tolist()}")
ax1.set_xlabel("broj 1..39")
ax1.set_ylabel("verovatnoća")
ax2.plot(trend_next, label='Trend Component', color='orange')
ax2.set_title('Polynomial Trend Basis (loto input vektor)')
ax3.plot(seasonal_next, label='Seasonal Component', color='green')
ax3.set_title('Fourier Seasonal Basis (loto input vektor)')
plt.tight_layout()
plt.savefig('/Users/4c/Desktop/GHQ/TimeSeriesModels/2_nbeats_decomposition.png')
if SHOW_PLOTS:
    plt.show()
print()
print("Plot snimljen u 2_nbeats_decomposition.png")
print(f"Trend std: {trend_next.std():.3f}, Seasonal std: {seasonal_next.std():.3f}")
print()



"""
START 2_N-BEATS_loto_v2 2026-05-25 08:16:35.016406

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Input dim (LOOK_BACK*feat): 1990
Train: 4220, Val: 200, Back-test: 100 
(prvih 100 se otpisuje)

Treniranje N-BEATS na loto podacima ...
epoch    1/100  train_loss=1.14453  val_loss=1.13932  best_epoch=1
epoch   50/100  train_loss=0.10327  val_loss=2.39089  best_epoch=1
epoch  100/100  train_loss=0.01454  val_loss=3.51021  best_epoch=1

✅ Trening završen. best_epoch=1, best_val_loss=1.13932

Predikcija sledeće Loto 7/39 kombinacije:
N-BEATS_best     -> [7, x, 10, y, 26, z, 34]  (suma=140, neparnih=2/7, niskih(<=19)=3/7, raspon=27)
N-BEATS_final    -> [15, x, 21, y, 26, z, 36]  (suma=170, neparnih=4/7, niskih(<=19)=2/7, raspon=21)
N-BEATS_ensemble -> [17, x, 23, y, 26, z, 36]  (suma=178, neparnih=4/7, niskih(<=19)=1/7, raspon=19)

Back-test (poslednjih 100 izvlačenja):
model              hits/7    hit%     AUC    LRAP
N-BEATS_best        1.130   16.1%   0.492   0.234
N-BEATS_final       1.270   18.1%   0.509   0.243
N-BEATS_ensemble    1.270   18.1%   0.499   0.238
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /2_N-BEATS_loto_v2_predikcija.txt

STOP 2026-05-25 08:21:12.393104

Ukupno vreme: 0:04:37  (277.4 s)


Plot snimljen u /2_nbeats_decomposition.png
Trend std: 0.325, Seasonal std: 0.308
"""

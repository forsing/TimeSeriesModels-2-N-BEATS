#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
2. N-BEATS: Interpretable Neural Basis Expansion

Hybrid forecasting architectures that combine deep learning with classical time-series models. 
These approaches capture complex patterns while retaining interpretability and statistical rigor.
"""


"""
(NBEATSBlock/NBEATS klase, sintetička sinusoida, print o trend/seasonal std)
plt.savefig + plt.show() su pod SHOW_PLOTS = False (stavi True ako želiš grafik).
Ispod je dodata N-BEATS adaptacija za Loto 7/39 sa CSV-om loto7hh_4620_k41.csv:
LOOK_BACK = 10 kola, feature-i: multi-hot 39 + rolling 20/50/100 + gap + statistike kola
flatten u 1D vektor → N-BEATS očekuje [batch, input_len]
svaki blok daje forecast (39 logita) + backcast (residual) — stack od N_STACKS = 8
BCEWithLogits + pos_weight = (39-7)/7
vremenski split train/val/back-test, bez shuffle
BEST / FINAL / ENSEMBLE kombinacije, validacija (sortirano, jedinstveno, 1..39)
back-test poslednjih 100: hits/7, hit%, AUC, LRAP + slučajan baseline ≈ 1.256
SEED = 39, single-thread, PyTorch deterministic
upis u 2_N-BEATS_loto_v2_predikcija.txt, vreme START/STOP/elapsed
"""



import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        
    def forward(self, x):
        # x: [batch, input_len]
        theta = self.backbone(x)
        theta_trend, theta_seasonal = torch.chunk(theta, 2, dim=1)
        
        # Generate interpretable components
        trend = self.trend_basis(theta_trend)
        seasonal = self.seasonality_basis(theta_seasonal)
        
        # Forward forecast (simple linear projection here, can be extended)
        forecast = trend + seasonal
        
        return forecast, trend, seasonal

class NBEATS(nn.Module):
    def __init__(self, input_len=168, forecast_len=24, n_stacks=30):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_len, theta_dim=64) for _ in range(n_stacks)
        ])
        self.forecast_len = forecast_len
        
    def forward(self, x):
        """
        Args:
            x: Historical values [batch, history_len]
        Returns:
            Decomposed forecasts for interpretability
        """
        block_forecasts = []
        block_trends = []
        block_seasonals = []
        
        # Iterative decomposition: each block refines residual
        residual = x
        for block in self.blocks:
            forecast, trend, seasonal = block(residual)
            block_forecasts.append(forecast)
            block_trends.append(trend)
            block_seasonals.append(seasonal)
            
            # Update residual for next block (different from standard implementation)
            residual = residual - trend[:, :residual.size(1)]
            
        # Sum all block contributions
        total_forecast = sum(block_forecasts)[:, -self.forecast_len:]
        total_trend = sum(block_trends)[:, -self.forecast_len:]
        total_seasonal = sum(block_seasonals)[:, -self.forecast_len:]
        
        return {
            'forecast': total_forecast,
            'trend': total_trend,
            'seasonal': total_seasonal
        }

# Train on monthly sales data with strong trend and seasonality
model = NBEATS(input_len=36, forecast_len=12, n_stacks=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Synthetic quarterly seasonal pattern with linear trend
history = torch.sin(torch.linspace(0, 8*np.pi, 36)) + torch.linspace(0, 2, 36)
history = history.unsqueeze(0)  # Add batch dimension

# Forward pass
output = model(history)

# Plot interpretable decomposition
SHOW_PLOTS = False  # postavi True ako želiš da otvori grafik (blokira terminal)
if SHOW_PLOTS:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    ax1.plot(output['forecast'].detach()[0], label='Total Forecast')
    ax1.set_title('N-BEATS Forecast')
    ax2.plot(output['trend'].detach()[0], label='Trend Component', color='orange')
    ax2.set_title('Polynomial Trend Basis')
    ax3.plot(output['seasonal'].detach()[0], label='Seasonal Component', color='green')
    ax3.set_title('Fourier Seasonal Basis')
    plt.tight_layout()
    plt.savefig('/nbeats_decomposition.png')
    plt.show()

# Print component contributions
print()
print(f"Trend std: {output['trend'].std():.3f}, Seasonal std: {output['seasonal'].std():.3f}")
print()
"""
Trend std: 0.528, Seasonal std: 0.478
"""


# =============================================================
# N-BEATS adaptacija za Loto 7/39 (multi-label, 39 sigmoid izlaza)
#   • CSV loto7hh_4620_k41.csv
#   • ulaz: poslednjih LOOK_BACK kola, feature-i kao u TFT loto v2
#     (multi-hot 39, rolling 20/50/100, gap, statistike kola)
#   • blokovi N-BEATS-a daju forecast (39 logita) + backcast (residual)
#   • predikcija sledećeg kola → top-7 jedinstvenih, sortirano
#   • BEST / FINAL / ENSEMBLE + back-test poslednjih 100 kola
#   • snimanje u 2_N-BEATS_loto_v2_predikcija.txt
# =============================================================

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

import pandas as pd
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


CSV_PATH = "/loto7hh_4620_k41.csv"
OUT_TXT = Path("/2_N-BEATS_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 10
WINDOWS = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 462
BATCH = 64
LR = 1e-3
HIDDEN_DIM = 256
N_STACKS = 8
DROPOUT = 0.10

T0 = time.time()
print()
print("START N-BEATS loto v2", datetime.today())
print()


df_loto = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df_loto.values, axis=1)
N_total_draws = draws.shape[0]

if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV učitan: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total_draws}, brojeva po kolu: {K}")
print()


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
    scores = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -scores))
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
avg_gap_col = gap_raw.mean(axis=1, keepdims=True).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col, avg_gap_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS))
feature_scaler = StandardScaler()
step_features_scaled = step_features_raw.copy()
step_features_scaled[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features_scaled[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq = make_sequences(step_features_scaled, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]

X_flat = X_seq.reshape(X_seq.shape[0], -1).astype(np.float32)

n_total = X_flat.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_train_full, Y_train_full = X_flat[:n_train], Y_seq[:n_train]
X_tr, Y_tr = X_train_full[:-VAL_N], Y_train_full[:-VAL_N]
X_val, Y_val = X_train_full[-VAL_N:], Y_train_full[-VAL_N:]
X_back, Y_back = X_flat[n_train:], Y_seq[n_train:]
X_next = step_features_scaled[-LOOK_BACK:].reshape(1, -1).astype(np.float32)

INPUT_LEN = X_flat.shape[1]
print(f"Input dim (LOOK_BACK*feat): {INPUT_LEN}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


class NBEATSBlockLoto(nn.Module):
    """Jedan N-BEATS blok prilagođen za multi-label izlaz (39 logita)."""

    def __init__(self, input_len, hidden, dropout):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.forecast_head = nn.Linear(hidden, N_MAX)
        self.backcast_head = nn.Linear(hidden, input_len)

    def forward(self, x):
        h = self.backbone(x)
        return self.forecast_head(h), self.backcast_head(h)


class NBEATSLoto(nn.Module):
    """Stack N-BEATS blokova: forecast (39 logita) se sumira, backcast skida residual."""

    def __init__(self, input_len, n_stacks=N_STACKS, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlockLoto(input_len, hidden, dropout) for _ in range(n_stacks)
        ])

    def forward(self, x):
        residual = x
        total_forecast = torch.zeros(x.size(0), N_MAX, dtype=x.dtype, device=x.device)
        for block in self.blocks:
            forecast, backcast = block(residual)
            total_forecast = total_forecast + forecast
            residual = residual - backcast
        return total_forecast


def make_loader(X, Y, batch_size, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_scores(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[start:start + BATCH])
            logits = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, Y):
    scores = predict_scores(model, X)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))

model_loto = NBEATSLoto(input_len=INPUT_LEN)
optimizer_loto = torch.optim.AdamW(model_loto.parameters(), lr=LR, weight_decay=1e-4)
scheduler_loto = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_loto,
    mode="min",
    factor=0.5,
    patience=50,
)

train_loader = make_loader(X_tr, Y_tr, BATCH, shuffle=False)
best_state = copy.deepcopy(model_loto.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje N-BEATS_loto_v2 ...")
for epoch in range(1, EPOCHS + 1):
    model_loto.train()
    train_loss = 0.0
    seen = 0
    for xb, yb in train_loader:
        optimizer_loto.zero_grad(set_to_none=True)
        logits = model_loto(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_loto.parameters(), max_norm=1.0)
        optimizer_loto.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)

    train_loss /= max(seen, 1)
    model_loto.eval()
    with torch.no_grad():
        val_logits = model_loto(torch.from_numpy(X_val))
        val_loss = float(criterion(val_logits, torch.from_numpy(Y_val)).detach().cpu())
    scheduler_loto.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model_loto.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model_loto.state_dict())

print()
print(f"✅ N-BEATS_loto_v2 trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


model_loto.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model_loto, X_back, Y_back)
next_best = predict_scores(model_loto, X_next)[0]
pick_best = topk_from_scores(next_best)

model_loto.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model_loto, X_back, Y_back)
next_final = predict_scores(model_loto, X_next)[0]
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


elapsed_loto = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total_draws}, epochs={EPOCHS}) ---\n")
    f.write(f"N-BEATS_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"N-BEATS_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"N-BEATS_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed_loto:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed_loto)))}  ({elapsed_loto:.1f} s)")
print()




"""

Trend std: 0.528, Seasonal std: 0.478


START N-BEATS loto v2 2026-05-25 00:12:15.009812

CSV učitan: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Input dim (LOOK_BACK*feat): 2000
Train: 4220, Val: 200, Back-test: 100

Treniranje N-BEATS_loto_v2 ...
epoch    1/462  train_loss=1.14659  val_loss=1.13981  best_epoch=1
epoch   50/462  train_loss=0.14625  val_loss=1.98115  best_epoch=1
epoch  100/462  train_loss=0.03317  val_loss=2.56772  best_epoch=1
epoch  150/462  train_loss=0.01179  val_loss=2.96506  best_epoch=1
epoch  200/462  train_loss=0.00596  val_loss=3.16788  best_epoch=1
epoch  250/462  train_loss=0.00386  val_loss=3.26089  best_epoch=1
epoch  300/462  train_loss=0.00241  val_loss=3.33875  best_epoch=1
epoch  350/462  train_loss=0.00256  val_loss=3.39676  best_epoch=1
epoch  400/462  train_loss=0.00191  val_loss=3.42192  best_epoch=1
epoch  450/462  train_loss=0.00182  val_loss=3.43325  best_epoch=1
epoch  462/462  train_loss=0.00224  val_loss=3.43432  best_epoch=1

✅ N-BEATS_loto_v2 trening završen. best_epoch=1, best_val_loss=1.13981

Predikcija sledeće Loto 7/39 kombinacije:
N-BEATS_best     -> [7, x, 23, y, 29, z, 34]  (suma=164, neparnih=4/7, niskih(<=19)=2/7, raspon=27)
N-BEATS_final    -> [2, x, 7, y, 25, z, 36]  (suma=110, neparnih=4/7, niskih(<=19)=4/7, raspon=34)
N-BEATS_ensemble -> [2, x, 9, y, 25, z, 36]  (suma=120, neparnih=4/7, niskih(<=19)=4/7, raspon=34)

Back-test (poslednjih 100 izvlačenja):
model              hits/7    hit%     AUC    LRAP
N-BEATS_best        1.200   17.1%   0.503   0.247
N-BEATS_final       1.250   17.9%   0.513   0.246
N-BEATS_ensemble    1.280   18.3%   0.516   0.250
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /2_N-BEATS_loto_v2_predikcija.txt

STOP 2026-05-25 00:28:16.045557
Ukupno vreme: 0:16:01  (961.0 s)
"""




"""
CSV ima 4620 kola.
START = max(LOOK_BACK, max(WINDOWS)) = max(10, 100) = 100 — prvih 100 redova se odbacuje jer rolling 100 feature nije validan dok nemamo bar 100 prethodnih kola (puni se nulama, nije pravi signal).
Ostaje 4620 - 100 = 4520 sekvenci.
Te se dele: 100 back-test (poslednjih), 200 validacija (pre back-testa), ostatak trening = 4220.
Dakle: 4220 + 200 + 100 = 4520, plus 100 odbačeno na početku = 4620. ✅
(moze npr. da rolling=100 koristi delimičnu istoriju)

feature rolling frekvencije u zadnjih 100 kola.

Za svaki broj 1..39 računam koliko se puta pojavio u zadnjih 20, 50 i 100 kola. Prozor od 100 ne postoji za prvih 100 kola — tada iza imamo manje od 100 izvlačenja u istoriji. Kôd te redove popunjava parcijalnim/nula vrednostima, što je lažan signal i model bi učio šum.

Zato START = max(LOOK_BACK, max(WINDOWS)) = max(10, 100) = 100 — koriste se samo sekvence kod kojih su svi feature-i validni (rolling-100, lag-5, gap, statistike prošlog kola).

Dakle:
prvih 100 kola → koriste se samo kao istorija za izračunavanje feature-a kasnijih redova
od 101. kola pa nadalje → ulaze u trening/val/back-test
Ako želiš da koristimo i prvih 100, dve opcije:
(moze npr. da rolling=100 koristi delimičnu istoriju)
Smanjiti najveći prozor (npr. WINDOWS = (20, 50)) → odbacuje se samo 50.
Dozvoliti delimične prozore (rolling-100 koristi koliko ima istorije) → tada se ne odbacuje ništa, ali rani feature-i nisu istog kvaliteta.
"""

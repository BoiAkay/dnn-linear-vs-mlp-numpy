# 🍄 Mushroom Classification — Linear Model vs MLP

**BITS Pilani WILP — M.Tech (AI/ML) | Deep Neural Networks Assignment**

> Comparing a from-scratch Logistic Regression against a from-scratch Multi-Layer Perceptron on a binary classification task. No sklearn models used — everything implemented with NumPy.

---

## 📋 Assignment Overview

| Field | Details |
|---|---|
| **Course** | Deep Neural Networks — BITS Pilani WILP (M.Tech AI/ML) |
| **Student** | Ayush Kumar Gupta |
| **Student ID** | 2025AA05064 |
| **Task** | Binary Classification — Edible vs Poisonous Mushrooms |
| **Primary Metric** | Recall *(missing a poisonous mushroom is fatal)* |
| **Best Model** | **Logistic Regression — 99.74% Recall** |

---

## 🗂️ Repository Structure

```
.
├── 2025AA05064_assignment.ipynb   # Main notebook — all implementations & results
├── mushrooms.csv                  # Dataset (Kaggle / UCI ML Repository)
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | Mushroom Classification |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) / [UCI ML Repository](https://archive.ics.uci.edu/dataset/73/mushroom) |
| **Samples** | 8,124 |
| **Features** | 22 categorical features (→ 94 after one-hot encoding) |
| **Target** | `class` — `e` (edible) or `p` (poisonous) |
| **Missing Values** | `stalk-root` column — imputed with column mode |
| **Train / Test Split** | 80 / 20 (stratified) |

**Problem Statement:** Classify mushrooms as edible or poisonous from categorical physical attributes. Accurate classification is safety-critical — incorrectly labelling a poisonous mushroom as edible can be fatal.

**Why Recall?** A false negative (missing a poisonous mushroom) carries far higher risk than a false positive. Recall directly measures how many actual poisonous samples the model correctly flags.

---

## ⚙️ Preprocessing Pipeline

1. **Missing value imputation** — `stalk-root` `?` entries replaced with column mode
2. **One-hot encoding** — `pd.get_dummies(drop_first=True)` on all 22 categorical features → 94 binary features
3. **Stratified train/test split** — 80/20, `random_state=42`
4. **Feature scaling** — `StandardScaler` fit on train, applied to both splits

---

## 🤖 Models Implemented (from scratch — NumPy only)

### Baseline — Logistic Regression

| Hyperparameter | Value |
|---|---|
| Learning Rate | 0.01 |
| Iterations | 1,000 |
| Loss Function | Binary Cross-Entropy |
| Optimization | Gradient Descent |

Key implementation details:
- Sigmoid activation with `np.clip(z, -250, 250)` to prevent overflow
- Vectorized gradient computation: `dw = (1/m) * X.T @ (A - y)`
- Loss tracked at every iteration via `loss_history`

---

### MLP — Multi-Layer Perceptron

| Hyperparameter | Value |
|---|---|
| Architecture | `[94 → 16 → 8 → 1]` |
| Hidden Activation | ReLU |
| Output Activation | Sigmoid |
| Weight Init | He initialization (`√(2/n_prev)`) |
| Learning Rate | 0.01 |
| Iterations | 1,000 |
| Total Parameters | 1,665 |

Layer-wise parameter breakdown:

| Layer | Shape | Parameters |
|---|---|---|
| W1, b1 | (16, 94), (16, 1) | 1,520 |
| W2, b2 | (8, 16), (8, 1) | 136 |
| W3, b3 | (1, 8), (1, 1) | 9 |

Full backpropagation implemented via chain rule — ReLU derivatives for hidden layers, sigmoid derivative at the output layer.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **99.88%** | **100.00%** | **99.74%** | — |
| MLP `[94→16→8→1]` | 95.94% | 99.72% | 91.83% | — |

> ✅ **Baseline wins** — the dataset is linearly separable. The MLP's Recall of 91.83% means ~8% of poisonous mushrooms were missed, which is unacceptable for a safety-critical classifier.

---

## 🔍 Key Findings

**Why did Logistic Regression win?**

The Mushroom dataset is fundamentally linearly separable — specific categorical features like `odor` almost deterministically separate edible from poisonous classes. Logistic Regression, with its convex loss surface, is guaranteed to find the global optimum.

The MLP introduced unnecessary complexity. Its non-convex loss landscape led the optimizer into a local minimum, causing it to "overthink" a problem that only needed a straight-line decision boundary. A textbook demonstration of **Occam's Razor in ML**: the simplest model that fits the data is usually the best one.

**Computational cost:** The baseline trained near-instantly (simple vectorized gradient updates). The MLP required sequential forward + backward passes across all layers for 1,000 iterations — significantly more compute for worse results.

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `numpy` | All model math — matrix ops, activations, gradients |
| `pandas` | Data loading, encoding, preprocessing |
| `matplotlib` | Loss curves, performance bar charts |
| `sklearn` | **Only** `train_test_split`, `StandardScaler` — no models |

> ⚠️ No `sklearn` models, `tensorflow`, `keras`, `pytorch`, or any high-level ML library was used for model implementation.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/mushroom-classification-dnn.git
   cd mushroom-classification-dnn
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

3. **Download the dataset** and place it in the project root:
   - [Mushroom Classification — Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) → save as `mushrooms.csv`

4. **Run the notebook**
   ```bash
   jupyter notebook 2025AA05064_assignment.ipynb
   ```
   Then: `Kernel → Restart & Run All`

---

*Submitted for BITS Pilani WILP M.Tech (AI/ML) — Deep Neural Networks Programming Assignment*

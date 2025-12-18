# =========================================================
# EA vs RL-Enhanced Estimation & Allocation (RL-EA)
# Dataset: Adult Income (UCI Census - Kaggle)
# Author: (Your Name)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Load and Clean Dataset
# ---------------------------------------------------------
df = pd.read_csv("adult.csv")   # rename if needed
df = df.replace("?", np.nan).dropna()

# Convert label to 0/1
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

# ---------------------------------------------------------
# 2. Define Predicates (Actions / Data Sources)
# ---------------------------------------------------------
predicates = {
    "young": df[df["age"] < 35],
    "high_education": df[df["education"].isin(["Bachelors", "Masters", "Doctorate", "Some-college"])],
    "long_hours": df[df["hours.per.week"] > 50],
    "managerial": df[df["occupation"].str.contains("Manager", na=False)],
    "capital_gain": df[df["capital.gain"] > 0],
    "married": df[df["marital.status"].str.contains("Married", na=False)]
}

# ---------------------------------------------------------
# 3. Fixed Model Pipeline (Same for EA and RL-EA)
# ---------------------------------------------------------
X = df.drop("income", axis=1)

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

def make_model():
    return Pipeline([
        ("prep", preprocess),
        ("clf", LinearSVC(max_iter=5000))
    ])


# ---------------------------------------------------------
# 4. Budget / Acquisition Parameters
# ---------------------------------------------------------
TOTAL_BUDGET = 8000
BATCH_SIZE = 500

# RL params
EPSILON_START = 0.3
EPSILON_DECAY = 0.95

# For reproducibility (optional)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# 5. Train & Evaluate Function
# ---------------------------------------------------------
def train_and_evaluate(model, data):
    # Guard: need at least 2 classes in y to train
    if data["income"].nunique() < 2:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("income", axis=1),
        data["income"],
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=data["income"]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# ---------------------------------------------------------
# 6. Helper: Sample a batch WITHOUT re-using rows
# ---------------------------------------------------------
def acquire_batch(predicates, action, batch_size, used_indices):
    # remove previously acquired rows from this predicate pool
    pool = predicates[action].drop(index=used_indices, errors="ignore")

    if len(pool) == 0:
        return None

    actual_batch_size = min(batch_size, len(pool))
    batch = pool.sample(actual_batch_size, replace=False, random_state=None)

    # mark indices as used (so we don't "buy" same rows again)
    used_indices.update(batch.index)
    return batch

# ---------------------------------------------------------
# 7. SIMPLE EA (Baseline): random predicate each step
# ---------------------------------------------------------
def run_simple_ea(df, predicates, total_budget, batch_size):
    model = make_model()
    used_indices = set()
    train_data = pd.DataFrame()
    accuracy_history = []

    remaining = total_budget
    steps = 0

    while remaining > 0:
        action = np.random.choice(list(predicates.keys()))
        batch = acquire_batch(predicates, action, batch_size, used_indices)

        # If chosen predicate empty, try another predicate (few attempts)
        tries = 0
        while batch is None and tries < 10:
            action = np.random.choice(list(predicates.keys()))
            batch = acquire_batch(predicates, action, batch_size, used_indices)
            tries += 1

        if batch is None:
            # No more data can be acquired from any predicate
            break

        train_data = pd.concat([train_data, batch], ignore_index=False)

        acc = train_and_evaluate(model, train_data)
        accuracy_history.append(acc)

        remaining -= batch_size
        steps += 1

    return accuracy_history

# ---------------------------------------------------------
# 8. RL-EA: ε-greedy over predicates using reward = accuracy gain
# ---------------------------------------------------------
def run_rlea(df, predicates, total_budget, batch_size, eps_start, eps_decay):
    model = make_model()
    used_indices = set()
    train_data = pd.DataFrame()
    accuracy_history = []

    Q = {p: 0.0 for p in predicates}
    counts = {p: 0 for p in predicates}
    policy_usage = {p: 0 for p in predicates}

    remaining = total_budget
    epsilon = eps_start
    current_accuracy = 0.0
    SMOOTH_K = 3  # moving window

    acc_window = {p: [] for p in predicates}
    current_smoothed = {p: 0.0 for p in predicates}
    while remaining > 0:
        # choose action
        if np.random.rand() < epsilon:
            action = np.random.choice(list(predicates.keys()))
        else:
            action = max(Q, key=Q.get)

        batch = acquire_batch(predicates, action, batch_size, used_indices)

        # If chosen predicate empty, try another predicate (few attempts)
        tries = 0
        while batch is None and tries < 10:
            # explore alternatives
            action = np.random.choice(list(predicates.keys()))
            batch = acquire_batch(predicates, action, batch_size, used_indices)
            tries += 1

        if batch is None:
            break

        train_data = pd.concat([train_data, batch], ignore_index=False)

        new_accuracy = train_and_evaluate(model, train_data)
        accuracy_history.append(new_accuracy)

        # update sliding window for the selected predicate
        acc_window[action].append(new_accuracy)
        if len(acc_window[action]) > SMOOTH_K:
            acc_window[action].pop(0)

        new_smoothed = float(np.mean(acc_window[action]))
        reward = new_smoothed - current_smoothed[action]
        current_smoothed[action] = new_smoothed

        # optional: stabilize learning
        reward = np.clip(reward, -0.05, 0.05)

        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        policy_usage[action] += 1

        remaining -= batch_size
        epsilon *= eps_decay

    return accuracy_history, policy_usage

# ---------------------------------------------------------
# 9. Run both experiments
# ---------------------------------------------------------
ea_acc = run_simple_ea(df, predicates, TOTAL_BUDGET, BATCH_SIZE)
rlea_acc, rlea_usage = run_rlea(df, predicates, TOTAL_BUDGET, BATCH_SIZE, EPSILON_START, EPSILON_DECAY)

# Make x-axis comparable (steps)
ea_steps = np.arange(1, len(ea_acc) + 1)
rlea_steps = np.arange(1, len(rlea_acc) + 1)

# ---------------------------------------------------------
# 10. Plot: EA vs RL-EA Accuracy Curves (Paper-ready)
# ---------------------------------------------------------
plt.figure()
plt.plot(ea_steps, ea_acc, marker="o", label="Simple EA (Random Predicate)")
plt.plot(rlea_steps, rlea_acc, marker="o", label="RL-EA (ε-greedy)")
plt.xlabel("Acquisition Step")
plt.ylabel("Accuracy")
plt.title("EA vs RL-EA: Accuracy Improvement with Data Acquisition")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------------------------------------
# 11. Plot: RL-EA predicate usage (evidence of exploitation)
# ---------------------------------------------------------
plt.figure()
plt.bar(rlea_usage.keys(), rlea_usage.values())
plt.xticks(rotation=30)
plt.ylabel("Times Selected")
plt.title("RL-EA Predicate Selection Frequency")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# 12. Print Final Results
# ---------------------------------------------------------
print("\n=== Final Results ===")
print("EA final accuracy:", ea_acc[-1] if len(ea_acc) else None)
print("RL-EA final accuracy:", rlea_acc[-1] if len(rlea_acc) else None)

print("\nRL-EA Predicate Usage:")
for k, v in rlea_usage.items():
    print(f"{k}: {v}")

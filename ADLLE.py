"""
ADLLE: Adaptive Discriminant Locally Linear Embedding with Boundary-Sensitive
Inter-class Neighborhood Selection.

Reproduction code for the manuscript submitted to Pattern Analysis and
Applications. Run a single experiment with:

    python ADLLE.py --dataset ORL

All datasets used in the paper are included under ./data (see README.md).
"""

import argparse
import os
import random

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def getdata_orl():
    """ORL (Olivetti faces), 40 subjects x 10 images, 64x64.

    Loads the exact copy used in the paper, stored in data/ORL/olivetti.npz
    (fetched via scikit-learn with shuffle=True, random_state=42). Falls back
    to downloading it if the file is missing.
    """
    npz_path = os.path.join(DATA_ROOT, "ORL", "olivetti.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        X, Y = data["X"], data["Y"]
    else:
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        X, Y = data.data, data.target
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for subject in range(40):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 10:
            train_indices = subject_indices[:5]
            test_indices = subject_indices[5:10]
            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_ar():
    """AR face dataset, 100 subjects, 32x32, 13 train / 13 test per subject."""
    dataset_path = os.path.join(DATA_ROOT, "AR")
    images = []
    labels = []

    for label, subject_dir in enumerate(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_dir)
        for file_name in os.listdir(subject_path):
            if file_name.endswith(".jpg"):
                file_path = os.path.join(subject_path, file_name)
                image = cv2.imread(file_path, 0)
                image = cv2.resize(image, (32, 32))
                images.append(image.flatten())
                labels.append(label)

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for subject in range(100):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 10:
            train_indices = subject_indices[:13]
            test_indices = subject_indices[13:26]
            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_yale():
    """Yale face dataset, 15 subjects, 64x64, 6 train / 5 test per subject."""
    dataset_path = os.path.join(DATA_ROOT, "Yale")
    images = []
    labels = []

    for label, subject_dir in enumerate(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_dir)
        for file_name in os.listdir(subject_path):
            if file_name.endswith(".bmp"):
                file_path = os.path.join(subject_path, file_name)
                image = cv2.imread(file_path, 0)
                image = cv2.resize(image, (64, 64))
                images.append(image.flatten())
                labels.append(label)

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for subject in range(15):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 10:
            train_indices = subject_indices[:6]
            test_indices = subject_indices[6:11]
            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_yaleb():
    """Extended Yale B, up to 38 subjects, 64x64, 10 train / 10 test per subject."""
    dataset_path = os.path.join(DATA_ROOT, "Yale B")
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path does not exist: {dataset_path}")
        return None, None, None, None

    subject_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('yaleB')])

    print(f"Found {len(subject_dirs)} class folders")

    for label, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(dataset_path, subject_dir)

        pgm_files = [f for f in os.listdir(subject_path) if f.endswith('.pgm')]
        pgm_files.sort()

        if len(pgm_files) < 20:
            print(f"Warning: class {subject_dir} has only {len(pgm_files)} images, needs at least 20, skipped")
            continue

        selected_files = random.sample(pgm_files, 20)

        for file_name in selected_files:
            file_path = os.path.join(subject_path, file_name)
            image = cv2.imread(file_path, 0)

            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image.flatten())
                labels.append(label)
            else:
                print(f"Warning: cannot read image {file_path}")

    if len(images) == 0:
        print("Error: no valid images found")
        return None, None, None, None

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for subject in range(len(subject_dirs)):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 20:
            train_indices = subject_indices[:10]
            test_indices = subject_indices[10:20]

            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])
        else:
            print(f"Warning: class {subject} has only {len(subject_indices)} images, skipped")

    if len(X_train) == 0:
        print("Error: training set is empty")
        return None, None, None, None

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_umist():
    """UMIST face dataset, 20 classes, 64x64, 5 train / 5 test per class."""
    dataset_path = os.path.join(DATA_ROOT, "UMIST")
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path does not exist: {dataset_path}")
        return None, None, None, None

    subject_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('1')])

    print(f"Found {len(subject_dirs)} class folders")

    for label, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(dataset_path, subject_dir)

        pgm_files = [f for f in os.listdir(subject_path) if f.endswith('.pgm')]
        pgm_files.sort()

        if len(pgm_files) < 16:
            print(f"Warning: class {subject_dir} has only {len(pgm_files)} images, needs 16, skipped")
            continue

        selected_files = random.sample(pgm_files, 16)

        for file_name in selected_files:
            file_path = os.path.join(subject_path, file_name)
            image = cv2.imread(file_path, 0)

            if image is not None:
                image = cv2.resize(image, (64, 64))
                images.append(image.flatten())
                labels.append(label)
            else:
                print(f"Warning: cannot read image {file_path}")

    if len(images) == 0:
        print("Error: no valid images found")
        return None, None, None, None

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for subject in range(len(subject_dirs)):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 16:
            train_indices = subject_indices[:5]
            test_indices = subject_indices[5:10]

            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])
        else:
            print(f"Warning: class {subject} has only {len(subject_indices)} images, skipped")

    if len(X_train) == 0:
        print("Error: training set is empty")
        return None, None, None, None

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_pie():
    """CMU PIE (Pose 05), 68 subjects x 49 images, 64x64, 25 train / 24 test per subject."""
    dataset_path = os.path.join(DATA_ROOT, "PIE", "Pose05_64x64_files")
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path does not exist: {dataset_path}")
        return None, None, None, None

    num_subjects = 68
    images_per_subject = 49
    total_images = num_subjects * images_per_subject

    print(f"Loading PIE dataset: {num_subjects} subjects, {images_per_subject} images per subject")

    for i in range(1, total_images + 1):
        file_name = f"{i}.jpg"
        file_path = os.path.join(dataset_path, file_name)

        if os.path.exists(file_path):
            image = cv2.imread(file_path, 0)

            if image is not None:
                images.append(image.flatten())
                label = (i - 1) // images_per_subject
                labels.append(label)
            else:
                print(f"Warning: cannot read image {file_path}")
        else:
            print(f"Warning: file not found {file_path}")

    if len(images) == 0:
        print("Error: no valid images found")
        return None, None, None, None

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for subject in range(num_subjects):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= images_per_subject:
            train_indices = subject_indices[:25]
            test_indices = subject_indices[25:49]

            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])
        else:
            print(f"Warning: class {subject} has only {len(subject_indices)} images, skipped")

    if len(X_train) == 0:
        print("Error: training set is empty")
        return None, None, None, None

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_coil20():
    """COIL-20 (processed), 20 classes x 72 images, 64x64, 10 train / 10 test per class."""
    dataset_path = os.path.join(DATA_ROOT, "coil-20-proc")
    images = []
    labels = []

    for i in range(1, 21):
        for j in range(72):
            file_name = f"obj{i}__{j}.png"
            file_path = os.path.join(dataset_path, file_name)

            if os.path.exists(file_path):
                image = cv2.imread(file_path, 0)
                image = cv2.resize(image, (64, 64))
                images.append(image.flatten())
                labels.append(i - 1)

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for subject in range(20):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 20:
            random_indices = random.sample(list(subject_indices), 20)

            train_indices = random_indices[:10]
            test_indices = random_indices[10:20]

            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])
        else:
            print(f"Warning: class {subject} has only {len(subject_indices)} images, needs at least 20")

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


def getdata_coil100():
    """COIL-100, 100 classes, 20 random views per class, 64x64, 10 train / 10 test per class."""
    dataset_path = os.path.join(DATA_ROOT, "coil-100")
    images = []
    labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path does not exist: {dataset_path}")
        return None, None, None, None

    num_subjects = 100
    angles = list(range(0, 360, 5))

    print(f"Loading COIL-100 dataset: {num_subjects} classes, 20 random images per class")

    for i in range(1, num_subjects + 1):
        selected_angles = random.sample(angles, 20)

        for k in selected_angles:
            file_name = f"obj{i}__{k}.png"
            file_path = os.path.join(dataset_path, file_name)

            if os.path.exists(file_path):
                image = cv2.imread(file_path, 0)
                image = cv2.resize(image, (64, 64))
                images.append(image.flatten())
                labels.append(i - 1)
            else:
                print(f"Warning: file not found {file_path}")

    if len(images) == 0:
        print("Error: no valid images found")
        return None, None, None, None

    X = np.array(images)
    Y = np.array(labels)
    scaler = StandardScaler()
    print(f"Data loaded: {X.shape}, classes: {len(np.unique(Y))}")

    X_train, X_test, Y_train, Y_test = [], [], [], []

    for subject in range(num_subjects):
        subject_indices = np.where(Y == subject)[0]
        if len(subject_indices) >= 20:
            train_indices = subject_indices[:10]
            test_indices = subject_indices[10:20]

            X_train.extend(X[train_indices])
            X_test.extend(X[test_indices])
            Y_train.extend(Y[train_indices])
            Y_test.extend(Y[test_indices])
        else:
            print(f"Warning: class {subject} has only {len(subject_indices)} images, skipped")

    if len(X_train) == 0:
        print("Error: training set is empty")
        return None, None, None, None

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


LOADERS = {
    "ORL": getdata_orl,
    "AR": getdata_ar,
    "Yale": getdata_yale,
    "YaleB": getdata_yaleb,
    "UMIST": getdata_umist,
    "PIE": getdata_pie,
    "COIL20": getdata_coil20,
    "COIL100": getdata_coil100,
}


def compute_dlle_weights_with_alle(X, Y, k1, k0, u, kmax, kmin, eta, rr):
    n_samples = X.shape[0]
    W_intrinsic = np.zeros((n_samples, n_samples))
    W_penalty = np.zeros((n_samples, n_samples))
    sample_logs = []
    most_dis_sample = []

    # ---------------- Intrinsic graph weight computation ----------------
    for i in range(n_samples):
        current_label = Y[i]
        same_class_indices = np.where(Y == current_label)[0]
        same_class_indices = same_class_indices[same_class_indices != i]

        if len(same_class_indices) > 0:
            distances = np.linalg.norm(X[i] - X[same_class_indices], axis=1)
            k_actual = min(k1, len(same_class_indices))
            neighbors = same_class_indices[np.argsort(distances)[:k_actual]]

            if len(neighbors) > 0:
                Z = X[neighbors] - X[i]
                G = Z @ Z.T
                G_reg = G + np.eye(k_actual) * 1e-8
                try:
                    G_inv = np.linalg.inv(G_reg)
                    weights = G_inv.sum(axis=1) / G_inv.sum()
                    W_intrinsic[i, neighbors] = weights
                except np.linalg.LinAlgError:
                    W_intrinsic[i, neighbors] = 1.0 / k_actual

                most_dis_sample.append(neighbors[-1])
            else:
                most_dis_sample.append(None)
        else:
            most_dis_sample.append(None)

    # ---------------- Penalty graph weight computation ----------------
    for i in range(n_samples):
        current_label = Y[i]
        other_indices = np.where(Y != current_label)[0]

        if len(other_indices) < kmin:
            sample_logs.append({
                'sample_id': i, 'final_ki': 0, 'energy_gain': np.nan,
                'iteration_count': 0, 'label': current_label
            })
            continue

        dist_to_hetero = np.linalg.norm(X[other_indices] - X[i], axis=1)
        sorted_dist_idx = np.argsort(dist_to_hetero)
        sorted_hetero_indices = other_indices[sorted_dist_idx]
        sorted_distances = dist_to_hetero[sorted_dist_idx]

        k_candidate_max = min(kmax, len(sorted_hetero_indices))
        if k_candidate_max < kmin:
            sample_logs.append({
                'sample_id': i, 'final_ki': 0, 'energy_gain': np.nan,
                'iteration_count': 0, 'label': current_label
            })
            continue

        if i < len(most_dis_sample) and most_dis_sample[i] is not None:
            d_im = np.linalg.norm(X[i] - X[most_dis_sample[i]])
        else:
            d_im = 1.0

        candidate_indices = sorted_hetero_indices[:k_candidate_max]
        candidate_distances = sorted_distances[:k_candidate_max]

        raw_weights = np.exp(-((candidate_distances - d_im) ** 2) / (2 * rr ** 2))

        max_weight_idx = np.argmax(raw_weights)
        max_weight_pos = max_weight_idx

        start_pos = max(max_weight_pos, kmin - 1)

        best_ki = kmin
        prev_energy = 0
        iteration_count = 0

        for k in range(start_pos + 1, k_candidate_max + 1):
            iteration_count += 1

            current_indices = candidate_indices[:k]
            current_distances = candidate_distances[:k]
            current_weights = raw_weights[:k]

            current_weights_norm = current_weights / (np.sum(current_weights) + 1e-10)

            current_energy = np.sum(current_weights_norm * (current_distances ** 2))

            if k > (start_pos + 1) and prev_energy > 0:
                energy_gain = (current_energy - prev_energy) / prev_energy

                if energy_gain < eta:
                    best_ki = k - 1
                    break

            prev_energy = current_energy
            best_ki = k

        if best_ki < kmin:
            prev_energy = 0
            for k in range(kmin, k_candidate_max + 1):
                iteration_count += 1

                current_indices = candidate_indices[:k]
                current_distances = candidate_distances[:k]
                current_weights = raw_weights[:k]
                current_weights_norm = current_weights / (np.sum(current_weights) + 1e-10)

                current_energy = np.sum(current_weights_norm * (current_distances ** 2))

                if k > kmin and prev_energy > 0:
                    energy_gain = (current_energy - prev_energy) / prev_energy
                    if energy_gain < eta:
                        best_ki = k - 1
                        break

                prev_energy = current_energy
                best_ki = k

        if best_ki > 0:
            final_indices = candidate_indices[:best_ki]
            final_distances = candidate_distances[:best_ki]
            final_weights = raw_weights[:best_ki]

            if len(final_weights) > 0:
                final_weights_norm = final_weights / (np.sum(final_weights) + 1e-10)

                for idx, neighbor in enumerate(final_indices):
                    if W_penalty[i, neighbor] == 0:
                        W_penalty[i, neighbor] = 1.0 / best_ki
                        W_penalty[neighbor, i] = 1.0 / best_ki
                    else:
                        W_penalty[i, neighbor] = (1.0 / best_ki + W_penalty[i, neighbor]) / 2
                        W_penalty[neighbor, i] = (1.0 / best_ki + W_penalty[neighbor, i]) / 2

        sample_logs.append({
            'sample_id': i,
            'final_ki': best_ki,
            'max_weight_pos': max_weight_pos + 1,
            'start_pos': start_pos + 1,
            'iteration_count': iteration_count,
            'label': current_label,
            'd_im': d_im,
            'total_candidates': k_candidate_max
        })

    M = W_intrinsic
    W_symmetric = M + M.T - M.T @ M

    return W_symmetric, W_penalty, sample_logs


def calculate_top1_accuracy_with_wa(X_train, Y_train, X_test, Y_test, W_a):
    # 1. Project the train and test sets with W_a
    X_train_proj = X_train @ W_a
    X_test_proj = X_test @ W_a
    print(f"After projection with W_a - train dims: {X_train_proj.shape}, test dims: {X_test_proj.shape}")

    # 2. Nearest-neighbour classification
    n_test = X_test_proj.shape[0]
    y_pred = np.zeros(n_test, dtype=Y_train.dtype)
    for i in range(n_test):
        distances = np.sqrt(np.sum((X_train_proj - X_test_proj[i]) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        y_pred[i] = Y_train[nearest_idx]

    # 3. Top-1 accuracy
    top1_acc = accuracy_score(Y_test, y_pred) * 100
    print(f"\nTop-1 accuracy: {top1_acc:.2f}%")
    return top1_acc, y_pred


def run(dataset):
    set_seed(42)

    X_train, X_test, Y_train, Y_test = LOADERS[dataset]()

    # Parameter settings
    k1 = 3
    k0 = 15
    u = 3
    kmax = 300
    kmin = 2
    eta = 0.25
    d = 80
    rr = 1e-1
    h = 10

    W_intrinsic, W_penalty, sample_logs = compute_dlle_weights_with_alle(
        X_train, Y_train, k1, k0, u=u, kmax=kmax, kmin=kmin, eta=eta, rr=rr)
    D_intrinsic = np.diag(np.sum(W_intrinsic, axis=1))
    L = D_intrinsic - W_intrinsic
    D_penalty = np.diag(np.sum(W_penalty, axis=1))
    Lp = D_penalty - W_penalty
    S_intrinsic = X_train.T @ L @ X_train
    S_penalty = X_train.T @ Lp @ X_train
    objective_matrix = S_penalty - h * S_intrinsic
    eigenvalues, eigenvectors = np.linalg.eigh(objective_matrix)
    idx = np.argsort(eigenvalues)[::-1][:d]
    W = eigenvectors[:, idx]

    calculate_top1_accuracy_with_wa(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        W_a=W,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ADLLE experiment on one dataset.")
    parser.add_argument("--dataset", default="ORL", choices=sorted(LOADERS),
                        help="Dataset to run the experiment on (default: ORL)")
    args = parser.parse_args()
    run(args.dataset)

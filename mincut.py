import numpy as np


class IMM_Cut:
    def __init__(self, col, threshold):
        self.col = col
        self.threshold = threshold


def get_min_mistakes_cut(X, y, centers, valid_centers, valid_cols, njobs=None):
    y = np.asarray(y, dtype=int)
    active_mask = valid_centers[y] == 1
    X = X[active_mask]
    y = y[active_mask]

    n = X.shape[0]
    k = centers.shape[0]
    d = valid_cols.shape[0]

    centers_count = np.zeros(k, dtype=int)
    cols_thresholds = np.zeros(d, dtype=float)
    cols_mistakes = np.zeros(d, dtype=int)

    best_col = -1
    best_threshold = None
    min_mistakes = float('inf')

    if n == 0:
        for col in range(d):
            if valid_cols[col] != 1:
                continue
            active_center_values = centers[valid_centers == 1, col]
            if active_center_values.size >= 2 and np.min(active_center_values) < np.max(active_center_values):
                return IMM_Cut(col, np.min(active_center_values)), 0
        return None

    for i in range(k):
        centers_count[i] = 0
    for i in range(n):
        centers_count[y[i]] += 1
    if njobs is None or njobs <= 1:
        for col in range(d):
            if valid_cols[col] == 1:
                update_col_min_mistakes_cut(X, y, centers, valid_centers, centers_count, cols_thresholds, cols_mistakes,
                                            col, n, d, k)
    else:
        for col in range(d):
            if valid_cols[col] == 1:
                update_col_min_mistakes_cut(X, y, centers, valid_centers, centers_count, cols_thresholds, cols_mistakes,
                                            col, n, d, k)
    for col in range(d):
        if valid_cols[col] == 1:
            if cols_mistakes[col] != -1:
                if cols_mistakes[col] < min_mistakes:
                    best_col = col
                    min_mistakes = cols_mistakes[col]

    if best_col != -1:
        best_threshold = cols_thresholds[best_col]

    if best_col == -1:
        return None
    else:
        return IMM_Cut(best_col, best_threshold),min_mistakes


def update_col_min_mistakes_cut(X, y, centers, valid_centers, centers_count, cols_thresholds, cols_mistakes, col, n, d,
                                k):
    active_center_values = centers[valid_centers == 1, col]
    if active_center_values.size < 2 or np.min(active_center_values) == np.max(active_center_values):
        cols_thresholds[col] = -1.0
        cols_mistakes[col] = -1
        return

    ix = 0
    ic = 0
    mistakes = 0
    prev_threshold = None
    threshold = None
    max_val = -float('inf')
    data_order = np.argsort(X[:, col])
    centers_order = np.argsort(centers[:, col])

    left_centers_count = np.zeros(k, dtype=int)
    valid_found = False
    is_center_threshold = True
    min_mistakes = float('inf')

    for i in range(k):
        if valid_centers[i] == 1:
            if centers[i, col] > max_val:
                max_val = centers[i, col]
    while valid_centers[centers_order[ic]] == 0:

        ic += 1


    threshold = centers[centers_order[ic], col]

    is_center_threshold = True


    while ix < n and X[data_order[ix], col] <= threshold:
        curr_center_idx = y[data_order[ix]]
        left_centers_count[curr_center_idx] += 1
        if centers[curr_center_idx, col] >= threshold:
            mistakes += 1
        ix += 1
    if ix == n - 1:
        mistakes = 0
        ic = 0
        while ic < k:
            if valid_centers[ic] != 0:
                if centers[ic, col] > threshold:
                    mistakes += centers_count[ic]
            ic += 1
        ic = y[data_order[n - 1]]
        if centers[ic, col] > threshold:
            mistakes -= 1
        if mistakes < min_mistakes:
            valid_found = True
            best_threshold = threshold
            min_mistakes = mistakes

    while ix < n - 1 and ic < k:
        if threshold >= max_val:
            break
        if is_center_threshold == False:
            curr_center_idx = y[data_order[ix]]
            left_centers_count[curr_center_idx] += 1
            if centers[curr_center_idx, col] >= threshold:
                mistakes += 1
            elif centers[curr_center_idx, col] < threshold:
                mistakes -= 1

            ix += 1
        else:
            mistakes += (centers_count[centers_order[ic]] - 2 * left_centers_count[centers_order[ic]])
            ic += 1
            while ic < k and valid_centers[centers_order[ic]] == 0:
                ic += 1

        prev_threshold = threshold

        if X[data_order[ix], col] < centers[centers_order[ic], col]:
            threshold = X[data_order[ix], col]
            is_center_threshold = False
        else:
            threshold = centers[centers_order[ic], col]
            is_center_threshold = True

        if prev_threshold != threshold and mistakes < min_mistakes:
            valid_found = True
            best_threshold = prev_threshold
            min_mistakes = mistakes
    if valid_found:
        cols_thresholds[col] = best_threshold
        cols_mistakes[col] = min_mistakes
    else:
        cols_thresholds[col] = -1.0
        cols_mistakes[col] = -1

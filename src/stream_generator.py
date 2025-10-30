# src/stream_generator.py
def stream_windows(X, y, init_size=5000, window_size=500):
    """
    Split dataset into an initial training chunk + stream windows.
    """
    n = len(X)
    init_end = init_size
    X_init, y_init = X.iloc[:init_end].copy(), y.iloc[:init_end].copy()
    for start in range(init_end, n, window_size):
        end = min(start + window_size, n)
        yield X.iloc[start:end].copy(), y.iloc[start:end].copy()
    return X_init, y_init

def stratified_stream_windows(X, y, window_size=1000):
    """
    Generator yielding class-balanced streaming windows using stratified splits.
    Each window contains a representative sample of each class.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    num_windows = int(len(X) // window_size)
    sss = StratifiedShuffleSplit(n_splits=num_windows, test_size=window_size, random_state=42)
    for _, win_idx in sss.split(X, y):
        yield X.iloc[win_idx].copy(), y.iloc[win_idx].copy()

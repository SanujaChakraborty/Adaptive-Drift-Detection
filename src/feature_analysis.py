import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(df):
    # 1️⃣ Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # 2️⃣ Replace inf / -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3️⃣ Drop columns that are entirely NaN
    df = df.dropna(axis=1, how='all')

    # 4️⃣ Fill remaining NaNs with column mean (numerical only)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # 5️⃣ Encode all categorical columns automatically
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 6️⃣ Check for label column
    if 'label' not in df.columns:
        raise ValueError(f"❌ 'label' column not found! Columns available: {df.columns.tolist()[:10]}...")

    # 7️⃣ Encode label if needed
    if df['label'].dtype == object:
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'].astype(str))
        label_encoders['label'] = le

    # 8️⃣ Final cleanup – ensure no infs/NaNs remain
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # 9️⃣ Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']

    print(f"✅ Preprocessing done! {len(label_encoders)} categorical columns encoded. Shape: {X.shape}")
    return X, y

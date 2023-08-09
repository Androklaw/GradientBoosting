import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def get_highly_correlated_features(df, num_features, corr_threshold=0.90):
    corr_df = df[num_features].corr(method='pearson')
    mask = np.tri(corr_df.shape[0], k=-1, dtype=bool)
    corr_df = corr_df.where(mask)
    highcorr_df = corr_df[corr_df.abs() > corr_threshold].stack().dropna().reset_index()
    highcorr_df = highcorr_df[highcorr_df['level_0'] != highcorr_df['level_1']]
    highcorr_features = list(set(highcorr_df['level_0'].to_numpy().flatten()))
    return highcorr_features



# df = pd.read_csv("./data/Breast Cancer/wdbc.data")
df = pd.read_csv("./data/diabetes/data.csv").reset_index(drop=True)


# target_name = 'diagnosis'
# cat_features = []
# num_features = [c for c in df.columns if c != target_name and c != 'id']
# pca_outlier_remove = True


# target_mapping = {'M': 'malignant', 'B': 'benign'}
# df[target_name] = df[target_name].replace(target_mapping)


target_name = 'diabetes'
num_features = ['HbA1c_level','bmi','age','blood_glucose_level']
cat_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
pca_outlier_remove = False


input_features = num_features + cat_features
if len(cat_features) != 0:
    df[cat_features] = df[cat_features].astype('category')


test_size = 0.20
df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_name])


highcorr_features = get_highly_correlated_features(df, num_features, corr_threshold=0.90)
num_features = list(set(num_features) - set(highcorr_features))
df = df.drop(highcorr_features, axis=1, errors='ignore')


t_df = df.copy()
if len(cat_features) > 0:
    t_df = pd.concat([t_df, pd.get_dummies(t_df[cat_features])], axis=1)
    t_df = t_df.drop(cat_features, axis=1)

if pca_outlier_remove:
    pca = PCA(n_components=None)
    pca_df = pd.DataFrame(pca.fit_transform(t_df.drop(['id', target_name], axis=1, errors='ignore')), columns=[f'V{i}'for i in range(t_df.shape[1]-2)])
    pca_df[target_name] = df[target_name]

    q1, q3 = pca_df['V0'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr 
    uw = q3 + 1.5 * iqr
    outliers_mask = (pca_df['V0'] > uw) | (pca_df['V0'] < lw)
    pca_df['is_outlier'] = False
    pca_df.loc[outliers_mask, 'is_outlier'] = True
    df = df[~pca_df['is_outlier'].values].reset_index(drop=True)


# df.to_csv('data/preprocessed/train_breast_cancer.csv', index=False)
# test_df.to_csv('data/preprocessed/test_breast_cancer.csv', index=False)

df.to_csv('data/preprocessed/train_diabetes.csv', index=False)
test_df.to_csv('data/preprocessed/test_diabetes.csv', index=False)




fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y='V1', hue='is_outlier', ax=ax)
plt.tight_layout()


ev_df = pd.DataFrame(zip(range(len(input_features)), pca.explained_variance_ratio_), columns=['component', 'value'])
fig, ax = plt.subplots(1, figsize=(10, 10))
sb.lineplot(data=ev_df, x='component', y='value',ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y=target_name, ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(10, 10))
sb.scatterplot(data=pca_df, x='V0', y='V1', hue=target_name, ax=ax)
plt.tight_layout()


m_df = df.melt(
    id_vars=[target_name,'id'], 
    value_vars=input_features,
    var_name='input_feature', 
    value_name='value'
)


features = [c for c in input_features if 'radius' in c]
fig, ax = plt.subplots(1, figsize=(10, 6))
sb.boxplot(data=m_df[m_df['input_feature'].isin(features)], x='value', y='input_feature', hue=target_name, showfliers=True, ax=ax)
plt.tight_layout()


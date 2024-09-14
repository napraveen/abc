Data Preprocessing
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path = os.path.join(dirname, filename) 
        print(path.ljust(100), os.path.getsize(path) // int(1e6), 'MB')
/kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z                                        77 MB
/kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z                                         35 MB
/kaggle/input/mercari-price-suggestion-challenge/sample_submission_stg2.csv.zip                      8 MB
/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip                                   308 MB
/kaggle/input/mercari-price-suggestion-challenge/sample_submission.csv.7z                            0 MB
!p7zip -d -f -k ../input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k ../input/mercari-price-suggestion-challenge/test.tsv.7z
!unzip -o -d /kaggle/working/ /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip
!p7zip -d -f -k ../input/mercari-price-suggestion-challenge/sample_submission.csv.7z
!unzip -o -d /kaggle/working/ /kaggle/input/mercari-price-suggestion-challenge/sample_submission_stg2.csv.zip
dir_path = '/kaggle/working'
train = pd.read_csv(f'{dir_path}/train.tsv', delimiter='\t')
test  = pd.read_csv(f'{dir_path}/test.tsv', delimiter='\t')
train.head()

test.head()

print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')
train.shape: (1482535, 8)
test.shape : (693359, 7)
target_feature = sorted(list(set(train.columns)-set(test.columns)))[0]
target_feature
'price'
def df_dtypes(df):
    pd.set_option('display.max_colwidth', None)

    df_dtypes = df.columns.groupby(df.dtypes)
    df_dtypes = pd.DataFrame({
        'dtype':     list(df_dtypes.keys()),
        '# columns': [len(df_dtypes[key])  for key in df_dtypes.keys()],
        'columns':   [list(df_dtypes[key]) for key in df_dtypes.keys()],
    })
    df_dtypes = df_dtypes.style.applymap(lambda x:'text-align: left', subset=['columns'])
    return df_dtypes
df_dtypes(train)
def compute_dtype(df):
    return [df[col].dtype                                              for col in df.columns]
def compute_count(df):
    return [df[col].count()                                            for col in df.columns]
def compute_mean(df, features):
    return [round(df[col].mean(), 1)        if col in features else '' for col in df.columns]
def compute_std(df, features):
    return [round(df[col].std(), 1)         if col in features else '' for col in df.columns]
def compute_min(df, features):
    return [round(df[col].min(), 1)         if col in features else '' for col in df.columns]
def compute_max(df, features):
    return [round(df[col].max(), 1)         if col in features else '' for col in df.columns]
def compute_quantile(df, features, percentage):
    return [round(df[col].quantile(percentage), 1) \
                                            if col in features else '' for col in df.columns]
def compute_IQR(df, features):
    return [round(df[col].max()-df[col].min(), 1) \
                                            if col in features else '' for col in df.columns]
def compute_nunique(df):
    return [df[col].nunique()                                          for col in df.columns]
def compute_unique(df, threshold):
    return [df[col].unique()  if df[col].nunique() < threshold else '' for col in df.columns]
def compute_mode(df):
    return [df[col].mode()[0]           if len(df[col].mode()) else '' for col in df.columns]
def compute_mode_count(df):
    return [df[col].value_counts().max()                               for col in df.columns]
def compute_mode_percentage(df):
    return [round(df[col].value_counts().max() * 100 / df.shape[0], 1) for col in df.columns]
def compute_null_count(df):
    return [df[col].isnull().sum()                                     for col in df.columns]
def compute_null_percentage(df):
    return [round(df[col].isnull().mean() * 100, 1)                    for col in df.columns]
def build_my_info_table(df, nunique_threshold=30):
    numerical_features = df.select_dtypes(include=[np.number])
    df_info = pd.DataFrame({
        '#':        np.arange(len(df.columns)),
        'column':   df.columns,
        'dtype':    compute_dtype(df),
        'count':    compute_count(df),
        'mean':     compute_mean(df, numerical_features),
        'std':      compute_std(df, numerical_features),
        'min':      compute_min(df, numerical_features),
        '25%':      compute_quantile(df, numerical_features, .25),
        '50%':      compute_quantile(df, numerical_features, .5),
        '75%':      compute_quantile(df, numerical_features, .75),
        'max':      compute_max(df, numerical_features),
        'IQR':      compute_IQR(df, numerical_features),
        'nunique':  compute_nunique(df),
        'unique':   compute_unique(df, nunique_threshold),
        'mode':     compute_mode(df),
        'mode #':   compute_mode_count(df),
        'mode %':   compute_mode_percentage(df),
        'null #':   compute_null_count(df),
        'null %':   compute_null_percentage(df),
    })
    df_info = df_info.sort_values(by='dtype')
    return df_info
build_my_info_table(train)
def plot_bar_chart(df, x, y, xlabel, ylabel, title, xmin=None, xmax=None, palette='deep'):
    if df.shape[0] == 0:
        return
    size = (12, df.shape[0] / 4 + 1)
    plt.figure(figsize=size)
    sns.barplot(y=df[y], x=df[x], palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not xmin and not xmax:
        xmin, xmax = df[x].min(), df[x].max()
        xrange = (xmax - xmin) * .1
        xmin, xmax = xmin-xrange, xmax+xrange
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()
train_info = build_my_info_table(train)
nan_df = train_info[train_info['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', xlabel='Null Percentage %', ylabel='Feature', title='Null Percentage in each Feature', xmin=0, xmax=100, palette='coolwarm')

train = train.drop(columns=list(nan_df[nan_df['null %'] > 50]['column']))
test  = test.drop(columns=list(nan_df[nan_df['null %'] > 50]['column']))
test_info = build_my_info_table(test)
nan_df = test_info[test_info['null %'] >= 10][['column', 'null %']].sort_values(by='null %')
plot_bar_chart(nan_df, x='null %', y='column', xlabel='Null Percentage %', ylabel='Feature', title='Null Percentage in each Feature', xmin=0, xmax=100, palette='coolwarm')

train = train.drop(columns=list(nan_df[nan_df['null %'] > 50]['column']))
test  = test.drop(columns=list(nan_df[nan_df['null %'] > 50]['column']))
def fillna(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_features:
        df[col] = df[col].fillna(df[col].mode()[0])

def replace_inf(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = df[numerical_features].replace([np.inf, -np.inf], np.nan)
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
replace_inf(train)
replace_inf(test)
fillna(train)
fillna(test)
def get_categorical_features(df, nunique_threshold=30):
    all_df_columns = df.columns
    categorical_features = [col for col in all_df_columns if df[col].nunique() < nunique_threshold]
    return categorical_features

def get_numerical_features(df, nunique_threshold=30):
    numerical_df_columns = df.select_dtypes(include=[np.number]).columns
    numerical_features   = [col for col in numerical_df_columns if df[col].nunique() >= nunique_threshold]
    return numerical_features
categorical_features = get_categorical_features(train)
numerical_features   = get_numerical_features(train)
print(f'categorical_features: {len(categorical_features)}\n{categorical_features}\n')
print(f'numerical_features:   {len(numerical_features)}\n{numerical_features}\n')
categorical_features: 2
['item_condition_id', 'shipping']

numerical_features:   2
['train_id', 'price']

print(f'train.shape: {train.shape}')
print(f'test.shape : {test.shape}')
train.shape: (1482535, 8)
test.shape : (693359, 7)
Exploratory Data Analysis
# Distribution Plots:
def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
def my_distplot(df, col, ax):
    sns.distplot(df[col], ax=ax)
    ax.set_title(f'Distribution Plot of {col}')
def my_kdeplot(df, col, ax):
    sns.kdeplot(df[col], ax=ax, fill=True)
    ax.set_title(f'KDE Plot of {col}')
    
# Categorical Plots:
def my_pie_chart(df, col, ax):
    labels = df[col].value_counts()
    ax.pie(labels, labels=labels.index, autopct='%1.1f%%')
    ax.set_title(f'Pie Chart of {col}')
def my_countplot(df, col, ax):
    sns.countplot(x=df[col], ax=ax)
    ax.set_title(f'Count Plot of {col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
def my_boxplot(df, col, ax):
    sns.boxplot(y=df[col], ax=ax)
def my_violinplot(df, col, ax):
    sns.violinplot(y=df[col], ax=ax)
    
# Matrix Plots:
def my_heatmap(df, size, cmap, cbar_kws, font_size):
    plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap=cmap, center=0, cbar_kws=cbar_kws, annot_kws={"size": font_size})
    plt.title('Correlation Heatmap')
    plt.show()
def plot_features(df, plot_funcs, width_ratios, height_ratios, n_col=1):
    def plot_feature(cols):
        n_plot_funcs = len(plot_funcs)
        fig = plt.figure(figsize=(sum(width_ratios), max(height_ratios)))
        gs = fig.add_gridspec(1, n_plot_funcs*n_col, width_ratios=width_ratios, height_ratios=height_ratios)
        axes = [0] * (n_plot_funcs * n_col)

        for i in range(n_col):
            for j in range(n_plot_funcs):
                k = i*n_plot_funcs+j
                axes[k] = fig.add_subplot(gs[0, k])
                plot_funcs[j](df, cols[i], axes[k])
                axes[k].set_xlabel(cols[i])

        plt.tight_layout()
        plt.show()
    
    for i in range(0, len(df.columns), n_col):
        plot_feature(df.columns[i:i+n_col])
def plot_numerical_features(df, plot_funcs=[my_boxplot, my_violinplot, my_distplot], width_ratios=[2, 2, 12], height_ratios=[4], n_col=1):
    plot_features(df, plot_funcs, width_ratios, height_ratios)

def plot_categorical_features(df, plot_funcs=[my_pie_chart, my_countplot], width_ratios=[5, 11], height_ratios=[5], n_col=1):
    plot_features(df, plot_funcs, width_ratios, height_ratios)
plot_numerical_features(train[numerical_features])


plot_categorical_features(train[categorical_features])


plot_numerical_features(train[[target_feature]])

for cat_feature in categorical_features:
    sns.catplot(data=train, x=cat_feature, y=target_feature, kind='box', height=4, aspect=3)


def plot_features_correlation(df, features, target_feature, plot_kinds, step=6, height=3, aspect=1):
    df = df.sample(min(1000, df.shape[0]))
    for i in range(0, len(features), step):
        for plot_kind in plot_kinds:
            sns.pairplot(df, x_vars=features[i:i+step], y_vars=[target_feature], kind=plot_kind, height=height, aspect=aspect)
plot_features_correlation(train, numerical_features, target_feature, plot_kinds=['reg', 'kde'])


Baseline Model
df_dtypes(train)
train_model1 = train.copy()
test_model1  = test.copy()
def drop_id_feature(df, id_col='id'):
    if id_col not in df.columns:
        return df, None
    df_id = df[id_col]
    df = df.drop(columns=[id_col])
    return df, df_id
train_model1, _ = drop_id_feature(train_model1, 'train_id')
test_model1, _  = drop_id_feature(test_model1, 'test_id')
from sklearn.preprocessing import LabelEncoder

def encode_str_features(df):
    categorical_features = get_categorical_features(df.select_dtypes(exclude=[np.number]))
    label_encoder = LabelEncoder()
    for col in categorical_features:
        df[col] = label_encoder.fit_transform(df[col])
encode_str_features(train_model1)
encode_str_features(test_model1)
df_dtypes(train_model1)
train_model1 = train_model1.drop(columns=train_model1.select_dtypes(exclude=[np.number]).columns)
test_model1  = test_model1.drop(columns=test_model1.select_dtypes(exclude=[np.number]).columns)
df_dtypes(train_model1)
from sklearn.metrics         import mean_absolute_error, \
                                    mean_squared_error, \
                                    mean_absolute_percentage_error, \
                                    median_absolute_error, \
                                    max_error, \
                                    r2_score, \
                                    explained_variance_score

def evaluate_model(y_test, y_pred):
    mae   = mean_absolute_error(y_test, y_pred)
    mse   = mean_squared_error(y_test, y_pred)
    mape  = mean_absolute_percentage_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    me    = max_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)
    ev    = explained_variance_score(y_test, y_pred)

    result = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Mean Absolute Percentage Error': mape,
        'Median Absolute Error': medae,
        'Max Error': me,
        'R^2': r2,
        'Explained Variance': ev,
    }
    return result
from sklearn.linear_model    import LinearRegression, \
                                    Ridge

from sklearn.tree            import DecisionTreeRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.ensemble        import RandomForestRegressor

baseline_models = {
    'LinearRegression':          LinearRegression(),
    'Ridge':                     Ridge(),
    'DecisionTreeRegressor':     DecisionTreeRegressor(),
}
#    'RandomForestRegressor':     RandomForestRegressor(),
#    'KNeighborsRegressor':       KNeighborsRegressor(),
import time

def run_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        print(f'Model {name} begining now ...')
        begin_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred)
        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        print(f'Model {name} run in'.ljust(50), f'{duration} minutes')
    results = pd.DataFrame(results).T
    results = results.reset_index()
    results = results.rename(columns={'index': 'Model'})
    return results
from sklearn.model_selection import train_test_split

def split_data_train_test(df, target_feature, test_size=0.2):
    y = df[target_feature]
    X = df.drop(columns=[target_feature])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_data_train_test(train_model1, target_feature)
models_result1 = run_models(baseline_models, X_train, X_test, y_train, y_test)
models_result1
metrics = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Absolute Percentage Error', 'Median Absolute Error', 'Max Error', 'R^2', 'Explained Variance']
for metric in metrics:
    plot_bar_chart(models_result1, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")

def get_best_model(results_df, models, metric):
    best_model_name = results_df.sort_values(by=[metric]).head(1)['Model'].iloc[0]
    best_model = models[best_model_name]
    return best_model
best_models1 = get_best_model(models_result1, baseline_models, 'Mean Squared Error')
print('Best Model of Baseline Models is:', best_models1.__class__.__name__)
Best Model of Baseline Models is: DecisionTreeRegressor
del train_model1, test_model1, X_train, X_test, y_train, y_test
Feature Engineering and Selection
df_dtypes(train)
train.select_dtypes(include=[np.number]).head(10)
train.select_dtypes(exclude=[np.number]).head(10)
build_my_info_table(train)
train_model2 = train.copy()
test_model2  = test.copy()
train_model2, _ = drop_id_feature(train_model2, 'train_id')
test_model2, _  = drop_id_feature(test_model2, 'test_id')
def add_columns(df):
    df[["first_category", "second_category", "third_category"]] = df["category_name"].str.split('/', n=2, expand=True)
    df["item_description"]     = df["item_description"].replace({"no description": np.nan})
    df["missing_category"]     = df["first_category"].isna()
    df["missing_brand"]        = df["brand_name"].isna()
    df["missing_description"]  = df["item_description"].isna()
    df["price_in_description"] = df["item_description"].str.contains("\[rm\]")
    df["price_in_name"]        = df["name"].str.contains("\[rm\]")
    df["descr_len"]            = df["item_description"].str.len()
    df["descr_len"]            = pd.cut(df["descr_len"], np.arange(0, 1055, 5), right=False)
    df["descr_len"]            = df["descr_len"].astype("object")
add_columns(train_model2)
add_columns(test_model2)
df_dtypes(train_model2)
encode_str_features(train_model2)
encode_str_features(test_model2)
df_dtypes(train_model2)
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df):
    categorical_features = get_categorical_features(df.select_dtypes(include=[np.number]))
    if target_feature in categorical_features:
        categorical_features.remove(target_feature)
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features), index=df.index)
    df = df.drop(columns=categorical_features)
    df = pd.concat([df, encoded_df], axis=1)
    return df
train_model2 = train_model2.drop(columns=train_model2.select_dtypes(exclude=[np.number]).columns)
test_model2  = test_model2.drop(columns=test_model2.select_dtypes(exclude=[np.number]).columns)
df_dtypes(train_model2)
def get_skewed_features(df, threshold=0.25):
    numerical_features = get_numerical_features(df)
    skew_df = df[numerical_features].apply(lambda x: x.skew())
    skew_df = skew_df.sort_values(ascending=False)
    skew_df = skew_df.reset_index()
    skew_df.columns = ['Feature', 'SkewFactor']
    skewed_features = list(skew_df[abs(skew_df['SkewFactor']) > threshold]['Feature'])
    non_skewed_features = list(set(numerical_features)-set(skewed_features))
    return skewed_features, non_skewed_features, skew_df
from sklearn.preprocessing import StandardScaler

def transform_skewed_features(df, skewed_features):
    for col in skewed_features:
        df[col] = np.log1p(df[col])

standardScaler = StandardScaler()

def transform_non_skewed_features(df, non_skewed_features):
    for col in non_skewed_features:
        df[col] = standardScaler.fit_transform(df[[col]])
skewed_features, non_skewed_features, skew_df = get_skewed_features(train_model2)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}\n')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}\n')
skewed_features:     1
['price']

non_skewed_features: 0
[]

plot_bar_chart(skew_df, x='SkewFactor', y='Feature', xlabel='Skew Factor', ylabel='Feature', title='Skew Factor in each Feature', palette='coolwarm')

plot_features_correlation(train_model2, skewed_features[:-2], target_feature, plot_kinds=['kde', 'reg'])
transform_skewed_features(train_model2, skewed_features)
plot_features_correlation(train_model2, skewed_features[:-2], target_feature, plot_kinds=['kde', 'reg'])
plot_features_correlation(train_model2, non_skewed_features[:], target_feature, plot_kinds=['kde', 'reg'])
transform_non_skewed_features(train_model2, non_skewed_features)
plot_features_correlation(train_model2, non_skewed_features[:], target_feature, plot_kinds=['kde', 'reg'])
skewed_features, non_skewed_features, skew_df = get_skewed_features(test_model2)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}\n')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}\n')
skewed_features:     0
[]

non_skewed_features: 0
[]

transform_skewed_features(test_model2, skewed_features)
transform_non_skewed_features(test_model2, non_skewed_features)
Enhanced Features Model
df_dtypes(train_model2)
replace_inf(train_model2)
replace_inf(test_model2)
fillna(train_model2)
fillna(test_model2)
def split_data_X_y(df, target_feature):
    y = df[target_feature]
    X = df.drop(columns=[target_feature])
    return X, y
X, y = split_data_X_y(train_model2, target_feature)
from sklearn.feature_selection import mutual_info_regression

def feature_selection(X, y, mutual_info_method, top_n=10):
    mutual_info = mutual_info_method(X, y)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'ImportanceFactor': mutual_info
    })
    importance_df = pd.DataFrame(importance_df, columns=['Feature', 'ImportanceFactor'])
    importance_df = importance_df.sort_values(by='ImportanceFactor', ascending=False)
    importance_df = importance_df.head(top_n)
    return importance_df
importance_df = feature_selection(X, y, mutual_info_regression)
plot_bar_chart(importance_df, x='ImportanceFactor', y='Feature', xlabel='Importance Factor', ylabel='Feature', title='Importance Factor in each Feature', palette='coolwarm')

selected_features = importance_df['Feature']
X_selected  = X[selected_features]
test_model2 = test_model2[selected_features]
df_dtypes(X_selected)
 	dtype	# columns	columns
0	int64	8	['shipping', 'first_category', 'item_condition_id', 'price_in_description', 'missing_category', 'missing_brand', 'missing_description', 'price_in_name']
X_train_test_model2 = pd.concat([X_selected, test_model2])
X_train_test_model2 = one_hot_encoding(X_train_test_model2)
X_selected  = X_train_test_model2[:len(X_selected)]
test_model2 = X_train_test_model2[-len(test_model2):]
df_dtypes(X_selected)
 	dtype	# columns	columns
0	float64	18	['shipping_1', 'first_category_1', 'first_category_2', 'first_category_3', 'first_category_4', 'first_category_5', 'first_category_6', 'first_category_7', 'first_category_8', 'first_category_9', 'item_condition_id_2', 'item_condition_id_3', 'item_condition_id_4', 'item_condition_id_5', 'price_in_description_1', 'price_in_description_2', 'missing_description_1', 'price_in_name_1']
def get_target_correlations(df, target_feature):
    numerical_features  = df.select_dtypes(include=[np.number]).columns
    correlation_matrix  = df[numerical_features].corr()
    target_correlations = correlation_matrix[target_feature].sort_values(ascending=False)
    target_correlations = target_correlations.reset_index()
    target_correlations.columns = ['Feature', 'TargetCorrelation']
    return target_correlations
correlations_df = get_target_correlations(train_model2[list(selected_features)+[target_feature]], target_feature)
plot_bar_chart(correlations_df, x='TargetCorrelation', y='Feature', xlabel='Target Correlation', ylabel='Feature', title='Target Correlation in each Feature', palette='coolwarm')

from matplotlib.colors import LinearSegmentedColormap

def plot_heatmap(df, size_factor=1/2):
    df = df.select_dtypes(include=[np.number])
    height = int(len(df.columns) * size_factor)
    font_size = max(min(12, 119 // height), 8)
    cmap = LinearSegmentedColormap.from_list(
        'custom_diverging',
        ['blue', 'lightblue', 'white', 'lightcoral', 'red'],
        N=5
    )
    cbar_kws = {'ticks': [-1, -.5, 0, .5, 1]}
    my_heatmap(df, size=(height+1, height+1), cmap=cmap, cbar_kws=cbar_kws, font_size=font_size)
plot_heatmap(pd.concat([X_selected, y], axis=1))

X_train, X_test, y_train, y_test = split_data_train_test(pd.concat([X_selected, y], axis=1), target_feature)
models_result2 = run_models(baseline_models, X_train, X_test, y_train, y_test)
models_result2
for metric in metrics:
    plot_bar_chart(models_result2, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")







best_models2 = get_best_model(models_result2, baseline_models, 'Mean Squared Error')
print('Best Model of Enhanced Features Models is:', best_models2.__class__.__name__)
Best Model of Enhanced Features Models is: DecisionTreeRegressor
del train_model2, test_model2, X_selected, X, y, X_train, X_test, y_train, y_test, X_train_test_model2
Outlier Detection and Treatment
df_dtypes(train)
 	dtype	# columns	columns
0	int64	3	['train_id', 'item_condition_id', 'shipping']
1	float64	1	['price']
2	object	4	['name', 'category_name', 'brand_name', 'item_description']
train_model3 = train.copy()
test_model3  = test.copy()
train_model3, _ = drop_id_feature(train_model3, 'train_id')
test_model3, _  = drop_id_feature(test_model3, 'test_id')
encode_str_features(train_model3)
encode_str_features(test_model3)
df_dtypes(train)
train_model3 = train_model3.drop(columns=train_model3.select_dtypes(exclude=[np.number]).columns)
test_model3  = test_model3.drop(columns=test_model3.select_dtypes(exclude=[np.number]).columns)
df_dtypes(train_model3)
def plot_feature_with_outlier(df, plot_func, outliers_indices, outliers_methods, x_col, y_col):
    for i in range(len(outliers_methods)):
        fig, ax = plt.subplots(1, 2, figsize=(12*2, 4))
        ax[0].set_title('Original Data')
        ax[1].set_title(f'Data without Outliers of {y_col} using {outliers_methods[i].__name__}')
        plot_func(x=x_col, y=y_col, ax=ax[0], data=df)
        plot_func(x=x_col, y=y_col, ax=ax[1], data=df.drop(outliers_indices[i]))
        plt.tight_layout()
        plt.show()
def outliers_iqr(df, col):
    outliers = pd.DataFrame()
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    outliers = df.index[outliers]
    return outliers

def outliers_zscore(df, col, threshold=3):
    outliers = pd.DataFrame()
    z_scores = np.abs((df[col] - np.mean(df[col])) / np.std(df[col]))
    outliers = z_scores > threshold
    outliers = df.index[outliers]
    return outliers
def detect_outliers(df, target_feature, outliers_method):
    outliers_indices = [item for item in outliers_method(df, target_feature)]
    return outliers_indices
def plot_feature_with_outlier_methods(df, target_feature, outliers_methods, plot_func):
    outliers = [0] * len(outliers_methods)
    for i in range(len(outliers_methods)):
        outliers[i] = detect_outliers(df, target_feature, outliers_methods[i])
        print(f'length of {target_feature} outliers_indices: {len(outliers[i])} using {outliers_methods[i].__name__}')
    if sum(len(curr_outliers) for curr_outliers in outliers) == 0:
        return
    for curr_col in df.columns:
        if curr_col == target_feature:
            continue
        plot_feature_with_outlier(df, plot_func, outliers, outliers_methods, x_col=curr_col, y_col=target_feature)
def outlier_summary(df, method1, method2):
    numerical_features = get_numerical_features(df)
    df = df[numerical_features]
    outlier_info = pd.DataFrame({
        '#':        np.arange(len(df.columns)),
        'column':   df.columns,
        'dtype':    compute_dtype(df),
        'count':    compute_count(df),
        'mean':     compute_mean(df, numerical_features),
        'std':      compute_std(df, numerical_features),
        'min':      compute_min(df, numerical_features),
        '25%':      compute_quantile(df, numerical_features, .25),
        '50%':      compute_quantile(df, numerical_features, .5),
        '75%':      compute_quantile(df, numerical_features, .75),
        'max':      compute_max(df, numerical_features),
        'IQR':      compute_IQR(df, numerical_features),

        f'{method1.__name__}':  [list(method1(df, col)) for col in numerical_features],
        f'{method2.__name__}':  [list(method2(df, col)) for col in numerical_features],
    })

    outlier_info[f'{method1.__name__} length']  = outlier_info[f'{method1.__name__}'].apply(len)
    outlier_info[f'{method2.__name__} length']  = outlier_info[f'{method2.__name__}'].apply(len)

    if outlier_info[f'{method1.__name__} length'].shape[0] == 0 or \
       outlier_info[f'{method2.__name__} length'].shape[0] == 0:
        outlier_info['all_indices'] = pd.Series([set() for _ in range(len(outlier_info))], 
                                                index=outlier_info.index, dtype='object')
    else:
        outlier_info['all_indices'] = outlier_info.apply(lambda x:set(x[f'{method1.__name__}']) & \
                                                                  set(x[f'{method2.__name__}']), axis=1)
    outlier_info['all_indices length'] = outlier_info['all_indices'].apply(len)
    
    outlier_info = outlier_info.sort_values(by='all_indices length', ascending=False)
    outlier_info = outlier_info[outlier_info['all_indices length'] > 0]
    return outlier_info
outlier_info = outlier_summary(train_model3, outliers_iqr, outliers_zscore)
outlier_info.drop(columns=['outliers_iqr', 'outliers_zscore', 'all_indices'])
def get_outlier_indices(outlier_info, method1, method2):
    col = 'all_indices'
    all_indices = set()
    for curr_indices in outlier_info[col]:
        all_indices |= curr_indices
    outlier_percentage = round(len(all_indices) / outlier_info['count'].max() * 100, 1)
    print(f'Total indices among all numerical features are: {len(all_indices)} ({outlier_percentage} %) of the dataset')
    return list(all_indices)
outlier_indices = get_outlier_indices(outlier_info, outliers_iqr, outliers_zscore)
plot_feature_with_outlier_methods(train_model3[[categorical_features[0]]+[target_feature]], target_feature, [outliers_iqr, outliers_zscore], sns.boxplot)


print('Dataset shape before drop the outliers', train_model3.shape)
train_model3 = train_model3.drop(index=outlier_indices)
print('Dataset shape after drop the outliers ', train_model3.shape)
Dataset shape before drop the outliers (1482535, 3)
Dataset shape after drop the outliers  (1460644, 3)
Outlier Treatment Model
df_dtypes(train_model3)
replace_inf(train_model3)
replace_inf(test_model3)
fillna(train_model3)
fillna(test_model3)
X_train, X_test, y_train, y_test = split_data_train_test(train_model3, target_feature)
models_result3 = run_models(baseline_models, X_train, X_test, y_train, y_test)
models_result3
for metric in metrics:
    plot_bar_chart(models_result3, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")







best_models3 = get_best_model(models_result3, baseline_models, 'Mean Squared Error')
print('Best Model of Outlier Treatment Models is:', best_models3.__class__.__name__)
Best Model of Outlier Treatment Models is: DecisionTreeRegressor
del train_model3, test_model3, X_train, X_test, y_train, y_test
Imbalanced Data Detection and Treatment
df_dtypes(train)
 	dtype	# columns	columns
0	int64	3	['train_id', 'item_condition_id', 'shipping']
1	float64	1	['price']
2	object	4	['name', 'category_name', 'brand_name', 'item_description']
train_model6 = train.copy()
test_model6  = test.copy()
train_model6, _ = drop_id_feature(train_model6, 'train_id')
test_model6, _  = drop_id_feature(test_model6, 'test_id')
encode_str_features(train_model6)
encode_str_features(test_model6)
df_dtypes(train_model6)
 	dtype	# columns	columns
0	int64	2	['item_condition_id', 'shipping']
1	float64	1	['price']
2	object	4	['name', 'category_name', 'brand_name', 'item_description']
train_model6 = train_model6.drop(columns=train_model6.select_dtypes(exclude=[np.number]).columns)
test_model6  = test_model6.drop(columns=test_model6.select_dtypes(exclude=[np.number]).columns)
df_dtypes(train_model6)
 	dtype	# columns	columns
0	int64	2	['item_condition_id', 'shipping']
1	float64	1	['price']
from collections import Counter
from imblearn.over_sampling import SMOTE

def oversampling_imbalanced_data(df, target_feature, k_neighbors=5):
    if k_neighbors >= min(Counter(df[target_feature]).values()):
        return df
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    oversampler = SMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                              pd.DataFrame(y_resampled, columns=[target_feature])], axis=1)
    return df_resampled
def plot_imbalanced_feature(df, df_resampled, target_feature):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x=df[target_feature])
    plt.title('Class Distribution Before Oversampling')
    plt.xlabel(target_feature)
    plt.subplot(1, 2, 2)
    sns.countplot(x=df_resampled[target_feature])
    plt.title('Class Distribution After Oversampling')
    plt.xlabel(target_feature)

    plt.tight_layout()
    plt.show()
def plot_imbalanced_features(df, features):
    for col in features:
        if df[col].nunique() == 1:
            continue
        df_resampled = oversampling_imbalanced_data(df, col)
        if df.shape != df_resampled.shape:
            print(f'col: {col}, df.shape: {df.shape}, df_resampled.shape: {df_resampled.shape}')
            plot_imbalanced_feature(df, df_resampled, col)
def oversampling_data(df, features):
    for col in features:
        if df[col].nunique() == 1:
            continue
        df_resampled = oversampling_imbalanced_data(df, col)
        df = df_resampled.copy()
    return df
oversamplying_features = get_categorical_features(train_model6.select_dtypes(exclude=['float']), 4)
build_my_info_table(train_model6[oversamplying_features])
#	column	dtype	count	mean	std	min	25%	50%	75%	max	IQR	nunique	unique	mode	mode #	mode %	null #	null %
0	0	shipping	int64	1482535	0.4	0.5	0	0.0	0.0	1.0	1	1	2	[1, 0]	0	819435	55.3	0	0.0
plot_imbalanced_features(train_model6, oversamplying_features)
col: shipping, df.shape: (1482535, 3), df_resampled.shape: (1638870, 3)

Imbalanced Data Treatment Model
df_dtypes(train_model6)
 	dtype	# columns	columns
0	int64	2	['item_condition_id', 'shipping']
1	float64	1	['price']
replace_inf(train_model6)
replace_inf(train_model6)
fillna(train_model6)
fillna(train_model6)
X_train, X_test, y_train, y_test = split_data_train_test(train_model6, target_feature)
models_result6 = run_models(baseline_models, X_train, X_test, y_train, y_test)
Model LinearRegression begining now ...
Model LinearRegression run in                      0.0 minutes
Model Ridge begining now ...
Model Ridge run in                                 0.0 minutes
Model DecisionTreeRegressor begining now ...
Model DecisionTreeRegressor run in                 0.0 minutes
models_result6
Model	Mean Absolute Error	Mean Squared Error	Mean Absolute Percentage Error	Median Absolute Error	Max Error	R^2	Explained Variance
0	LinearRegression	18.217559	1549.500110	7.321885e+13	13.242494	1976.838806	0.009370	0.009397
1	Ridge	18.217560	1549.500110	7.321885e+13	13.242485	1976.838794	0.009370	0.009397
2	DecisionTreeRegressor	18.180687	1546.531774	7.394878e+13	13.396501	1978.603499	0.011268	0.011295
for metric in metrics:
    plot_bar_chart(models_result6, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")







best_models6 = get_best_model(models_result6, baseline_models, 'Mean Squared Error')
print('Best Model of Imbalanced Data Treatment Models is:', best_models6.__class__.__name__)
Best Model of Imbalanced Data Treatment Models is: DecisionTreeRegressor
del train_model6, test_model6, X_train, X_test, y_train, y_test
Combine All Enhancements
df_dtypes(train)
 	dtype	# columns	columns
0	int64	3	['train_id', 'item_condition_id', 'shipping']
1	float64	1	['price']
2	object	4	['name', 'category_name', 'brand_name', 'item_description']
train_model4 = train.copy()
test_model4  = test.copy()
train_model4, train_id = drop_id_feature(train_model4, 'train_id')
test_model4,  test_id  = drop_id_feature(test_model4, 'test_id')
add_columns(train_model4)
add_columns(test_model4)
df_dtypes(train_model4)
 	dtype	# columns	columns
0	bool	4	['missing_category', 'missing_brand', 'missing_description', 'price_in_name']
1	int64	2	['item_condition_id', 'shipping']
2	float64	1	['price']
3	object	9	['name', 'category_name', 'brand_name', 'item_description', 'first_category', 'second_category', 'third_category', 'price_in_description', 'descr_len']
encode_str_features(train_model4)
encode_str_features(test_model4)
df_dtypes(train_model4)
 	dtype	# columns	columns
0	int64	8	['item_condition_id', 'shipping', 'first_category', 'missing_category', 'missing_brand', 'missing_description', 'price_in_description', 'price_in_name']
1	float64	1	['price']
2	object	7	['name', 'category_name', 'brand_name', 'item_description', 'second_category', 'third_category', 'descr_len']
outlier_info = outlier_summary(train_model4, outliers_iqr, outliers_zscore)
outlier_indices = get_outlier_indices(outlier_info, outliers_iqr, outliers_zscore)
Total indices among all numerical features are: 21891 (1.5 %) of the dataset
print('Dataset shape before drop the outliers', train_model4.shape)
train_model4 = train_model4.drop(index=outlier_indices)
print('Dataset shape after drop the outliers ', train_model4.shape)
Dataset shape before drop the outliers (1482535, 16)
Dataset shape after drop the outliers  (1460644, 16)
train_model4 = train_model4.drop(columns=train_model4.select_dtypes(exclude=[np.number]).columns)
test_model4  = test_model4.drop(columns=test_model4.select_dtypes(exclude=[np.number]).columns)
df_dtypes(train_model4)
 	dtype	# columns	columns
0	int64	8	['item_condition_id', 'shipping', 'first_category', 'missing_category', 'missing_brand', 'missing_description', 'price_in_description', 'price_in_name']
1	float64	1	['price']
oversamplying_features = get_categorical_features(train_model4.select_dtypes(exclude=['float']), 4)
build_my_info_table(train_model4[oversamplying_features])
plot_imbalanced_features(train_model4, oversamplying_features)
col: shipping, df.shape: (1460644, 9), df_resampled.shape: (1610304, 9)

col: price_in_name, df.shape: (1460644, 9), df_resampled.shape: (2906694, 9)

print('Dataset shape before apply the oversampling', train_model4.shape)
train_model4 = oversampling_data(train_model4, oversamplying_features)
print('Dataset shape after apply the oversampling ', train_model4.shape)
Dataset shape before apply the oversampling (1460644, 9)
Dataset shape after apply the oversampling  (3204470, 9)
skewed_features, non_skewed_features, skew_df = get_skewed_features(train_model4)
print(f'skewed_features:     {len(skewed_features)}\n{skewed_features}\n')
print(f'non_skewed_features: {len(non_skewed_features)}\n{non_skewed_features}\n')
skewed_features:     1
['price']

non_skewed_features: 0
[]

transform_skewed_features(train_model4, skewed_features)
transform_non_skewed_features(test_model4, non_skewed_features)
transform_skewed_features(train_model4, skewed_features)
transform_non_skewed_features(test_model4, non_skewed_features)
Combine All Enhancements Model
df_dtypes(train_model4)
 	dtype	# columns	columns
0	int64	8	['item_condition_id', 'first_category', 'missing_category', 'missing_brand', 'missing_description', 'price_in_description', 'shipping', 'price_in_name']
1	float64	1	['price']
replace_inf(train_model4)
replace_inf(test_model4)
fillna(train_model4)
fillna(test_model4)
X, y = split_data_X_y(train_model4, target_feature)
#importance_df = feature_selection(train_model4, mutual_info_regression)
selected_features = importance_df['Feature']
X_selected  = X[selected_features]
test_model4 = test_model4[selected_features]
df_dtypes(X_selected)
 	dtype	# columns	columns
0	int64	8	['shipping', 'first_category', 'item_condition_id', 'price_in_description', 'missing_category', 'missing_brand', 'missing_description', 'price_in_name']
X_train_test_model4 = pd.concat([X_selected, test_model4])
X_train_test_model4 = one_hot_encoding(X_train_test_model4)
X_selected  = X_train_test_model4[:len(X_selected)]
test_model4 = X_train_test_model4[-len(test_model4):]
df_dtypes(X_selected)
 	dtype	# columns	columns
0	float64	18	['shipping_1', 'first_category_1', 'first_category_2', 'first_category_3', 'first_category_4', 'first_category_5', 'first_category_6', 'first_category_7', 'first_category_8', 'first_category_9', 'item_condition_id_2', 'item_condition_id_3', 'item_condition_id_4', 'item_condition_id_5', 'price_in_description_1', 'price_in_description_2', 'missing_description_1', 'price_in_name_1']
X_train, X_test, y_train, y_test = split_data_train_test(pd.concat([X_selected, y], axis=1), target_feature)
models_result4 = run_models(baseline_models, X_train, X_test, y_train, y_test)
models_result4
for metric in metrics:
    plot_bar_chart(models_result4, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")







best_models4 = get_best_model(models_result4, baseline_models, 'Mean Squared Error')
print('Best Model of Combine All Enhancements Models is:', best_models4.__class__.__name__)
Best Model of Combine All Enhancements Models is: DecisionTreeRegressor
Business Insights
def plot_top_n(df, xlabel, ylabel, n=25, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    sns.barplot(x=df.index, y=df)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    yminlim = df.min()
    ymaxlim = df.max()
    yrange = (ymaxlim - yminlim) * .05
    yminlim -= yrange
    ymaxlim += yrange
    plt.ylim(yminlim, ymaxlim)
    plt.title(f'Top {n} {xlabel} based on the {ylabel}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
def plot_single_feature_analysis(df, features, n=5):
    for col in features:
        top_artists = df.groupby(col)[target_feature].mean().sort_values(ascending=False)
        plot_top_n(top_artists.head(n), col, f'mean of {target_feature}', n)
plot_single_feature_analysis(train, categorical_features)


def plot_multiple_feature_analysis(df, columns, plot_func, y_col, size=(12, 4), n_col=1):
    if len(columns) == 0:
        return
    n_rows = (len(columns) + n_col-1) // n_col
    fig, axes = plt.subplots(n_rows, n_col, figsize=(size[0]*n_col, size[1]*n_rows))
    if len(columns) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, label in enumerate(columns):
        plot_func(df[df[y_col] >= df[y_col].median()], x=label, kde=True, color='green', label='Above Median', element='step', alpha=0.5, ax=axes[i])
        plot_func(df[df[y_col] <  df[y_col].median()], x=label, kde=True, color='red',   label='Below Median', element='step', alpha=0.5, ax=axes[i])
        axes[i].set_xlabel(label)
        axes[i].set_title(f'Above Median vs. Below Median {y_col} for {label}')

    for j in range(i+1, n_rows*n_col):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
plot_multiple_feature_analysis(train, numerical_features, sns.histplot, target_feature)

Other Predictive Models
X_selected5 = X_selected.copy()
test_model5 = test_model4.copy()
del train_model4, test_model4, X_selected, X_train, X_test, y_train, y_test, X_train_test_model4
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder
def get_preprocessor(df):
    numerical_features   = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(exclude=[np.number]).columns
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline,   numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor
preprocessor = get_preprocessor(X_selected5)
X_selected5   = pd.DataFrame(preprocessor.fit_transform(X_selected5))
preprocessor = get_preprocessor(test_model5)
test_model5  = pd.DataFrame(preprocessor.fit_transform(test_model5))
from sklearn.linear_model    import LinearRegression, \
                                    Ridge
from sklearn.ensemble        import RandomForestRegressor, \
                                    GradientBoostingRegressor, \
                                    AdaBoostRegressor, \
                                    ExtraTreesRegressor
from sklearn.svm             import SVR
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neighbors       import KNeighborsRegressor
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor

models = {
    'XGBRegressor':              XGBRegressor(),
    'LGBMRegressor':             LGBMRegressor(verbose=-1),
    'LinearRegression':          LinearRegression(),
    'Ridge':                     Ridge(alpha=10),
    'DecisionTreeRegressor':     DecisionTreeRegressor(),
    'AdaBoostRegressor':         AdaBoostRegressor(),
}
#    'GradientBoostingRegressor': GradientBoostingRegressor(),
#    'ExtraTreesRegressor':       ExtraTreesRegressor(),
#    'KNeighborsRegressor':       KNeighborsRegressor(),
#    'RandomForestRegressor':     RandomForestRegressor(),
#    'SupportVectorRegressor':    SVR(),
from sklearn.metrics         import mean_absolute_error, \
                                    mean_squared_error, \
                                    mean_absolute_percentage_error, \
                                    median_absolute_error, \
                                    max_error, \
                                    r2_score, \
                                    explained_variance_score

def evaluate_model(y_tests, y_preds, kf):
    mae, mse, mape, medae, me, r2, ev = 0, 0, 0, 0, 0, 0, 0
    for y_test, y_pred in zip(y_tests, y_preds):
        mae   += mean_absolute_error(y_test, y_pred)
        mse   += mean_squared_error(y_test, y_pred)
        mape  += mean_absolute_percentage_error(y_test, y_pred)
        medae += median_absolute_error(y_test, y_pred)
        me    += max_error(y_test, y_pred)
        r2    += r2_score(y_test, y_pred)
        ev    += explained_variance_score(y_test, y_pred)

    result = {
        'Mean Absolute Error': mae / kf,
        'Mean Squared Error': mse / kf,
        'Mean Absolute Percentage Error': mape / kf,
        'Median Absolute Error': medae / kf,
        'Max Error': me / kf,
        'R^2': r2 / kf,
        'Explained Variance': ev / kf,
    }
    return result
def run_model(model, X, y, kf):
    kfold = KFold(n_splits=kf, shuffle=True, random_state=42)
    y_tests, y_preds = [], []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_tests.append(y_test)
        y_preds.append(y_pred)

    return y_tests, y_preds
import time

def run_models(models, X, y, kf=5):
    results = {}
    for name, model in models.items():
        print(f'Model {name} begining now ...')
        begin_time = time.time()
        y_tests, y_preds = run_model(model, X, y, kf)
        results[name] = evaluate_model(y_tests, y_preds, kf)
        end_time = time.time()
        duration = round((end_time - begin_time) / 60, 2)
        print(f'Model {name} run in'.ljust(50), f'{duration} minutes')
    results = pd.DataFrame(results).T
    results = results.reset_index()
    results = results.rename(columns={'index': 'Model'})
    return results
models_result5 = run_models(models, X_selected5, y)
models_result5
for metric in metrics:
    plot_bar_chart(models_result5, x=metric, y='Model', xlabel=metric, ylabel='Model', title=f"Models Comparison using {metric} metric")







best_models5 = get_best_model(models_result5, models, 'Mean Squared Error')
print('Best Model of Other Predictive Models is:', best_models5.__class__.__name__)
Best Model of Other Predictive Models is: DecisionTreeRegressor
Submission
submission = pd.read_csv(f'{dir_path}/sample_submission_stg2.csv')
submission.head()
submission = pd.DataFrame({
    'test_id' : test_id,
    target_feature : np.expm1(best_models5.predict(test_model5)),
})
submission.head()
submission.to_csv("submission.csv", index = False, header = True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.genel_resim_analizi import check_df
from utils.degiskenlerin_yakalanmasi import grab_col_names
from utils.sayisal_degisken_analizi import num_summary
from utils.kategorik_degisken_analizi import cat_summary
from utils.hedef_degisken_analizi import target_summary_with_num, target_summary_with_cat
from utils.aykiri_degisken_analizi import check_outlier, outlier_thresholds, grab_outliers, remove_outlier, \
    local_outlier_factor, replace_with_thresholds
from utils.eksik_deger_analizi import missing_values_table, missing_vs_target
from utils.korelasyon_analizi import high_correlated_cols, drop_high_correlated_cols
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Özellik Mühendisliği/Ödevler/diabetes.csv")

# Görev 1: Keşifçi Veri Analizi

# Genel resmi inceleyiniz
check_df(df)

# Numerik ve kategorik değişkenleri yakalayınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerik ve kategorik değişkenlerin analizini yapınız.
for col in num_cols:
    num_summary(df, col)

for col in cat_cols:
    cat_summary(df, col)

# Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

# Aykırı gözlem analizi yapınız.
for col in num_cols:
    print(col, check_outlier(df, col), sep=":")

# Eksik gözlem analizi yapınız.
"""na_columns = missing_values_table(df)
missing_vs_target(df, "Outcome", na_columns)

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()"""

# Korelasyon analizi yapınız
high_correlated_cols(df, plot=True, corr_th=0.6)

# Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp

na_columns = ["BloodPressure", "Glucose", "SkinThickness", "Insulin", "BMI"]
df[na_columns] = df[na_columns].apply(lambda x: np.where(x == 0, np.nan, x), axis=0)

std_scaler = StandardScaler()

dff = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

df = pd.DataFrame(std_scaler.inverse_transform(dff), columns=dff.columns)

# sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

for col in num_cols:
    replace_with_thresholds(df, col, 0.05, 0.95)

check_df(df)

for col in num_cols:
    print(check_outlier(df, col))
# Yeni değişkenler oluşturunuz
df.head()

df["birth_frequency"] = ((df["Age"] - 18) / df["Pregnancies"]).apply(lambda x: np.where(x == np.inf, 0, x))

df.groupby(["Outcome"])["birth_frequency"].mean()

"""df_outliers = local_outlier_factor(df, num_cols)
df = df.loc[~df.index.isin(df_outliers.index)]
"""
# Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerik değişkenler için standartlaştırma yapınız.
s_scaler = StandardScaler()
df_scaled = pd.DataFrame(s_scaler.fit_transform(df[num_cols]), columns=df[num_cols].columns)

#  Model oluşturunuz.
x = df_scaled
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy_score(y_test, y_pred)
    
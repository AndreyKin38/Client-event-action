import datetime

import pandas as pd
import tqdm
import dill

from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from preprocessor import *


binary_args = {
        'utm_source': [
            'QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
            'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
        ],
        'utm_medium': ['organic', 'referral', '(none)']
    }


ohe_args = {
    'device_category': ['mobile', 'desktop', 'tablet'],
    'device_brand': ['Apple', 'Samsung', 'Xiaomi', 'Huawei'],
    'device_os': ['Android', 'iOS', 'Windows', 'Macintosh', 'Linux'],
    'device_screen_resolution': ['1920x1080', '2560x1440', '3840x2160', '2560x1080', '3440x1440', '1440x900', '1280x720'],
    'device_browser': ['Chrome', 'Safari', 'YaBrowser', 'Samsung', 'Android', 'Opera', 'Firefox', 'Edge'],
    'geo_country': ['Russia'],
    'geo_city': [
        'Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Nizhny Novgorod',
        'Kazan', 'Samara', 'Krasnodar', 'Ufa',
        'Novosibirsk', 'Krasnoyarsk', 'Chelyabinsk', 'Tula',
        'Rostov-on-Don', 'Irkutsk', 'Vladivostok', 'Grozny'
    ]
}

columns_to_drop = ['session_id', 'client_id']


def main():
    df_full = pd.read_parquet("../preprocessed_data/df_full.parquet")
    target = pd.read_parquet("../preprocessed_data/target.parquet")

    df_full = pd.merge(left=target, right=df_full, on='session_id')

    df_train, df_test = get_train_test_datasets(
        df=df_full,
        random_state=42
    )

    x_train = df_train.drop(['target'], axis=1)
    y_train = df_train['target']

    x_test = df_test.drop(['target'], axis=1)
    y_test = df_test['target']

    preprocessor = Pipeline(steps=[
        ('dropper', ColumnsToDrop(columns_to_drop)),
        ('binary_classificator', BinaryClassificator(binary_args)),
        ('category_classificator', CategoryClassificator(ohe_args)),
        ('encoder', Encoder(ohe_args)),
        ('scaler', MinMaxScaler())
    ])

    hgbr = HistGradientBoostingClassifier(
        max_depth=20,
        learning_rate=0.08404746250511728,
        max_iter=90
    )

    cat = CatBoostClassifier(
        eval_metric='AUC',
        max_depth=6,
        learning_rate=0.010382221313419943,
        n_estimators=2484,
        verbose=False
    )

    best_score = 0
    best_model = None

    for model in tqdm.tqdm([hgbr, cat]):
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, x_train, y_train, cv=4, scoring='roc_auc', n_jobs=-1)

        print()
        print(f"mean_score: {score.mean()}")
        print(f"std: {score.std()}")
        print(f"best_model: {pipe['classifier'].__class__.__name__}\n")

        if score.mean() > best_score:
            best_score = score.mean()
            best_model = pipe

    print(f"roc_auc_score: {best_score}")
    print(f"best_model: {best_model['classifier'].__class__.__name__}\n")

    best_model.fit(x_train, y_train)
    test_pred_probs = best_model.predict_proba(x_test)[:, 1]
    test_score = roc_auc_score(y_test, test_pred_probs)

    print(f"test_pred_probs: {test_score}")

    with open('event_action_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_model,
            'metadata': {
                'name': 'Event action pipeline',
                'date': datetime.datetime.now(),
                'type': type(best_model.named_steps['classifier']).__name__,
                'roc_auc': best_score
            }
        }, file)


if __name__ == "__main__":
    main()







import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from guppy import hpy
import joblib
from random import sample

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import set_config as sklearn_config
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from IPython.core.display import HTML
from IPython.display import Image, display


def cross_validation_ts(model, X_training, y_training, model_name, kfold=5, verbose=False, single_score=None):
    """
    Cross Validation para Time Series (Prequential expanding).
    Retorna um DataFrame com todas as métricas calculadas por `ml_error`.
    """
    result_list = []

    for k in range(kfold, 0, -1):
        if verbose:
            print("K-fold {}/{}".format(kfold-k+1, kfold))
            
        # start and end date of validation dataset
        val_start_date = X_training['date'].max() - dt.timedelta(weeks=k * 6)
        val_end_date = val_start_date + dt.timedelta(weeks=6)
        
        # filtering
        train_cv = X_training.query('date < @val_start_date')
        val_cv = X_training.query('date >= @val_start_date & date <= @val_end_date')
        
        # training dataset
        X_train_cv = train_cv.drop(columns=["date"])
        y_train_cv = y_training[train_cv.index]
        
        # validation dataset
        X_val_cv = val_cv.drop(columns=["date"])
        y_val_cv = y_training[val_cv.index]

        # model training
        model.fit(X_train_cv, y_train_cv)
        
        # prediction
        y_hat_cv = model.predict(X_val_cv)
        
        # performance and append to result_list
        result_list += [ml_error(model_name, np.expm1(y_val_cv), np.expm1(y_hat_cv))]
    
    df_result = pd.concat(result_list)
    mean_cv = df_result.mean(axis=0).rename('mean').round(3).astype(str)
    std_cv = df_result.std(axis=0).rename('std').round(3).astype(str)
    
    # concat columns
    df_result = mean_cv + " (" + std_cv + ")"
    df_result = df_result.to_frame().T
    
    # insert the model's name
    df_result['Model Name'] = model_name
    
    if single_score is None:
        return df_result 
    else:
        return np.float64(df_result[single_score].str.split()[0][0])


def mean_absolute_percentage_error(y, y_hat):
    """
    Retorna o Mean Absolute Percentage Error (MAPE)
    """
    return np.mean(np.abs(( y - y_hat) / y))


def root_mean_square_percentage_error(y, y_hat):
    """
    Retorna o Root Mean Square Percentage Error (RMSPE)
    """
    return np.sqrt(np.mean(np.square(( y - y_hat) / y)))


def mean_percentage_error(y, y_hat):
    """
    Retorna o Mean Percentage Error (MPE)
    Utilizado para visualizar o bias positivo ou negativo do modelo
    """
    return np.mean(( y - y_hat) / y)


def ml_error(model_name, y, y_hat):
    """
    Retorna um dataframe com 1 linha contendo as 3 métricas implementadas
    *É necessário converter y e y_hat para a inversa da transformação que foi aplicado em 5.3.2. 
    * Nesse projeto a inversa de np.log1p é np.expm1.
    """
    mae = mean_absolute_error(y, y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    rmspe = root_mean_square_percentage_error(y, y_hat)
    mpe = mean_percentage_error(y, y_hat)
    
    return pd.DataFrame({
        'Model Name': model_name, 
        'MAE': mae, 'MAPE': mape, 
        'RMSE': rmse, 'RMSPE': rmspe,
        'MPE': mpe}, 
        index=[0]
    )


def cramer_v(x, y):
    """ Método de Cramér's V
    Mede a correlação entre duas váriaveis categóricas ou entre categórica e numérica
    Referência : https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    confusion_matrix = pd.crosstab(x, y).values
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    # correção de bias
    chi2corr = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))
    kcorr = k - (k - 1)**2 / (n - 1)
    rcorr = r - (r - 1)**2 / (n - 1)
    
    return np.sqrt((chi2corr / n) / (min(kcorr - 1, rcorr - 1)))


def jupyter_settings():    
    # plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = "bold"
    plt.rcParams['xtick.labelsize'] = 14
    
    display( HTML( '<style>.container { width:100% !important; }</style>') )
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    
    sns.set()
    sklearn_config(display='diagram')
    warnings.filterwarnings('ignore')


class MemoryProfiler():
    """
    Context Manager para memory profile. 
    EX de uso:
    with MemoryProfiler() as mp:
        X_train = joblib.load("data/X_train_transformed.joblib.bz2") 
        mp.print_heap()

    REF: https://smira.ru/wp-content/uploads/2011/08/heapy.html
    """
    def __init__(self):
        self.hpy = hpy()
        self.hpy.setrelheap()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def print_heap(self):
        print(self.hpy.heap())

    def size_obj(self, obj):
        print(self.hpy.iso(obj).size/(1024*1024), "MB")

def memory_profiler(obj=None, hp=None):
    # hp é um objeto do tipo hpy.setrelheap(), que pode ser chamado antes da execução do código
    h = hpy() if hp is None else hp
    return h.heap() if obj is None else h.iso(obj)



def kfold_cv_ts(X_training, kfold=5):
    """ 
    Divide o dataset em k-folds no mesmo formato de cross_validation_ts()
    Retorna uma lista com a data inicial e final de cada fold.
    """
    result_list = []

    for k in range(kfold, 0, -1):            
        # start and end date of validation dataset
        val_start_date = X_training['date'].max() - dt.timedelta(weeks=k * 6)
        val_end_date = val_start_date + dt.timedelta(weeks=6)
        
        result_list.append(((X_training['date'].min(), val_start_date), (val_start_date, val_end_date)))
        
    return result_list

def print_cv_ts(X_training, kfold=5):
    """ 
    Imprime um gráfico de barras horizontal contendo a quantidade de 
        semanas por cada fold.
    """
    plt.Figure(figsize=(14, 6))

    df_plot = (pd.DataFrame(kfold_cv_ts(training,kfold), index=range(1, kfold+1))
                .sort_index(ascending=False)
                .rename(columns={0:"Treino", 1:"Validação"}))

    df_weeks = df_plot.apply(lambda row: row.apply(lambda col: (col[1] - col[0]).days//7))

    ax = df_weeks.plot(kind="barh", stacked=True)

    xmax = df_weeks.sum(axis=1).max()
    for bar in ax.patches:
        y = bar.get_y() + bar.get_height() / 2
        width = bar.get_width()
        ax.text(bar.get_x() + width/2, y, f"{int(width)}", ha='center', va='center', fontsize=15, fontweight="bold")
        
        if width != 6:
            d = plt.vlines(width, linestyle="--", ymin=bar.get_y()-1,
                           ymax=bar.get_y()+1, color="gray", lw=1.5, alpha=0.6)

    plt.xticks([0, xmax//2, xmax])
    plt.title("Prequential Expanding - OOS")
    plt.xlabel("Semanas")
    plt.ylabel("Fold")
    plt.xlim(0, df_weeks.sum(axis=1).max())
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()


def preprocess_full_ds(X_full: pd.DataFrame, y_full: pd.Series, save_parameters=True):
    """
    Aplicar as transformações, scalers e encodings no conjunto completo de treino, além de
        salvar essas transformações para aplicar nos dados recebidos em produção.
    Retorna uma tupla com arrays numpy com os dados (X, y) processados.
    """
    df = X_full.copy()
    def apply_scaler(scaler, col_name):
        nonlocal df
        df[col_name] = scaler.fit_transform(df[[col_name]].values)
        if save_parameters:      
            joblib.dump(scaler, f'parameter/{col_name}_scaler.pkl.bz2')

    def apply_onehot(encoder, col_name, save_parameter=True):
        nonlocal df
        tr = encoder.fit_transform(df[[col_name]].values)
        columns = [*map(lambda c: col_name+"_"+c, encoder.categories_[0])]
        df = df.join(pd.DataFrame(tr.toarray(), columns=columns, index=df.index))
        if save_parameters:      
            joblib.dump(encoder, f'parameter/{col_name}_encoding.pkl.bz2')

    def apply_ord_encoder(encoder, col_name, save_parameter=True):
        nonlocal df
        df[col_name] = encoder.fit_transform(df[[col_name]].values)
        if save_parameters:      
            joblib.dump(encoder, f'parameter/{col_name}_encoding.pkl.bz2')

    def cyclic_transform(col_name, n):
        nonlocal df
        df[col_name + '_sin'] = np.sin(df[col_name].values * (2 * np.pi / n))
        df[col_name + '_cos'] = np.cos(df[col_name].values * (2 * np.pi / n))

    rs_features = [
        'competition_distance', 
        'competition_time_month'
    ]

    mms_features = [ 
        'promo_time_week',  
        'year',  
        'promo2_since_week', 
        'promo2_since_year',
        'competition_open_since_year'
    ]

    # Robust scaler: possui muitos outliers
    rs = RobustScaler()
    for feature in rs_features:
        apply_scaler(rs, feature)

    # MinMax scaler: Possui poucos ou nenhum outlier
    mms = MinMaxScaler()
    for feature in mms_features:
        apply_scaler(mms, feature)

    # OneHotEncoder
    oh = OneHotEncoder()
    apply_onehot(oh, "state_holiday")
    apply_onehot(oh, "store_type")

    # OrdinalEncoder
    oe = OrdinalEncoder(categories=[['basic', 'extended', 'extra']])
    apply_ord_encoder(oe, "assortment")

    # Cyclic transformation
     # day_of_week
    cyclic_transform('day_of_week', 7)
    cyclic_transform('month', 12)
    cyclic_transform('day', 30)
    cyclic_transform('week_of_year', 52)

    cols_selected = joblib.load("parameter/cols_selected.pkl.bz2")
    
    return df[cols_selected].values, np.log1p(y_full).values.ravel()


def random_search(model, params, x_train_dt, y_train, model_name="---", kfold=5, max_eval=10, cv_verbose=False):
    final_result = pd.DataFrame()
    param_list = []
    params_hash = []

    try:
        for i in range(max_eval):

            hp = {k: sample(v, 1)[0] for k, v in params.items()}
            
            # Para não escolher o mesmo conjunto de parâmetros
            if hash(tuple(hp.items())) in params_hash:
                i -= 1
                continue
                
            param_list.append(hp)

            print(i, ":", hp)

            # performance
            result = helpers.cross_validation_ts(model, x_train_dt, y_train,
                                         f"{model_name}-{i}", kfold=kfold, verbose=cv_verbose)
                
            final_result = pd.concat([final_result, result])
            
            display(result)
            
            print("-"*100,"\n\n")
            params_hash.append(hash(tuple(hp.items())))
    finally:
        display(final_result)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statistics as st
from sklearn.model_selection import GridSearchCV

thisFolder = os.path.dirname(__file__)
dataFolder = os.path.join(thisFolder, "dados")

# Seleciona as pastas de cada operação (operação normal e falhas)
desbalanceadaFolder = os.path.join(dataFolder, "helice_desbalanceada")
quebradaFolder = os.path.join(dataFolder, "helice_quebrada")
normalFolder = os.path.join(dataFolder, "normal")
invertidaFolder = os.path.join(dataFolder, "rotacao_invertida")

folders = {
    "normal": normalFolder,
    "desbalanceada": desbalanceadaFolder,
    "quebrada": quebradaFolder,
    "invertida": invertidaFolder
}

def transitory_stacionary(df):
    
    # Adquire a localização em índice da coluna de "Corrente_A" do dataframe utilizado como argumento
    # Há a soma "+ 1" pois, para a indexação em cada linha através do da função "itertuples()" do pandas
    loc_i = df.columns.get_loc("Corrente_A") + 1

    # Obtém os intervalos em que estão localizados 30% e 70% dos valores de corrente armazenados
    q1 = df["Corrente_A"].quantile(0.30)
    q3 = df["Corrente_A"].quantile(0.70)

    # Cria um novo dataframe cujos valores são aqueles entre q1 e q3, ou seja, 40% dos dados centrais
    df_40 = df[(df["Corrente_A"] >= q1) & (df["Corrente_A"] <= q3)]

    # "Flag" criada para indicar que o ponto que termina o transitório inicial e inicia a seção do regime permanente já foi identificado
    flag_stac = False

    # "Flag" para indicar que o ponto que termina o regime permanente e inicia o transitório final ainda não foi identificado
    flag_transitory_1 = True

    # Valor em que o transitório 2 será armazenado
    transitory_2 = None

    # Média do dataframe com os 40% dos dados centrais
    mean = df_40["Corrente_A"].mean()
    
    # Para cada linha (row) no dataframe, numerando as iterações pela variável "idx"...
    for idx, row in enumerate(df.itertuples()):

        # Verifica se o valor de corrente está entre 99% e 130% da média, além de verificar se flag_transitory_1 == True
        if ((np.abs(row[loc_i]) >= 0.99*mean) & (np.abs(row[loc_i]) <= 1.3*mean) & flag_transitory_1):
            
            # O valor de índice atual somado a 10 posições é armazenado na variável "transitory_1"
            transitory_1 = idx + 12

            # As flags são atualizadas
            flag_stac = True
            flag_transitory_1 = False

        # Se a flag_stac == True... 
        if flag_stac:

            # Realiza a verificação se o valor atual de corrente está entre 30 e 82% da média
            if ((np.abs(row[loc_i]) <= 0.82*mean) & (np.abs(row[loc_i]) >= 0.3*mean)):
                
                # Armazena o índice atual subtraído de 8 posições na variável transitory_2
                transitory_2 = idx - 8
                break

    # O dataframe referente ao transitório inicial é definido
    df_0 = df.iloc[:transitory_1]

    # Se for for identificado o transitório final...
    if transitory_2 != None:

        # O dataframe do regime permanente é definido entre as duas posições armazenadas anteriormente
        df_1 = df.iloc[transitory_1:transitory_2]

        # O dataframe do transitório final é definido
        df_2 = df.iloc[transitory_2:]
    
    # Caso contrário...
    else:
        
        # O dataframe do regime permanente é definido entre a posição do "transitory_1" até o final do dataframe
        df_1 = df.iloc[transitory_1:]
        # O dataframe do transitório final é definido como "None" por não ter sido identificado
        df_2 = None

    return df_0, df_1, df_2



class statisticsParams:
    def __init__(self, name = None, df = None, tr1_qty = None, regime_qty = None, tr2_qty = None):
        
        # Inicialização dos dicionários vazios
        mean = {}
        median = {}
        max = {}
        min = {}
        desv_padr = {}
        skew = {}
        kurtosis = {}
        mode = {}

        # Caso haja um argumento de df não nulo:
        if df is not None:

            # Adquire a lista com o cabeçalho do dataframe
            header = df.columns.to_list()    

            # Para cada elemento na lista do cabecalho...
            for h in header:
                mean[h] = df[h].mean()  # Adquire o valor da média para o parâmetro atual
                median[h] = df[h].median()  # Adquire o valor da mediana para o parâmetro atual
                max[h] = df[h].max()    # Adquire o valor máximo para o parâmetro atual
                min[h] = df[h].min()    # Adquire o valor mínimo para o parâmetro atual
                desv_padr[h] = st.stdev(df[h])  # Adquire o desvio padrão para o parâmetro atual
                skew[h] = df[h].skew()  # Adquire a assimetria entre os dados para o parâmetro atual
                kurtosis[h] = df[h].kurtosis()  # Adquire o excesso de curtose para o parâmetro atual
                mode[h] = st.mode(df[h])    # Adquire a moda para o parâmetro atual

            # Armazenamento do que foi medido nos parâmetros do objeto
            self.name = name
            self.header = header
            self.mean = mean
            self.median = median
            self.max = max
            self.min = min
            self.desv_padr = desv_padr
            self.skew = skew
            self.kurtosis = kurtosis
            self.mode = mode

            # Armazenamento nos parâmetros do objeto da quantidade de dados para cada regime de operação, sendo esses argumentos da classe.
            self.tr1_qty = tr1_qty
            self.regime_qty = regime_qty
            self.tr2_qty = tr2_qty

        # Caso o argumento "df" seja "None", os parâmetros da função se mantém como definidos no início, strings vazias ou zeros
        else:
            self.name = ""
            self.mean = mean
            self.median = median
            self.max = max
            self.min = min
            self.desv_padr = desv_padr
            self.skew = skew
            self.kurtosis = kurtosis
            self.mode = mode

            self.tr1_qty = 0
            self.regime_qty = 0
            self.tr2_qty = 0


def trainModel(dados_extraidos):
    df = pd.DataFrame(dados_extraidos)

    X = df.drop('condicao', axis=1)
    y = df['condicao']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])

    parametros_busca = {
        'svm__C': [0.1, 1, 10, 100, 1000],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(pipeline, parametros_busca, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    print("Melhor desempenho médio na validação:", grid_search.best_score_)

    melhor_modelo = grid_search.best_estimator_

    y_pred = melhor_modelo.predict(X_test)

    print("\n=== Relatório do Modelo Otimizado ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=melhor_modelo.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=melhor_modelo.classes_, yticklabels=melhor_modelo.classes_)
    plt.title('Matriz de Confusão')
    plt.ylabel('Condição Real')
    plt.xlabel('Condição Prevista')
    plt.show()


def validacaoCruzada(dados_extraidos):
    df = pd.DataFrame(dados_extraidos)

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=42))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        cv=4,
        scoring='accuracy',
        n_jobs=-1
    )

    X = df.drop('condicao', axis=1)
    y = df['condicao']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    grid_search.fit(X_train, y_train)

    print("Melhores hiperparâmetros:", grid_search.best_params_)
    print("Melhor desempenho médio na validação:", grid_search.best_score_)


dados_extraidos = []

for fold in folders: # Para cada chave definida no dicionário "folders" definido anteriormente...

    # Criação de listas vazias.
    df0s = []   # Lista de dataframes do transitório inicial
    df1s = []   # Lista de dataframes do regime permanente
    df2s = []   # Lista de dataframes do transitório final

    tr1_tot_qty = 0 # Quantidade de dados no transitório inicial
    tr2_tot_qty = 0 # Quantidade de dados no regime permanente
    regime_tot_qty = 0  # Quantidade de pontos no transitório final

    print("")
    print("")
    print("==================", fold, "==================")   # Imprime o título da operação atual (normal, desbalanceada, quebrada ou invertida)
    print("")
    
    # Armazena todos os diretórios presentes na pasta atual em uma lista chamada "files_temp"
    files_temp = [os.path.join(folders[fold], x) for x in os.listdir(folders[fold]) if x.endswith(".xlsx")] 

    # Cria uma lista com todos os DataFrames contidos na pasta atual
    dfs = [pd.read_excel(f) for f in files_temp]

    # Acessa cada DataFrame "df_i" da lista de DataFrames da pasta atual, numerando a partir da variável "i"
    for i, df_i in enumerate(dfs):

        # Remoção dos parâmetros não utilizados na análise
        df_i = df_i.drop(columns="Celula3_Kgf")
        df_i = df_i.drop(columns="Tempo")
        df_i = df_i.drop(columns="RPM")
    
        df_0, df_1, df_2 = transitory_stacionary(df_i)  # Segmenta o dataframe "df_i" nos dataframes de transitórios (df_0 e df_2) e regime permanente (df_1)

        # Realiza o somatório da quantidade de dados presentes em cada planilha de uma operação
        tr1_tot_qty = len(df_0)
        regime_tot_qty = len(df_1)
        tr2_tot_qty = len(df_2)
        
        windowLenght = 300
        if not df_1.empty:
            for inicio in range(0, len(df_1), windowLenght):
                
                janela = df_1.iloc[inicio : inicio + windowLenght]
                
                if not janela.empty:
                    estatisticas_arquivo = {'condicao': fold}
                    
                    for coluna in df_1.columns:
                        
                        estatisticas_arquivo[f'{coluna}_mean'] = df_1[coluna].mean()
                        estatisticas_arquivo[f'{coluna}_std'] = df_1[coluna].std()
                        estatisticas_arquivo[f'{coluna}_max'] = df_1[coluna].max()
                        estatisticas_arquivo[f'{coluna}_min'] = df_1[coluna].min()
                        estatisticas_arquivo[f'{coluna}_median'] = df_1[coluna].median()
                        estatisticas_arquivo[f'{coluna}_skew'] = df_1[coluna].skew()
                        estatisticas_arquivo[f'{coluna}_kurtosis'] = df_1[coluna].kurtosis()
                        estatisticas_arquivo[f'{coluna}_rms'] = np.sqrt(np.mean(df_1[coluna]**2))
                    
                    dados_extraidos.append(estatisticas_arquivo)

#validacaoCruzada(dados_extraidos)
trainModel(dados_extraidos)
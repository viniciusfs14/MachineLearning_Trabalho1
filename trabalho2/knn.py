import shutil
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

# Altere para o caminho onde estão seus arquivos originais
BASE_ORIGINAL = Path(r"dataset")

# Pasta onde o código vai criar treino e teste separados
BASE_DIVIDIDA = Path(r"dataset_dividido")

PASTA_TREINO = BASE_DIVIDIDA / "treino"
PASTA_TESTE = BASE_DIVIDIDA / "teste"

# Nomes das pastas/classes
CLASSES = [
    "normal",
    "helice_desbalanceada",
    "helice_quebrada",
    "rotacao_invertida"
]

# Colunas que serão lidas da planilha
COLUNAS_ORIGINAIS = [
    "Tempo",
    "Celula2_Kgf",
    "Tensao_V",
    "Corrente_A",
    "Potencia_W"
]

# Sinais físicos que serão usados para extrair características
COLUNAS_SINAIS = [
    "Celula2_Kgf",
    "Tensao_V",
    "Corrente_A",
    "Potencia_W"
]

TAMANHO_JANELA = 50
TEST_SIZE = 0.35
RANDOM_STATE = 42

N_VIZINHOS = 3


# ============================================================
# 2. SEPARAR ARQUIVOS EM TREINO E TESTE
# ============================================================

def separar_arquivos_treino_teste():
    """
    Copia os arquivos originais para uma nova estrutura:

    dataset_dividido/
        treino/
            normal/
            helice_desbalanceada/
            helice_quebrada/
            rotacao_invertida/

        teste/
            normal/
            helice_desbalanceada/
            helice_quebrada/
            rotacao_invertida/
    """

    if BASE_DIVIDIDA.exists():
        shutil.rmtree(BASE_DIVIDIDA)

    PASTA_TREINO.mkdir(parents=True, exist_ok=True)
    PASTA_TESTE.mkdir(parents=True, exist_ok=True)

    resumo_divisao = []

    for classe in CLASSES:
        pasta_classe = BASE_ORIGINAL / classe

        if not pasta_classe.exists():
            print(f"Aviso: pasta não encontrada: {pasta_classe}")
            continue

        arquivos = sorted([
            arquivo for arquivo in pasta_classe.glob("*.xlsx")
            if not arquivo.name.startswith("~$")
        ])

        if len(arquivos) == 0:
            print(f"Aviso: nenhum arquivo encontrado na classe {classe}")
            continue

        pasta_treino_classe = PASTA_TREINO / classe
        pasta_teste_classe = PASTA_TESTE / classe

        pasta_treino_classe.mkdir(parents=True, exist_ok=True)
        pasta_teste_classe.mkdir(parents=True, exist_ok=True)

        if len(arquivos) == 1:
            arquivos_treino = arquivos
            arquivos_teste = []
        else:
            arquivos_treino, arquivos_teste = train_test_split(
                arquivos,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

        for arquivo in arquivos_treino:
            destino = pasta_treino_classe / arquivo.name
            shutil.copy2(arquivo, destino)

            resumo_divisao.append({
                "Arquivo": arquivo.name,
                "Classe": classe,
                "Conjunto": "treino"
            })

        for arquivo in arquivos_teste:
            destino = pasta_teste_classe / arquivo.name
            shutil.copy2(arquivo, destino)

            resumo_divisao.append({
                "Arquivo": arquivo.name,
                "Classe": classe,
                "Conjunto": "teste"
            })

    df_resumo = pd.DataFrame(resumo_divisao)

    print("\n==============================")
    print("DIVISÃO DOS ARQUIVOS")
    print("==============================")

    print(
        df_resumo
        .sort_values(by=["Classe", "Conjunto", "Arquivo"])
        .to_string(index=False)
    )

    df_resumo.to_csv(BASE_DIVIDIDA / "divisao_treino_teste.csv", index=False)

    print("\nArquivos separados com sucesso!")
    print(f"Resumo salvo em: {BASE_DIVIDIDA / 'divisao_treino_teste.csv'}")


# ============================================================
# 3. CARREGAR PLANILHA
# ============================================================

def carregar_planilha(caminho_arquivo):
    """
    Lê uma planilha Excel, seleciona as colunas necessárias e converte
    o tempo absoluto para tempo relativo.
    """

    df = pd.read_excel(caminho_arquivo, sheet_name="Dados")

    colunas_faltando = [
        coluna for coluna in COLUNAS_ORIGINAIS
        if coluna not in df.columns
    ]

    if len(colunas_faltando) > 0:
        raise ValueError(
            f"O arquivo {caminho_arquivo.name} não possui as colunas: {colunas_faltando}"
        )

    df = df[COLUNAS_ORIGINAIS].copy()

    for coluna in COLUNAS_ORIGINAIS:
        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.dropna()

    df = df.sort_values(by="Tempo").reset_index(drop=True)

    # O tempo original parece ser um timestamp grande.
    # Aqui transformamos para tempo relativo, começando em zero.
    df["Tempo_s"] = df["Tempo"] - df["Tempo"].iloc[0]

    # Intervalo entre amostras
    df["Delta_t"] = df["Tempo_s"].diff().fillna(0)

    return df


# ============================================================
# 4. EXTRAÇÃO DE CARACTERÍSTICAS
# ============================================================

def extrair_caracteristicas(df, classe, nome_arquivo, tamanho_janela=50):
    """
    Divide o ensaio em janelas e extrai características estatísticas.
    """

    dados = []

    for inicio in range(0, len(df) - tamanho_janela + 1, tamanho_janela):
        janela = df.iloc[inicio:inicio + tamanho_janela]

        features = {}

        # Características temporais
        features["Tempo_inicio_s"] = janela["Tempo_s"].iloc[0]
        features["Tempo_fim_s"] = janela["Tempo_s"].iloc[-1]
        features["Duracao_janela_s"] = janela["Tempo_s"].iloc[-1] - janela["Tempo_s"].iloc[0]
        features["Delta_t_medio"] = janela["Delta_t"].mean()
        features["Delta_t_desvio"] = janela["Delta_t"].std()

        # Características dos sinais escolhidos
        for coluna in COLUNAS_SINAIS:
            sinal = janela[coluna]

            features[f"{coluna}_media"] = sinal.mean()
            features[f"{coluna}_desvio"] = sinal.std()
            features[f"{coluna}_min"] = sinal.min()
            features[f"{coluna}_max"] = sinal.max()
            features[f"{coluna}_amplitude"] = sinal.max() - sinal.min()
            features[f"{coluna}_rms"] = np.sqrt(np.mean(sinal ** 2))

        features["Classe"] = classe
        features["Arquivo"] = nome_arquivo

        dados.append(features)

    return pd.DataFrame(dados)


# ============================================================
# 5. MONTAR DATASET A PARTIR DAS PASTAS
# ============================================================

def montar_dataset_por_pasta(pasta_base):
    todos_dados = []

    for classe in CLASSES:
        pasta_classe = pasta_base / classe

        if not pasta_classe.exists():
            continue

        arquivos = sorted([
            arquivo for arquivo in pasta_classe.glob("*.xlsx")
            if not arquivo.name.startswith("~$")
        ])

        for arquivo in arquivos:
            df = carregar_planilha(arquivo)

            df_features = extrair_caracteristicas(
                df=df,
                classe=classe,
                nome_arquivo=arquivo.name,
                tamanho_janela=TAMANHO_JANELA
            )

            todos_dados.append(df_features)

    if len(todos_dados) == 0:
        raise ValueError(f"Nenhum dado encontrado em {pasta_base}")

    dataset = pd.concat(todos_dados, ignore_index=True)

    return dataset


# ============================================================
# 6. TREINAR MODELO KNN
# ============================================================

def treinar_modelo():
    df_treino = montar_dataset_por_pasta(PASTA_TREINO)
    df_teste = montar_dataset_por_pasta(PASTA_TESTE)

    print("\n==============================")
    print("AMOSTRAS POR CLASSE - TREINO")
    print("==============================")
    print(df_treino["Classe"].value_counts())

    print("\n==============================")
    print("AMOSTRAS POR CLASSE - TESTE")
    print("==============================")
    print(df_teste["Classe"].value_counts())

    X_train = df_treino.drop(["Classe", "Arquivo"], axis=1)
    y_train = df_treino["Classe"]

    X_test = df_teste.drop(["Classe", "Arquivo"], axis=1)
    y_test = df_teste["Classe"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_knn = KNeighborsClassifier(
        n_neighbors=N_VIZINHOS,
        metric="minkowski",
        p=2
    )

    modelo_knn.fit(X_train_scaled, y_train)

    y_pred = modelo_knn.predict(X_test_scaled)

    print("\n==============================")
    print("AVALIAÇÃO POR JANELA")
    print("==============================")

    print("\nAcurácia:")
    print(accuracy_score(y_test, y_pred))

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    avaliar_por_arquivo(df_teste, y_pred)

    joblib.dump(modelo_knn, BASE_DIVIDIDA / "modelo_knn_helice.pkl")
    joblib.dump(scaler, BASE_DIVIDIDA / "scaler_knn_helice.pkl")
    joblib.dump(list(X_train.columns), BASE_DIVIDIDA / "colunas_modelo.pkl")

    print("\n==============================")
    print("MODELO SALVO")
    print("==============================")
    print(f"Modelo: {BASE_DIVIDIDA / 'modelo_knn_helice.pkl'}")
    print(f"Scaler: {BASE_DIVIDIDA / 'scaler_knn_helice.pkl'}")
    print(f"Colunas: {BASE_DIVIDIDA / 'colunas_modelo.pkl'}")

    return modelo_knn, scaler


# ============================================================
# 7. AVALIAÇÃO POR ARQUIVO
# ============================================================

def avaliar_por_arquivo(df_teste, y_pred):
    df_resultado = df_teste.copy()
    df_resultado["Predicao"] = y_pred

    print("\n==============================")
    print("AVALIAÇÃO POR ARQUIVO")
    print("==============================")

    acertos = 0
    total = 0

    for arquivo in df_resultado["Arquivo"].unique():
        dados_arquivo = df_resultado[df_resultado["Arquivo"] == arquivo]

        classe_real = dados_arquivo["Classe"].iloc[0]
        predicoes = dados_arquivo["Predicao"].tolist()

        contagem = Counter(predicoes)
        classe_final = contagem.most_common(1)[0][0]

        print("\nArquivo:", arquivo)
        print("Classe real:", classe_real)
        print("Predições por janela:", dict(contagem))
        print("Classe final prevista:", classe_final)

        if classe_real == classe_final:
            print("Resultado: ACERTOU")
            acertos += 1
        else:
            print("Resultado: ERROU")

        total += 1

    if total > 0:
        print("\nAcurácia por arquivo:")
        print(f"{acertos}/{total} = {acertos / total:.4f}")


# ============================================================
# 8. TESTAR UM ARQUIVO NOVO
# ============================================================

def testar_arquivo_novo(caminho_arquivo):
    modelo_knn = joblib.load(BASE_DIVIDIDA / "modelo_knn_helice.pkl")
    scaler = joblib.load(BASE_DIVIDIDA / "scaler_knn_helice.pkl")
    colunas_modelo = joblib.load(BASE_DIVIDIDA / "colunas_modelo.pkl")

    caminho_arquivo = Path(caminho_arquivo)

    df = carregar_planilha(caminho_arquivo)

    df_features = extrair_caracteristicas(
        df=df,
        classe="desconhecida",
        nome_arquivo=caminho_arquivo.name,
        tamanho_janela=TAMANHO_JANELA
    )

    X_novo = df_features.drop(["Classe", "Arquivo"], axis=1)

    X_novo = X_novo[colunas_modelo]

    X_novo_scaled = scaler.transform(X_novo)

    predicoes = modelo_knn.predict(X_novo_scaled)

    contagem = Counter(predicoes)
    classe_final = contagem.most_common(1)[0][0]

    print("\n==============================")
    print("TESTE DE ARQUIVO NOVO")
    print("==============================")

    print("\nArquivo:", caminho_arquivo.name)
    print("Predições por janela:", dict(contagem))
    print("Classe final prevista:", classe_final)

    return classe_final


# ============================================================
# 9. EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    separar_arquivos_treino_teste()

    modelo, scaler = treinar_modelo()

    # Para testar manualmente um arquivo depois, use assim:
    #
    # testar_arquivo_novo(
    #     r"C:\Users\Vinicius\Desktop\dataset_dividido\teste\normal\terceiroteste_helicenormal.xlsx"
    # )
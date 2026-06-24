import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

# >>> ALTERE ESTE CAMINHO PARA A SUA PASTA ORIGINAL <<<
BASE_ORIGINAL = Path(r"dataset")

# Pasta onde o código vai criar treino/teste e salvar resultados
BASE_DIVIDIDA = Path(r"dataset_dividido")

PASTA_TREINO = BASE_DIVIDIDA / "treino"
PASTA_TESTE = BASE_DIVIDIDA / "teste"

CLASSES = [
    "normal",
    "helice_desbalanceada",
    "helice_quebrada",
    "rotacao_invertida"
]

# Colunas que serão lidas do Excel
COLUNAS_ORIGINAIS = [
    "Tempo",
    "Celula2_Kgf",
    "Tensao_V",
    "Corrente_A",
    "Potencia_W"
]

# Variáveis físicas usadas para extrair características
COLUNAS_SINAIS = [
    "Celula2_Kgf",
    "Tensao_V",
    "Corrente_A",
    "Potencia_W"
]

# Parâmetros do modelo
TAMANHO_JANELA = 50
TEST_SIZE = 0.35
RANDOM_STATE = 42
N_VIZINHOS = 3


# ============================================================
# 2. SEPARAR ARQUIVOS EM TREINO E TESTE
# ============================================================

def separar_arquivos_treino_teste():
    """
    Copia os arquivos originais para:
    dataset_dividido/treino/classe
    dataset_dividido/teste/classe
    """

    if BASE_DIVIDIDA.exists():
        shutil.rmtree(BASE_DIVIDIDA)

    PASTA_TREINO.mkdir(parents=True, exist_ok=True)
    PASTA_TESTE.mkdir(parents=True, exist_ok=True)

    resumo_divisao = []

    for classe in CLASSES:
        pasta_classe = BASE_ORIGINAL / classe

        if not pasta_classe.exists():
            print(f"[AVISO] Pasta não encontrada: {pasta_classe}")
            continue

        arquivos = sorted([
            arquivo for arquivo in pasta_classe.glob("*.xlsx")
            if not arquivo.name.startswith("~$")
        ])

        if len(arquivos) == 0:
            print(f"[AVISO] Nenhum arquivo encontrado na classe: {classe}")
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
    print(df_resumo.sort_values(by=["Classe", "Conjunto", "Arquivo"]).to_string(index=False))

    df_resumo.to_csv(BASE_DIVIDIDA / "divisao_treino_teste.csv", index=False)

    print("\nArquivos separados com sucesso!")
    print(f"Resumo salvo em: {BASE_DIVIDIDA / 'divisao_treino_teste.csv'}")


# ============================================================
# 3. CARREGAR PLANILHA
# ============================================================

def carregar_planilha(caminho_arquivo):
    """
    Lê a planilha, seleciona as colunas de interesse e converte
    o tempo absoluto para tempo relativo.
    """

    df = pd.read_excel(caminho_arquivo, sheet_name="Dados")

    colunas_faltando = [c for c in COLUNAS_ORIGINAIS if c not in df.columns]

    if len(colunas_faltando) > 0:
        raise ValueError(
            f"O arquivo {caminho_arquivo.name} não possui as colunas: {colunas_faltando}"
        )

    df = df[COLUNAS_ORIGINAIS].copy()

    # Converte vírgula para ponto e força numérico
    for coluna in COLUNAS_ORIGINAIS:
        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Remove linhas inválidas
    df = df.dropna()

    # Ordena por tempo
    df = df.sort_values(by="Tempo").reset_index(drop=True)

    # Tempo relativo
    df["Tempo_s"] = df["Tempo"] - df["Tempo"].iloc[0]

    # Intervalo entre amostras
    df["Delta_t"] = df["Tempo_s"].diff().fillna(0)

    return df


# ============================================================
# 4. EXTRAÇÃO DE CARACTERÍSTICAS POR JANELA
# ============================================================

def extrair_caracteristicas(df, classe, nome_arquivo, tamanho_janela=50):
    """
    Divide o arquivo em janelas e extrai características.
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

        # Características estatísticas dos sinais
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

            if len(df_features) > 0:
                todos_dados.append(df_features)

    if len(todos_dados) == 0:
        raise ValueError(f"Nenhum dado válido foi encontrado em {pasta_base}")

    dataset = pd.concat(todos_dados, ignore_index=True)
    return dataset


# ============================================================
# 6. AVALIAÇÃO POR ARQUIVO
# ============================================================

def avaliar_por_arquivo(df_teste, y_pred):
    df_resultado = df_teste.copy()
    df_resultado["Predicao"] = y_pred

    print("\n==============================")
    print("AVALIAÇÃO POR ARQUIVO")
    print("==============================")

    resultados = []
    acertos = 0
    total = 0

    for arquivo in df_resultado["Arquivo"].unique():
        dados_arquivo = df_resultado[df_resultado["Arquivo"] == arquivo]

        classe_real = dados_arquivo["Classe"].iloc[0]
        predicoes = dados_arquivo["Predicao"].tolist()

        contagem = Counter(predicoes)
        classe_final = contagem.most_common(1)[0][0]

        acertou = classe_real == classe_final

        print(f"\nArquivo: {arquivo}")
        print(f"Classe real: {classe_real}")
        print(f"Predições por janela: {dict(contagem)}")
        print(f"Classe final prevista: {classe_final}")
        print("Resultado:", "ACERTOU" if acertou else "ERROU")

        resultados.append({
            "Arquivo": arquivo,
            "Classe_real": classe_real,
            "Classe_prevista": classe_final,
            "Acertou": acertou
        })

        if acertou:
            acertos += 1

        total += 1

    if total > 0:
        print(f"\nAcurácia por arquivo: {acertos}/{total} = {acertos / total:.4f}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(BASE_DIVIDIDA / "resultado_por_arquivo.csv", index=False)


# ============================================================
# 7. GERAR FRONTEIRA DE DECISÃO 2D
# ============================================================

def gerar_fronteira_decisao_knn(df_treino, df_teste, n_vizinhos=3):
    """
    Gera um gráfico tipo fronteira de decisão usando apenas 2 features:
    X = Corrente_A_rms
    Y = Celula2_Kgf_desvio
    """

    feature_x = "Corrente_A_rms"
    feature_y = "Celula2_Kgf_desvio"

    X_train_2d = df_treino[[feature_x, feature_y]].copy()
    y_train = df_treino["Classe"].copy()

    X_test_2d = df_teste[[feature_x, feature_y]].copy()
    y_test = df_teste["Classe"].copy()

    # Codificação numérica das classes
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_test_num = le.transform(y_test)

    # Normalização apenas para o modelo 2D da figura
    scaler_2d = StandardScaler()
    X_train_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_scaled = scaler_2d.transform(X_test_2d)

    # Modelo 2D só para visualização
    knn_2d = KNeighborsClassifier(
        n_neighbors=n_vizinhos,
        metric="minkowski",
        p=2
    )
    knn_2d.fit(X_train_scaled, y_train_num)

    acc_test = knn_2d.score(X_test_scaled, y_test_num)

    # Malha em coordenadas ORIGINAIS
    x_min = X_train_2d[feature_x].min() - 0.5
    x_max = X_train_2d[feature_x].max() + 0.5
    y_min = X_train_2d[feature_y].min() - 0.5
    y_max = X_train_2d[feature_y].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.03),
        np.arange(y_min, y_max, 0.03)
    )

    grid_original = pd.DataFrame(
    np.c_[xx.ravel(), yy.ravel()],
    columns=[feature_x, feature_y]
    )

    grid_scaled = scaler_2d.transform(grid_original)

    Z = knn_2d.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    # Cores de fundo
    cores_fundo = ListedColormap([
        "#cfe2f3",  # azul claro
        "#f4cccc",  # vermelho claro
        "#d9ead3",  # verde claro
        "#ead1dc"   # roxo claro
    ])

    # Cores dos pontos
    cores_pontos = {
        "normal": "#1f77b4",
        "helice_desbalanceada": "#d62728",
        "helice_quebrada": "#2ca02c",
        "rotacao_invertida": "#9467bd"
    }

    marcadores = {
        "normal": "o",
        "helice_desbalanceada": "s",
        "helice_quebrada": "^",
        "rotacao_invertida": "D"
    }

    nomes_legenda = {
        "normal": "Normal",
        "helice_desbalanceada": "Desbalanceada",
        "helice_quebrada": "Hélice quebrada",
        "rotacao_invertida": "Rotação invertida"
    }

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap=cores_fundo)

    # Plota apenas os pontos de treino
    for classe in le.classes_:
        dados = df_treino[df_treino["Classe"] == classe]

        plt.scatter(
            dados[feature_x],
            dados[feature_y],
            c=cores_pontos.get(classe, "#000000"),
            marker=marcadores.get(classe, "o"),
            s=25,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            label=nomes_legenda.get(classe, classe)
        )

    plt.title(f"k={n_vizinhos} | teste={acc_test:.2f}")
    plt.xlabel("Corrente RMS (A)")
    plt.ylabel("Desvio da Célula 2 (kgf)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    caminho_saida = BASE_DIVIDIDA / "fronteira_decisao_knn.png"
    plt.savefig(caminho_saida, dpi=300)
    plt.show()

    print(f"\nFigura salva em: {caminho_saida}")


# ============================================================
# 8. TREINAR O MODELO PRINCIPAL
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

    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo principal
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

    acc = accuracy_score(y_test, y_pred)
    print("\nAcurácia:")
    print(acc)

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    avaliar_por_arquivo(df_teste, y_pred)

    # Gera a figura semelhante à que você pediu
    gerar_fronteira_decisao_knn(df_treino, df_teste, n_vizinhos=N_VIZINHOS)

    # Salvar modelo e objetos auxiliares
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
# 9. TESTAR UM ARQUIVO NOVO
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

    if len(df_features) == 0:
        print("Nenhuma janela válida foi gerada para esse arquivo.")
        return None

    X_novo = df_features.drop(["Classe", "Arquivo"], axis=1)
    X_novo = X_novo[colunas_modelo]

    X_novo_scaled = scaler.transform(X_novo)

    predicoes = modelo_knn.predict(X_novo_scaled)

    contagem = Counter(predicoes)
    classe_final = contagem.most_common(1)[0][0]

    print("\n==============================")
    print("TESTE DE ARQUIVO NOVO")
    print("==============================")
    print(f"\nArquivo: {caminho_arquivo.name}")
    print(f"Predições por janela: {dict(contagem)}")
    print(f"Classe final prevista: {classe_final}")

    return classe_final


# ============================================================
# 10. EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    separar_arquivos_treino_teste()
    modelo, scaler = treinar_modelo()

    # Exemplo de teste manual de um arquivo:
    # testar_arquivo_novo(
    #     r"C:\Users\Vinicius\Desktop\dataset_dividido\teste\normal\terceiroteste_helicenormal.xlsx"
    # )
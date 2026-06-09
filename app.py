import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

#Configuração inicial do dashboard

st.set_page_config(
    page_title="Dashboard TRIA",
    layout="wide"
)

st.title("Dashboard TRIA")

#Carregando os dados

df = pd.read_csv("data/dados_tria.csv")

# Define os títulos das abas
aba1, aba2 = st.tabs(["Análise de dados", "Classificação Probabilística"])

with aba1:
    #Filtros

    st.sidebar.header("Filtros")

    regioes = st.sidebar.multiselect(
        "Selecione a(s) Região(ões)",
        options=sorted(df["Região"].unique()),
        default=sorted(df["Região"].unique())
    )
    ufs_disponiveis = sorted(
        df[df["Região"].isin(regioes)]["UF"].unique()
    )

    ufs = st.sidebar.multiselect(
        "Selecione a(s) UF(s)",
        options=ufs_disponiveis,
        default=ufs_disponiveis
    )

    municipios_disponiveis = sorted(
        df[
            (df["Região"].isin(regioes))
            &
            (df["UF"].isin(ufs))
        ]["Município"].unique()
    )

    municipios = st.sidebar.multiselect(
        "Selecione o(s) Município(s)",
        options=municipios_disponiveis,
        default=municipios_disponiveis
    )

    df_filtrado = df[
        (df["Região"].isin(regioes))
        &
        (df["UF"].isin(ufs))
        &
        (df["Município"].isin(municipios))
    ]

    # KPIs

    st.subheader("Resumo Geral")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Municípios",
        df_filtrado["Município"].nunique()
    )

    col2.metric(
        "População Total",
        f"{df_filtrado['População'].sum():,.0f}"
    )

    col3.metric(
        "TRIA Aplicadas",
        f"{df_filtrado['Domicílios com a TRIA aplicada'].sum():,.0f}"
    )

    col4.metric(
        "Pessoas em Risco",
        f"{df_filtrado['Pessoas em domicílios em risco de insegurança alimentar'].sum():,.0f}"
    )



    st.divider()

    col1, col2 = st.columns(2)

    #Gráfico 1

    with col1:

        top10 = (
            df_filtrado
            .sort_values(
                "% Domicílios em risco de insegurança alimentar",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top10,
            x="% Domicílios em risco de insegurança alimentar",
            y="Município",
            orientation="h",
            text="% Domicílios em risco de insegurança alimentar",
            title="Top 10 Municípios com Maior Insegurança Alimentar"
        )

        st.plotly_chart(fig, use_container_width=True)

    #Gráfico 2

    with col2:

        uf_risco = (
            df_filtrado
            .groupby("UF", as_index=False)
            ["% Domicílios em risco de insegurança alimentar"]
            .mean()
            .sort_values(
                "% Domicílios em risco de insegurança alimentar",
                ascending=False
            )
        )

        fig = px.bar(
            uf_risco,
            x="UF",
            y="% Domicílios em risco de insegurança alimentar",
            text="% Domicílios em risco de insegurança alimentar",
            title="Percentual Médio de Insegurança Alimentar por UF"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    #Gráfico 3

    with col3:

        raca = pd.DataFrame({

            "Grupo": [
                "Pardo",
                "Branco",
                "Preto",
                "Indígena",
                "Amarelo"
            ],

            "Quantidade": [

                df_filtrado[
                    "Domicílios em risco de insegurança alimentar com RF respondente autodeclarado pardo"
                ].sum(),

                df_filtrado[
                    "Domicílios em risco de insegurança alimentar com RF respondente autodeclarado branco"
                ].sum(),

                df_filtrado[
                    "Domicílios em risco de insegurança alimentar com RF respondente autodeclarado preto"
                ].sum(),

                df_filtrado[
                    "Domicílios em risco de insegurança alimentar com RF respondente autodeclarado indígena"
                ].sum(),

                df_filtrado[
                    "Domicílios em risco de insegurança alimentar com RF respondente autodeclarado amarelo"
                ].sum()
            ]
        })

        fig = px.bar(
            raca,
            x="Grupo",
            y="Quantidade",
            text="Quantidade",
            title="Perfil Racial dos Domicílios em Risco"
        )

        st.plotly_chart(fig, use_container_width=True)

    #Gráfico 4

    with col4:

        top_pessoas = (
            df_filtrado
            .sort_values(
                "Pessoas em domicílios em risco de insegurança alimentar",
                ascending=False
            )
            .head(10)
        )

        fig = px.bar(
            top_pessoas,
            x="Município",
            y="Pessoas em domicílios em risco de insegurança alimentar",
            text="Pessoas em domicílios em risco de insegurança alimentar",
            title="Municípios com Mais Pessoas em Risco"
        )

        st.plotly_chart(fig, use_container_width=True)

    #Gráfico 5

    regiao = (
        df_filtrado
        .groupby("Região", as_index=False)
        .agg({
            "Domicílios com a TRIA aplicada":"sum",
            "Domicílios em risco de insegurança alimentar":"sum"
        })
    )

    regiao_melt = regiao.melt(
        id_vars="Região",
        var_name="Indicador",
        value_name="Quantidade"
    )

    fig = px.bar(
        regiao_melt,
        x="Região",
        y="Quantidade",
        color="Indicador",
        barmode="group",
        title="Cobertura da TRIA e Domicílios em Risco por Região"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabela 

    st.subheader("Dados filtrados")
    st.dataframe(df_filtrado)


with aba2:

    st.header("Classificação com Teorema de Bayes")

    # ==================================================
    # CRIAR VARIÁVEL ALVO
    # ==================================================

    mediana_risco = df[
        "% Domicílios em risco de insegurança alimentar"
    ].median()

    df["Alto_Risco"] = (
        df["% Domicílios em risco de insegurança alimentar"]
        >= mediana_risco
    ).astype(int)

    # ==================================================
    # CRIAR FAIXA POPULACIONAL
    # ==================================================

    df["Faixa_Populacao"] = pd.qcut(
        df["População"],
        q=3,
        labels=[
            "Baixa",
            "Média",
            "Alta"
        ]
    )

    # ==================================================
    # FUNÇÃO BAYES
    # ==================================================

    def calcular_bayes(df, regiao, faixa_pop):

        # P(A)
        p_alto_risco = (
            df["Alto_Risco"]
            .mean()
        )

        # P(Região | Alto Risco)
        p_regiao_dado_alto = (
            df[
                df["Alto_Risco"] == 1
            ]["Região"]
            .eq(regiao)
            .mean()
        )

        # P(Faixa | Alto Risco)
        p_faixa_dado_alto = (
            df[
                df["Alto_Risco"] == 1
            ]["Faixa_Populacao"]
            .eq(faixa_pop)
            .mean()
        )

        # P(Região)
        p_regiao = (
            df["Região"]
            .eq(regiao)
            .mean()
        )

        # P(Faixa)
        p_faixa = (
            df["Faixa_Populacao"]
            .eq(faixa_pop)
            .mean()
        )

        # P(B)
        p_evidencia = (
            p_regiao *
            p_faixa
        )

        if p_evidencia == 0:

            posterior = 0

        else:

            posterior = (
                p_regiao_dado_alto *
                p_faixa_dado_alto *
                p_alto_risco
            ) / p_evidencia

        return {
            "p_alto_risco": p_alto_risco,
            "p_regiao_dado_alto": p_regiao_dado_alto,
            "p_faixa_dado_alto": p_faixa_dado_alto,
            "posterior": posterior
        }

    # ==================================================
    # FILTROS
    # ==================================================

    st.subheader("Selecione as características")

    col1, col2 = st.columns(2)

    with col1:

        regiao = st.selectbox(
            "Região",
            sorted(
                df["Região"].unique()
            )
        )

    with col2:

        faixa_pop = st.selectbox(
            "Faixa Populacional",
            [
                "Baixa",
                "Média",
                "Alta"
            ]
        )

    # ==================================================
    # CALCULAR BAYES
    # ==================================================

    resultado = calcular_bayes(
        df,
        regiao,
        faixa_pop
    )

    # ==================================================
    # EXIBIR COMPONENTES
    # ==================================================

    st.subheader("Componentes do Teorema")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "P(Alto Risco)",
        f"{resultado['p_alto_risco']:.2%}"
    )

    c2.metric(
        f"P({regiao} | Alto Risco)",
        f"{resultado['p_regiao_dado_alto']:.2%}"
    )

    c3.metric(
        f"P({faixa_pop} | Alto Risco)",
        f"{resultado['p_faixa_dado_alto']:.2%}"
    )

    st.divider()

    # ==================================================
    # RESULTADO FINAL
    # ==================================================

    probabilidade_final = resultado["posterior"]

    st.subheader("Resultado Final")

    st.metric(
        "Probabilidade de Alto Risco",
        f"{probabilidade_final:.2%}"
    )

    if probabilidade_final >= 0.70:

        st.error(
            "🔴 Alta probabilidade de insegurança alimentar."
        )

    elif probabilidade_final >= 0.40:

        st.warning(
            "🟡 Probabilidade moderada de insegurança alimentar."
        )

    else:

        st.success(
            "🟢 Baixa probabilidade de insegurança alimentar."
        )

    # ==================================================
    # EXPLICAÇÃO
    # ==================================================

    st.markdown(
        f"""
        ### 📖 Interpretação Bayesiana

        Considerando um município da região *{regiao}*
        e com faixa populacional *{faixa_pop}*,
        o Teorema de Bayes estimou uma probabilidade
        de *{probabilidade_final:.2%}*
        de apresentar alto risco de insegurança alimentar.
        
        ---
        
        *🔍 Como interpretar:*
        - *P(Alto Risco) = {resultado['p_alto_risco']:.2%}* é a probabilidade a priori (antes de saber região e faixa)
        - *P({regiao} | Alto Risco) = {resultado['p_regiao_dado_alto']:.2%}* é a chance de ser dessa região, dado que tem alto risco
        - *P({faixa_pop} | Alto Risco) = {resultado['p_faixa_dado_alto']:.2%}* é a chance dessa faixa, dado que tem alto risco
        
        O resultado final combina essas informações usando o Teorema de Bayes.
        """
    )

    st.divider()
    
    # ==================================================
    # CRIAÇÃO DA VARIÁVEL ALVO (TEM QUE VIR ANTES)
    # ==================================================
    
    mediana_risco = df["% Domicílios em risco de insegurança alimentar"].median()
    df["Alto_Risco"] = (df["% Domicílios em risco de insegurança alimentar"] >= mediana_risco).astype(int)
    
    # Cria faixa populacional também (se já não tiver criado)
    if "Faixa_Populacao" not in df.columns:
        df["Faixa_Populacao"] = pd.qcut(
            df["População"],
            q=3,
            labels=["Baixa", "Média", "Alta"]
        )
    
    # ==================================================
    # PREPARAÇÃO DOS DADOS PARA OS MODELOS
    # ==================================================
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import plotly.express as px
    import numpy as np
    
    st.subheader("Preparando os modelos")
    
    # Seleciona as features (características) que vamos usar
    features_ml = [
        'População',
        'Domicílios com a TRIA aplicada',
        'Domicílios em risco de insegurança alimentar com pessoas menores de 18 anos',
        'Domicílios em risco de insegurança alimentar com pessoas com deficiência ',
        'Domicílios em risco de insegurança alimentar com RF respondente do sexo feminino',
        'Domicílios em risco de insegurança alimentar com pessoas de povo ou comunidade tradicional'
    ]
    
    # Verifica se todas as features existem
    for col in features_ml:
        if col not in df.columns:
            st.warning(f"Coluna não encontrada: '{col}'. Verifique se o nome está exato.")
    
    # Remove linhas com valores vazios nessas colunas E na Região
    df_ml = df[features_ml + ['Região', 'Alto_Risco']].dropna()
    
    if len(df_ml) < 10:
        st.error(f"Poucos dados disponíveis: {len(df_ml)} linhas. Verifique se as colunas estão corretas.")
        st.stop()
    
    # Codifica a coluna Região (texto vira número)
    le_regiao = LabelEncoder()
    df_ml['Região_cod'] = le_regiao.fit_transform(df_ml['Região'])
    
    # Adiciona a região codificada nas features
    features_completas = features_ml + ['Região_cod']
    
    # Separa X (features) e y (alvo)
    X = df_ml[features_completas]
    y = df_ml['Alto_Risco']
    
    # Divide em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Padroniza os dados (importante pra regressão logística)
    # Só padroniza as colunas numéricas originais, a Região_cod já está ok
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features_ml])
    X_test_scaled = scaler.transform(X_test[features_ml])
    
    # Junta a Região_cod de volta
    X_train_final = np.column_stack([X_train_scaled, X_train['Região_cod'].values])
    X_test_final = np.column_stack([X_test_scaled, X_test['Região_cod'].values])
    
    # Lista de nomes das features após padronização (pra exibir)
    features_finais = features_ml + ['Região_cod']
    
    st.success(f"Dados preparados: {len(X_train)} treino, {len(X_test)} teste")
    
    # ==================================================
    # MODELO 1: REGRESSÃO LOGÍSTICA
    # ==================================================
    
    st.subheader("📊 1. Regressão Logística")
    
    with st.spinner("Treinando Regressão Logística..."):
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        log_reg.fit(X_train_final, y_train)
        
        # Predições
        y_pred_log = log_reg.predict(X_test_final)
        y_proba_log = log_reg.predict_proba(X_test_final)[:, 1]
        
        # Métricas
        acc_log = accuracy_score(y_test, y_pred_log)
        prec_log = precision_score(y_test, y_pred_log)
        rec_log = recall_score(y_test, y_pred_log)
        f1_log = f1_score(y_test, y_pred_log)
    
    # Exibe métricas
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Acurácia", f"{acc_log:.2%}")
    col_met2.metric("Precisão", f"{prec_log:.2%}")
    col_met3.metric("Recall", f"{rec_log:.2%}")
    col_met4.metric("F1-Score", f"{f1_log:.2%}")
    
    # Importância dos coeficientes (Regressão Logística)
    coeficientes = pd.DataFrame({
        'Feature': features_finais,
        'Coeficiente': log_reg.coef_[0]
    })
    coeficientes = coeficientes.sort_values('Coeficiente', key=abs, ascending=False)
    
    with st.expander("Ver fatores que influenciam"):
        st.caption("Positivo = aumenta risco, Negativo = diminui risco")
        for _, row in coeficientes.head(5).iterrows():
            nome_amigavel = row['Feature'].replace('_', ' ').title().replace('Cod', '')
            impacto = "🔴 aumenta" if row['Coeficiente'] > 0 else "🟢 diminui"
            st.write(f"- *{nome_amigavel}*: {impacto} o risco")
    
    # ==================================================
    # MODELO 2: RANDOM FOREST
    # ==================================================
    
    st.subheader("🌲 2. Random Forest")
    
    with st.spinner("Treinando Random Forest..."):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train_final, y_train)  # Random Forest não precisa de scaling
        
        # Predições
        y_pred_rf = rf.predict(X_test_final)
        y_proba_rf = rf.predict_proba(X_test_final)[:, 1]
        
        # Métricas
        acc_rf = accuracy_score(y_test, y_pred_rf)
        prec_rf = precision_score(y_test, y_pred_rf)
        rec_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
    
    # Exibe métricas
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    col_met1.metric("Acurácia", f"{acc_rf:.2%}")
    col_met2.metric("Precisão", f"{prec_rf:.2%}")
    col_met3.metric("Recall", f"{rec_rf:.2%}")
    col_met4.metric("F1-Score", f"{f1_rf:.2%}")
    
    # Importância das features (Random Forest)
    importancia = pd.DataFrame({
        'Feature': features_finais,
        'Importância': rf.feature_importances_
    }).sort_values('Importância', ascending=False)
    
    st.subheader("📈 Quais fatores mais influenciam?")
    
    # Limpa o nome pra exibir
    importancia['Feature_Limpa'] = importancia['Feature'].str.replace('_', ' ').str.title().str.replace('Cod', '')
    
    fig_importancia = px.bar(
        importancia.head(8),
        x='Importância',
        y='Feature_Limpa',
        orientation='h',
        title="Importância das características (Random Forest)",
        labels={'Feature_Limpa': 'Característica', 'Importância': 'Importância relativa'},
        color='Importância',
        color_continuous_scale='Reds'
    )
    fig_importancia.update_layout(height=400)
    st.plotly_chart(fig_importancia, use_container_width=True)
    
    # ==================================================
    # COMPARAÇÃO DOS MODELOS
    # ==================================================
    
    st.subheader("🏆 Comparação entre Modelos")
    
    comparacao = pd.DataFrame({
        'Modelo': ['Regressão Logística', 'Random Forest'],
        'Acurácia': [acc_log, acc_rf],
        'Precisão': [prec_log, prec_rf],
        'Recall': [rec_log, rec_rf],
        'F1-Score': [f1_log, f1_rf]
    })
    
    fig_comp = px.bar(
        comparacao.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
        x='Métrica',
        y='Valor',
        color='Modelo',
        barmode='group',
        title='Comparação de Desempenho',
        text_auto='.2%'
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Escolhe o melhor modelo baseado no F1-Score
    if f1_rf > f1_log:
        melhor_modelo = "Random Forest"
        melhor_f1 = f1_rf
    else:
        melhor_modelo = "Regressão Logística"
        melhor_f1 = f1_log
    
    st.success(f"✅ *Melhor modelo: {melhor_modelo}* (F1-Score: {melhor_f1:.2%})")
    
    # ==================================================
    # PREDIÇÃO INTERATIVA
    # ==================================================
    
    st.subheader("🎯 Faça sua própria predição")
    st.markdown("Preencha os dados de um município para prever o risco:")
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        populacao_pred = st.number_input("População", min_value=1000, value=50000, step=10000, key="pop_pred")
        menores_pred = st.number_input("Domicílios com menores de 18 anos em risco", min_value=0, value=100, key="menores_pred")
        pcd_pred = st.number_input("Domicílios com PcD em risco", min_value=0, value=50, key="pcd_pred")
    
    with col_pred2:
        feminino_pred = st.number_input("Domicílios com RF feminino em risco", min_value=0, value=200, key="fem_pred")
        comunidade_pred = st.number_input("Domicílios com comunidade tradicional em risco", min_value=0, value=20, key="comunidade_pred")
        tria_pred = st.number_input("Domicílios com TRIA aplicada", min_value=0, value=1000, key="tria_pred")
        
        regiao_pred = st.selectbox(
            "Região",
            sorted(df['Região'].unique()),
            key="reg_pred"
        )
    
    if st.button("🔮 Prever Risco", key="btn_pred"):
        try:
            # Prepara input
            regiao_cod = le_regiao.transform([regiao_pred])[0]
            
            # Cria DataFrame com os dados do usuário
            input_df = pd.DataFrame([{
                'População': populacao_pred,
                'Domicílios com a TRIA aplicada': tria_pred,
                'Domicílios em risco de insegurança alimentar com pessoas menores de 18 anos': menores_pred,
                'Domicílios em risco de insegurança alimentar com pessoas com deficiência ': pcd_pred,
                'Domicílios em risco de insegurança alimentar com RF respondente do sexo feminino': feminino_pred,
                'Domicílios em risco de insegurança alimentar com pessoas de povo ou comunidade tradicional': comunidade_pred,
            }])
            
            # Padroniza os dados numéricos
            input_scaled = scaler.transform(input_df[features_ml])
            
            # Junta com a região codificada
            input_final = np.column_stack([input_scaled, [regiao_cod]])
            
            # Predições
            prob_log = log_reg.predict_proba(input_final)[0][1]
            prob_rf = rf.predict_proba(input_final)[0][1]
            
            st.divider()
            st.subheader("📊 Resultado da Predição")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("*Regressão Logística*")
                if prob_log >= 0.5:
                    st.error(f"🔴 ALTO RISCO ({prob_log:.1%})")
                else:
                    st.success(f"🟢 BAIXO RISCO ({prob_log:.1%})")
            
            with col_res2:
                st.markdown("*Random Forest*")
                if prob_rf >= 0.5:
                    st.error(f"🔴 ALTO RISCO ({prob_rf:.1%})")
                else:
                    st.success(f"🟢 BAIXO RISCO ({prob_rf:.1%})")
            
            # Predição consenso
            prob_medio = (prob_log + prob_rf) / 2
            st.markdown("---")
            st.markdown("*💡 Consenso dos modelos:*")
            if prob_medio >= 0.7:
                st.error(f"⚠️ *ALERTA!* Probabilidade média de {prob_medio:.1%} de alto risco")
                st.markdown("📌 Recomendação: Priorizar políticas públicas para este perfil")
            elif prob_medio >= 0.4:
                st.warning(f"📌 *Atenção!* Probabilidade média de {prob_medio:.1%} de alto risco")
                st.markdown("📌 Recomendação: Monitoramento constante")
            else:
                st.success(f"✅ *Tranquilo!* Probabilidade média de {prob_medio:.1%} de alto risco")
                st.markdown("📌 Recomendação: Manter ações preventivas")
                
        except Exception as e:
            st.error(f"Erro na predição: {e}")
            st.info("Verifique se os valores estão dentro do esperado ou se há dados suficientes no treinamento.")
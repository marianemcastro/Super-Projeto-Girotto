# Super Projeto Girotto
Trabalho avaliativo do 2° bimestre de probabilidade e estatística.
## Alunos:
- José Vitor Matos Monteiro Seabra;
- Mariane Marinho de Castro;
- Lais Pereira Cunha
# Análise de Risco de Insegurança Alimentar no Brasil
Este projeto realiza uma análise exploratória e preditiva dos dados do DATASUS sobre **Triagem de Risco de Insegurança Alimentar (TRIA)** para todos os municípios brasileiros, com o objetivo de identificar fatores associados ao alto risco de insegurança alimentar e comparar diferentes abordagens de classificação.
## 📊 Dataset

- **Fonte:** DATASUS (datasus.saude.gov.br)
- **Arquivo:** `TABELA_BRASIL_bagunçada.csv`
- **Período:** nov/2023 a dez/2025 (competência única)
- **Instâncias:** 5.568 municípios (após limpeza)
- **Atributos:** 32.Incluindo:
  - Identificação: `Região`, `UF`, `IBGE`, `Município`, `Competência`
  - Demografia: `População`, `Domicílios com a TRIA aplicada`
  - Risco: `Domicílios em risco de insegurança alimentar`, `% Domicílios em risco`
  - Subgrupos vulneráveis: contagens e porcentagens de domicílios em risco que contêm:
    - Pessoas com deficiência
    - Pessoas em situação de rua
    - Pessoas de povos/comunidades tradicionais
    - Pessoas menores de 18 anos
    - Responsável familiar (RF) do sexo feminino/masculino
    - RF autodeclarado pardo, branco, amarelo, preto, indígena

> **Variável alvo (classificação):** 
## 🎯 Objetivos
1. **Explorar** a distribuição da insegurança alimentar por região e grupos vulneráveis.
2. **Aplicar o Teorema de Bayes** (Naive Bayes) como classificador probabilístico.
3. **Comparar** o desempenho do Naive Bayes com outros algoritmos de machine learning (Regressão Logística, Random Forest, XGBoost).
4. **Identificar** os principais fatores associados ao alto risco de insegurança alimentar.

## 🧪 Metodologia

### 🔧 Tratamento dos dados
- 

### 📈 Análise exploratória
- Matriz de correlação entre variáveis.
- Distribuição da raça/cor do responsável familiar nos municípios de alto risco.

## 📈 Resultados

### Principais insights
- As regiões **Norte** e **Nordeste** apresentam as maiores medianas de risco (~18% e ~15%).
- Presença de **menores de 18 anos** e **chefia feminina** são os fatores mais correlacionados com o alto risco.
- Nos municípios de alto risco, **62% dos responsáveis familiares são pardos**, 22% brancos, 11% pretos e 5% indígenas.

## 🚀 Como reproduzir

### Pré‑requisitos
- Python 3.10 ou superior
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

### Passos
1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/inseguranca-alimentar-brasil.git
   cd inseguranca-alimentar-brasil

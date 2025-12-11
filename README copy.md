# BioModel Pipeline - Computational Biology Project

Pipeline automatizado para modelagem e simulação de circuitos genéticos usando Redes de Petri Estocásticas, Algoritmo de Gillespie e Equações Diferenciais Ordinárias.

## Visão Geral

Este projeto implementa um workflow completo para:

1. **Buscar** modelos biológicos do repositório [BioModels](https://www.ebi.ac.uk/biomodels/) automaticamente
2. **Parsear** arquivos SBML (Systems Biology Markup Language) incluindo Assignment Rules e Function Definitions
3. **Gerar** visualizações de Redes de Petri (PNG, PNML, formato GreatSPN)
4. **Simular** dinâmicas do modelo via Gillespie SSA (estocástico) ou ODE (determinístico)

## Instalação

### Dependências do Sistema

- **Python 3.10+** (testado com Python 3.12)
- **Graphviz** (para gerar imagens de Petri Nets)

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

### Ambiente Python

```bash
# Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install python-libsbml numpy matplotlib scipy networkx graphviz requests
```

## Uso

### Execução do Pipeline

```bash
cd rascunho_base
source venv/bin/activate

# Sintaxe
python3 src/run_pipeline.py [duração] [num_runs] [método] "nome_modelo"

# Parâmetros:
#   duração   - Tempo total de simulação (unidades do modelo)
#   num_runs  - Número de execuções para análise estatística (Gillespie)
#   método    - "auto", "gillespie" ou "ode"
#   modelo    - Nome comum ou ID BioModels
```

### Exemplos

```bash
# Repressilator - Oscilador de 3 genes (Gillespie)
python3 src/run_pipeline.py 100 5 auto "repressilator"

# Toggle Switch - Memória biestável (ODE)
python3 src/run_pipeline.py 200 3 ode "toggle switch"

# Relógio Circadiano - Oscilações amortecidas (ODE)
python3 src/run_pipeline.py 200 1 ode "circadian"

# Feed-Forward Loop - Resposta adaptativa (Gillespie com tau-leaping)
python3 src/run_pipeline.py 100 3 gillespie "feed-forward loop"

# Operon Lac - Regulação gênica (ODE)
python3 src/run_pipeline.py 200 2 ode "lac operon"
```

### Saídas Geradas

Para cada modelo, o pipeline gera em `output/<nome_modelo>/`:

| Arquivo | Descrição |
|---------|-----------|
| `*_parsed.json` | Dados estruturados do modelo |
| `*_petri_net.png` | Visualização da rede de Petri |
| `*.pnml` | Petri Net Markup Language |
| `*.net` + `*.def` | Formato nativo GreatSPN |
| `*_simulation*.png` | Gráficos de simulação |

## Visualização com GreatSPN

O GreatSPN torna possível visualizar as redes de petri estocásticas com seu formato nativo .net/.def.

Os arquivos `*.net` e `*.def` gerados em `output/<modelo>/` são o formato nativo do GreatSPN e podem ser abertos na GUI do GreatSPN para visualização, edição e análise da Rede de Petri.

- Se o GreatSPN estiver instalado localmente, abra o arquivo `.net` (o `.def` deve permanecer no mesmo diretório). Exemplos de comando:

```bash
# se você tiver o link em `tools/greatSPN/GreatSPN` (aponta para /opt/greatspn/bin/GreatSPN)
./tools/greatSPN/GreatSPN output/<modelo>/<arquivo>.net

# ou usar o executável do sistema diretamente
/opt/greatspn/bin/GreatSPN output/<modelo>/<arquivo>.net
```

- Na GUI: `File -> Open` e selecione o `*.net`; o arquivo `*.def` é carregado automaticamente. Use os menus de Layout/Visualization e as ferramentas de análise do GreatSPN para explorar marcações, taxas e simulações internas.

- Exportação: o GreatSPN permite exportar para PNML ou salvar imagens; utilize `File -> Export` / `Save As` conforme necessário.

Se o GreatSPN não estiver instalado via repositório oficial, fornecemos um ZIP com binários prontos para diferentes sistemas operacionais:

Download direto dos binários (ZIP):

https://datacloud.di.unito.it/index.php/s/MnWgcYamrHdDXZk/download

Dentro do ZIP você encontrará executáveis/binários para múltiplas plataformas (Linux, macOS, Windows). A instalação depende do seu sistema
### Demonstração (GIF)

Uma demonstração rápida do GreatSPN carregando um `*.net` está disponível abaixo:

![GreatSPN demo](tex/Figures/GreatSPN_usage.gif)

## Seleção Automática do Método

O sistema seleciona automaticamente o método mais apropriado baseado nas quantidades iniciais das espécies:

- **ODE**: valores fracionários pequenos (< 0.1) sugerem concentrações molares
- **ODE**: valores muito grandes (> 10000) atingem o limite termodinâmico
- **Gillespie**: valores intermediários (1-10000) representam contagens discretas onde ruído é relevante
- **Tau-leaping**: ativado automaticamente quando a_total > 50000 (sistemas rápidos)

## Execução Manual dos Módulos

Cada módulo pode ser executado independentemente:

```bash
# 1. Buscar modelo do BioModels
python3 src/fetch_biomodel.py "repressilator"

# 2. Parsear SBML para JSON
python3 src/parse_sbml.py output/<pasta>/modelo.xml

# 3. Gerar Rede de Petri
python3 src/generate_petri_net.py output/<pasta>/*_parsed.json

# 4. Simular (método, duração, runs)
python3 src/simulate_circuit.py output/<pasta>/*_parsed.json gillespie 500 4
```

## Modificação de Parâmetros

O arquivo `*_parsed.json` é editável. Para testar hipóteses:

1. Abra o JSON e localize a seção `"parameters"`
2. Modifique o campo `"value"` do parâmetro desejado
3. Execute apenas o simulador para ver o efeito

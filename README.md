# Analise Estatistica - NBA 2024-25

Trabalho pratico de Probabilidade e Estatistica A, da Universidade Federal de Goias, com foco em Estatistica Descritiva aplicada a dados de desempenho de jogadores da NBA.

## Integrantes

| Nome | Matricula |
|---|---|
| Joao Vitor Fernandes Ferreira | 202504011 |
| Mateus Nery Bailao | 202504020 |
| Felipe Orlow de Oliveira Sene | 202503999 |

## Objetivo do repositorio

Este repositorio concentra:

- o tratamento do banco de dados;
- os calculos estatisticos pedidos no roteiro;
- a geracao dos graficos usados na analise;
- uma opcao complementar de visualizacao interativa com Streamlit e Plotly, em estilo BI.

O relatorio final da disciplina nao e gerado por codigo neste projeto. A redacao final deve ser feita manualmente pelo grupo, usando as tabelas, medidas e figuras produzidas aqui como apoio.

## Estrutura do projeto

```text
Trabalho PE/
|-- dados/
|   `-- nba_dailyleaders_full_24_25.csv
|-- codigo/
|   `-- nba_analise.py
|-- graficos/
|   `-- 01 a 17 - figuras em PNG
|-- Interface/
|   `-- dashboard.py
|-- entrega/
|   `-- pasta reservada para a versao final montada manualmente
`-- README.md
```

## Escopo da analise

O conjunto de dados utilizado e o `NBA Daily Leaders Full 24/25`, disponivel no Kaggle. A base usada no projeto cobre registros entre `2024-10-22` e `2025-06-22`, conforme disponibilizado no arquivo original.

Para a amostra analitica, os dados sao agregados por jogador e filtrados para manter apenas atletas com `GP >= 20`.

As analises contemplam o que o roteiro pede:

- variaveis quantitativas continuas;
- variavel quantitativa discreta (`GP`);
- variaveis qualitativas (`Tm`, `POS`, `POS_simple`);
- tabelas de frequencia;
- medidas de tendencia central e dispersao;
- medidas de assimetria e curtose;
- box plots e histogramas;
- comparacoes por grupo de posicao.

## Como rodar

### 1. Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scipy streamlit plotly
```

Se preferir, tambem e possivel instalar tudo de uma vez e depois executar apenas o painel.

### 2. Gerar as analises e os graficos

```bash
python codigo/nba_analise.py
```

Esse script:

- le o CSV da pasta `dados/`;
- prepara a base agregada por jogador;
- calcula as estatisticas descritivas;
- monta as tabelas de frequencia;
- salva os graficos na pasta `graficos/`.

### 3. Abrir a visualizacao interativa

Entre na pasta do projeto e execute:

```bash
streamlit run Interface/dashboard.py
```

O dashboard e opcional e serve apenas como apoio visual para explorar os resultados de forma interativa.

Ele foi organizado em estilo BI para fortalecer a apresentacao do trabalho, com:

- indicadores executivos;
- filtros por posicao, franquia e faixa de jogos;
- graficos interativos para comparacoes entre jogadores e grupos;
- rankings dinamicos e tabela exportavel.

## Acesso rapido ao dashboard

Para facilitar a leitura do professor, o caminho mais direto para abrir o painel e:

1. Abrir um terminal na pasta `Trabalho PE`.
2. Instalar as dependencias com:

```bash
pip install pandas numpy matplotlib seaborn scipy streamlit plotly
```

3. Executar o comando:

```bash
streamlit run Interface/dashboard.py
```

4. Aguardar a abertura automatica no navegador.

Observacao importante: para abrir o dashboard, nao e necessario executar antes o arquivo `codigo/nba_analise.py`. O painel le diretamente a base da pasta `dados/`.

Se o navegador nao abrir sozinho, basta copiar o endereco local mostrado no terminal, geralmente:

```text
http://localhost:8501
```

O painel foi pensado como uma visualizacao complementar em estilo Power BI, permitindo:

- filtrar jogadores por posicao, franquia e quantidade de jogos;
- comparar estatisticas entre perfis e grupos;
- visualizar rankings dinamicos;
- explorar a base de forma interativa sem alterar o relatorio escrito.

## Variaveis trabalhadas

| Variavel | Tipo | Descricao |
|---|---|---|
| `PTS_mean` | Quantitativa continua | Media de pontos por jogo |
| `REB_mean` | Quantitativa continua | Media de rebotes por jogo |
| `AST_mean` | Quantitativa continua | Media de assistencias por jogo |
| `MP_mean` | Quantitativa continua | Media de minutos por jogo |
| `FG_pct_mean` | Quantitativa continua | Percentual medio de arremessos de campo |
| `P3_pct_mean` | Quantitativa continua | Percentual medio de arremessos de 3 pontos |
| `STL_mean` | Quantitativa continua | Media de roubos de bola por jogo |
| `BLK_mean` | Quantitativa continua | Media de tocos por jogo |
| `TOV_mean` | Quantitativa continua | Media de turnovers por jogo |
| `PF_mean` | Quantitativa continua | Media de faltas por jogo |
| `GP` | Quantitativa discreta | Numero de jogos disputados |
| `Tm` | Qualitativa nominal | Franquia do jogador |
| `POS` | Qualitativa nominal | Posicao detalhada |
| `POS_simple` | Qualitativa nominal | Posicao simplificada |

## Saidas principais

- `graficos/01_hist_pts.png`
- `graficos/02_hist_mp.png`
- `graficos/03_box_pts.png`
- `graficos/04_box_reb.png`
- `graficos/05_box_ast.png`
- `graficos/06_bar_posicao.png`
- `graficos/07_bar_times.png`
- `graficos/08_bar_agrupado.png`
- `graficos/09_hist_reb.png`
- `graficos/10_hist_ast.png`
- `graficos/11_hist_fg.png`
- `graficos/12_hist_3p.png`
- `graficos/13_hist_gp.png`
- `graficos/14_box_gp.png`
- `graficos/15_box_continuas.png`
- `graficos/16_bar_pos5.png`
- `graficos/17_bar_todos_times.png`

## Observacao final

O projeto foi organizado para sustentar a parte tecnica da analise. A escrita, selecao final de comentarios e formatacao academica do relatorio devem ser feitas manualmente pelo grupo.

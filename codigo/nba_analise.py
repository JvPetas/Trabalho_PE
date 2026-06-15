# -*- coding: utf-8 -*-
# =============================================================================
# UNIVERSIDADE FEDERAL DE GOIÁS — INSTITUTO DE MATEMÁTICA E ESTATÍSTICA
# Disciplina: Probabilidade e Estatística A — 1º Semestre/2026
# Professor: Márcio Augusto F. Rodrigues
#
# TRABALHO PRÁTICO — ESTATÍSTICA DESCRITIVA
# Tema: Análise Descritiva de Desempenho de Jogadores da NBA — Temporada 2024-25
# Fonte: Kaggle — NBA Daily Leaders Full 24/25
#        https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425
#
# Integrantes: [NOME 1], [NOME 2], [NOME 3]
# Entrega: 24/06/2026
# Ferramenta: Python 3.13 — pandas, matplotlib, seaborn, scipy
# =============================================================================

# =============================================================================
# SEÇÃO 1 — IMPORTAÇÕES
# =============================================================================
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SEÇÃO 2 — CARREGAMENTO E LIMPEZA
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "dados" / "nba_dailyleaders_full_24_25.csv"
GRAFICOS_DIR = BASE_DIR / "graficos"
GRAFICOS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("ANÁLISE DESCRITIVA — NBA 2024-25")
print("=" * 70)

# --- 2.1 Carregamento ---
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print(f"\nDataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
print(f"Jogadores únicos no dataset: {df['Player'].nunique()}")
print(f"Período coberto na base: {df['Date'].min().date()} a {df['Date'].max().date()}")

# --- 2.2 Renomear e preencher Home_Away ---
df = df.rename(columns={"Unnamed: 3": "Home_Away"})
df["Home_Away"] = df["Home_Away"].fillna("Home")
df["Home_Away"] = df["Home_Away"].replace("@", "Away")

# --- 2.3 Converter MP de MM:SS para minutos decimais ---
def mp_to_decimal(val):
    try:
        s = str(val).strip()
        if ":" in s:
            m, sec = s.split(":")
            return int(m) + int(sec) / 60
        return float(s)
    except Exception:
        return np.nan

df["MP_decimal"] = df["MP"].apply(mp_to_decimal)

for col in ["FG%", "3P%", "FT%", "PTS", "TRB", "AST", "STL", "BLK", "TOV", "PF"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- 2.4 Agregação por jogador ---
gp_series = df.groupby("Player").size().rename("GP")

def safe_mode(s):
    m = s.mode()
    return m.iloc[0] if len(m) > 0 else s.iloc[0]

agg = df.groupby("Player").agg({
    "PTS":        "mean",
    "TRB":        "mean",
    "AST":        "mean",
    "STL":        "mean",
    "BLK":        "mean",
    "TOV":        "mean",
    "MP_decimal": "mean",
    "FG%":        "mean",
    "3P%":        "mean",
    "PF":         "mean",
    "Tm":         safe_mode,
}).reset_index()

agg.columns = ["Player", "PTS_mean", "REB_mean", "AST_mean", "STL_mean",
               "BLK_mean", "TOV_mean", "MP_mean", "FG_pct_mean", "P3_pct_mean",
               "PF_mean", "Tm"]

agg = agg.merge(gp_series.reset_index(), on="Player")

# --- 2.5 Filtrar GP >= 20 ---
df_players = agg[agg["GP"] >= 20].copy().reset_index(drop=True)
print(f"Jogadores com GP ≥ 20: {len(df_players)}")

# --- 2.6 Dicionário completo de posições — 461 jogadores (temporada 2024-25) ---
POS_DICT = {
    # A
    "A.J. Green":                  "G",
    "A.J. Lawson":                 "G-F",
    "AJ Johnson":                  "F",
    "Aaron Gordon":                "F",
    "Aaron Holiday":               "G",
    "Aaron Nesmith":               "G-F",
    "Aaron Wiggins":               "G-F",
    "Adam Flagler":                "G",
    "Adem Bona":                   "C",
    "Ajay Mitchell":               "G",
    "Al Horford":                  "F-C",
    "Alec Burks":                  "G",
    "Alex Caruso":                 "G",
    "Alex Ducas":                  "G-F",
    "Alex Len":                    "C",
    "Alex Sarr":                   "F-C",
    "Alperen Şengün":              "C",
    "Amen Thompson":               "G-F",
    "Amir Coffey":                 "G-F",
    "Andre Drummond":              "C",
    "Andre Jackson Jr.":           "G-F",
    "Andrew Nembhard":             "G",
    "Andrew Wiggins":              "F",
    "Anfernee Simons":             "G",
    "Anthony Black":               "G",
    "Anthony Davis":               "C",
    "Anthony Edwards":             "G",
    "Anthony Gill":                "F",
    "Antonio Reeves":              "G",
    "Ariel Hukporti":              "C",
    "Ausar Thompson":              "F",
    "Austin Reaves":               "G",
    "Ayo Dosunmu":                 "G",
    # B
    "Bam Adebayo":                 "C",
    "Baylor Scheierman":           "G-F",
    "Ben Sheppard":                "G-F",
    "Ben Simmons":                 "G-F",
    "Bennedict Mathurin":          "G-F",
    "Bilal Coulibaly":             "G-F",
    "Bismack Biyombo":             "C",
    "Blake Wesley":                "G",
    "Bobby Portis":                "F-C",
    "Bogdan Bogdanović":           "G",
    "Bol Bol":                     "F-C",
    "Bones Hyland":                "G",
    "Bradley Beal":                "G",
    "Branden Carlson":             "C",
    "Brandin Podziemski":          "G",
    "Brandon Boston Jr.":          "G-F",
    "Brandon Clarke":              "F",
    "Brandon Miller":              "F",
    "Brandon Williams":            "G",
    "Brice Sensabaugh":            "G-F",
    "Bronny James":                "G",
    "Brook Lopez":                 "C",
    "Bruce Brown":                 "G-F",
    "Bub Carrington":              "G",
    "Buddy Hield":                 "G",
    # C
    "CJ McCollum":                 "G",
    "Cade Cunningham":             "G",
    "Caleb Houstan":               "G-F",
    "Caleb Martin":                "F",
    "Cam Reddish":                 "F",
    "Cam Spencer":                 "G",
    "Cam Thomas":                  "G",
    "Cam Whitmore":                "F",
    "Cameron Johnson":             "F",
    "Cameron Payne":               "G",
    "Caris LeVert":                "G-F",
    "Cason Wallace":               "G",
    "Charles Bassey":              "C",
    "Chet Holmgren":               "F-C",
    "Chris Boucher":               "F-C",
    "Chris Livingston":            "G-F",
    "Chris Paul":                  "G",
    "Christian Braun":             "G",
    "Christian Koloko":            "C",
    "Clint Capela":                "C",
    "Coby White":                  "G",
    "Cody Martin":                 "G-F",
    "Cody Williams":               "G",
    "Colby Jones":                 "G-F",
    "Cole Anthony":                "G",
    "Colin Castleton":             "C",
    "Collin Gillespie":            "G",
    "Collin Sexton":               "G",
    "Corey Kispert":               "G-F",
    "Cory Joseph":                 "G",
    "Craig Porter Jr.":            "G",
    # D
    "D'Angelo Russell":            "G",
    "DaQuan Jeffries":             "G-F",
    "Dalano Banton":               "G-F",
    "Dalen Terry":                 "G",
    "Dalton Knecht":               "G-F",
    "Damian Lillard":              "G",
    "Damion Lee":                  "G",
    "Daniel Gafford":              "C",
    "Daniel Theis":                "F-C",
    "Danté Exum":                  "G",
    "Dariq Whitehead":             "G-F",
    "Darius Garland":              "G",
    "David Roddy":                 "F",
    "Davion Mitchell":             "G",
    "Day'Ron Sharpe":              "C",
    "De'Aaron Fox":                "G",
    "De'Andre Hunter":             "F",
    "DeAndre Jordan":              "C",
    "DeMar DeRozan":               "G-F",
    "Dean Wade":                   "F",
    "Deandre Ayton":               "C",
    "Dejounte Murray":             "G",
    "Delon Wright":                "G",
    "Deni Avdija":                 "F",
    "Dennis Schröder":             "G",
    "Dereck Lively II":            "C",
    "Derrick Jones Jr.":           "F",
    "Derrick White":               "G",
    "Desmond Bane":                "G-F",
    "Devin Booker":                "G",
    "Devin Carter":                "G",
    "Devin Vassell":               "G",
    "Dillon Brooks":               "G-F",
    "Dillon Jones":                "F",
    "Domantas Sabonis":            "C",
    "Dominick Barlow":             "F",
    "Donovan Clingan":             "C",
    "Donovan Mitchell":            "G",
    "Donte DiVincenzo":            "G",
    "Dorian Finney-Smith":         "F",
    "Doug McDermott":              "F",
    "Draymond Green":              "F",
    "Drew Eubanks":                "C",
    "Drew Peterson":               "F",
    "Duncan Robinson":             "G-F",
    "Duop Reath":                  "C",
    "Dwight Powell":               "C",
    "Dyson Daniels":               "G",
    # E
    "Elfrid Payton":               "G",
    "Enrique Freeman":             "F-C",
    "Eric Gordon":                 "G",
    "Evan Mobley":                 "F-C",
    # F
    "Franz Wagner":                "G-F",
    "Fred VanVleet":               "G",
    # G
    "GG Jackson II":               "F",
    "Gabe Vincent":                "G",
    "Garrett Temple":              "G",
    "Garrison Mathews":            "G",
    "Gary Harris":                 "G",
    "Gary Payton II":              "G",
    "Gary Trent Jr.":              "G",
    "Georges Niang":               "F",
    "Giannis Antetokounmpo":       "F",
    "Goga Bitadze":                "C",
    "Gradey Dick":                 "G-F",
    "Grayson Allen":               "G",
    "Guerschon Yabusele":          "F",
    "Gui Santos":                  "F",
    # H
    "Harrison Barnes":             "F",
    "Haywood Highsmith":           "F",
    "Herbert Jones":               "G-F",
    "Hunter Tyson":                "F",
    # I
    "Immanuel Quickley":           "G",
    "Isaac Jones":                 "F",
    "Isaac Okoro":                 "G-F",
    "Isaiah Collier":              "G",
    "Isaiah Hartenstein":          "C",
    "Isaiah Joe":                  "G",
    "Isaiah Stewart":              "F-C",
    "Isaiah Wong":                 "G",
    "Ivica Zubac":                 "C",
    # J
    "JD Davison":                  "G",
    "JT Thor":                     "F",
    "Ja Morant":                   "G",
    "Ja'Kobe Walter":              "G",
    "Jabari Smith Jr.":            "F",
    "Jabari Walker":               "F",
    "Jaden Hardy":                 "G",
    "Jaden Ivey":                  "G",
    "Jaden McDaniels":             "F",
    "Jaden Springer":              "G",
    "Jae'Sean Tate":               "F",
    "Jaime Jaquez Jr.":            "G-F",
    "Jake LaRavia":                "F",
    "Jakob Poeltl":                "C",
    "Jalen Brunson":               "G",
    "Jalen Duren":                 "C",
    "Jalen Green":                 "G",
    "Jalen Johnson":               "F",
    "Jalen Pickett":               "G",
    "Jalen Smith":                 "F-C",
    "Jalen Suggs":                 "G",
    "Jalen Williams":              "G",
    "Jalen Wilson":                "F",
    "Jamal Cain":                  "F",
    "Jamal Murray":                "G",
    "Jamal Shead":                 "G",
    "James Harden":                "G",
    "Jamison Battle":              "F",
    "Jarace Walker":               "F-C",
    "Jared Butler":                "G",
    "Jared McCain":                "G",
    "Jaren Jackson Jr.":           "F-C",
    "Jarred Vanderbilt":           "F",
    "Jarrett Allen":               "C",
    "Javonte Green":               "G-F",
    "Jaxson Hayes":                "C",
    "Jay Huff":                    "C",
    "Jaylen Brown":                "F",
    "Jaylen Clark":                "G",
    "Jaylen Wells":                "G-F",
    "Jaylin Williams":             "F",
    "Jaylon Tyson":                "G-F",
    "Jayson Tatum":                "F",
    "Jeff Dowtin Jr.":             "G",
    "Jeff Green":                  "F",
    "Jerami Grant":                "F",
    "Jeremiah Robinson-Earl":      "F",
    "Jeremy Sochan":               "F",
    "Jericho Sims":                "C",
    "Jett Howard":                 "G-F",
    "Jevon Carter":                "G",
    "Jimmy Butler":                "F",
    "Jock Landale":                "C",
    "John Collins":                "F",
    "John Konchar":                "G-F",
    "Johnny Davis":                "G",
    "Johnny Furphy":               "G-F",
    "Johnny Juzang":               "G",
    "Jonas Valančiūnas":           "C",
    "Jonathan Isaac":              "F",
    "Jonathan Kuminga":            "F",
    "Jonathan Mogbo":              "F-C",
    "Jordan Clarkson":             "G",
    "Jordan Goodwin":              "G",
    "Jordan Hawkins":              "G",
    "Jordan McLaughlin":           "G",
    "Jordan Miller":               "G-F",
    "Jordan Poole":                "G",
    "Jordan Walsh":                "G-F",
    "Jose Alvarado":               "G",
    "Josh Giddey":                 "G-F",
    "Josh Green":                  "G-F",
    "Josh Hart":                   "G-F",
    "Josh Minott":                 "F",
    "Josh Okogie":                 "G-F",
    "Jrue Holiday":                "G",
    "Julian Champagnie":           "F",
    "Julian Phillips":             "F",
    "Julian Strawther":            "G-F",
    "Julius Randle":               "F",
    "Justin Champagnie":           "F",
    "Justin Edwards":              "G-F",
    "Jusuf Nurkić":                "C",
    # K
    "KJ Martin":                   "F",
    "KJ Simpson":                  "G",
    "Kai Jones":                   "C",
    "Karl-Anthony Towns":          "C",
    "Karlo Matković":              "F-C",
    "Kawhi Leonard":               "F",
    "Keaton Wallace":              "G",
    "Keegan Murray":               "F",
    "Kel'el Ware":                 "C",
    "Keldon Johnson":              "F",
    "Kelly Olynyk":                "F-C",
    "Kelly Oubre Jr.":             "G-F",
    "Kenrich Williams":            "G-F",
    "Kentavious Caldwell-Pope":    "G",
    "Keon Ellis":                  "G",
    "Keon Johnson":                "G",
    "Kessler Edwards":             "F",
    "Kevin Durant":                "F",
    "Kevin Huerter":               "G-F",
    "Kevin Love":                  "F-C",
    "Kevin Porter Jr.":            "G",
    "Kevon Looney":                "C",
    "Keyonte George":              "G",
    "Khris Middleton":             "G-F",
    "Klay Thompson":               "G",
    "Kobe Brown":                  "F",
    "Kris Dunn":                   "G",
    "Kris Murray":                 "F",
    "Kristaps Porziņģis":          "C",
    "Kyle Anderson":               "G-F",
    "Kyle Filipowski":             "F-C",
    "Kyle Kuzma":                  "F",
    "Kyle Lowry":                  "G",
    "Kyrie Irving":                "G",
    "Kyshawn George":              "G",
    # L
    "LaMelo Ball":                 "G",
    "Lamar Stevens":               "F",
    "Landry Shamet":               "G",
    "Larry Nance Jr.":             "F",
    "Lauri Markkanen":             "F",
    "LeBron James":                "F",
    "Lindy Waters III":            "G-F",
    "Lonnie Walker IV":            "G",
    "Lonzo Ball":                  "G",
    "Luguentz Dort":               "G-F",
    "Luka Dončić":                 "G",
    "Luka Garza":                  "C",
    "Luke Kennard":                "G",
    "Luke Kornet":                 "C",
    # M
    "Malaki Branham":              "G",
    "Malcolm Brogdon":             "G",
    "Malik Beasley":               "G",
    "Malik Monk":                  "G",
    "MarJon Beauchamp":            "G-F",
    "Marcus Sasser":               "G",
    "Marcus Smart":                "G",
    "Mark Williams":               "C",
    "Markelle Fultz":              "G",
    "Marvin Bagley III":           "F-C",
    "Mason Plumlee":               "C",
    "Matas Buzelis":               "F",
    "Max Christie":                "G",
    "Max Strus":                   "G-F",
    "Maxi Kleber":                 "F-C",
    "Maxwell Lewis":               "G-F",
    "Micah Potter":                "C",
    "Michael Porter Jr.":          "F",
    "Mikal Bridges":               "F",
    "Mike Conley":                 "G",
    "Miles Bridges":               "F",
    "Miles McBride":               "G",
    "Mitchell Robinson":           "C",
    "Mo Bamba":                    "C",
    "Monte Morris":                "G",
    "Moritz Wagner":               "C",
    "Moses Moody":                 "G-F",
    "Mouhamed Gueye":              "F",
    "Moussa Diabaté":              "F-C",
    "Myles Turner":                "C",
    # N
    "Naji Marshall":               "F",
    "Nate Williams":               "G-F",
    "Naz Reid":                    "F-C",
    "Neemias Queta":               "C",
    "Nic Claxton":                 "C",
    "Nick Richards":               "C",
    "Nick Smith Jr.":              "G",
    "Nickeil Alexander-Walker":    "G",
    "Nicolas Batum":               "F",
    "Nikola Jokić":                "C",
    "Nikola Jović":                "F",
    "Nikola Vučević":              "C",
    "Noah Clowney":                "F",
    "Norman Powell":               "G-F",
    # O
    "OG Anunoby":                  "F",
    "Obi Toppin":                  "F",
    "Ochai Agbaji":                "G-F",
    "Olivier-Maxence Prosper":     "F",
    "Onyeka Okongwu":              "C",
    "Orlando Robinson":            "C",
    "Oso Ighodaro":                "C",
    "Ousmane Dieng":               "G-F",
    # P
    "P.J. Washington":             "F",
    "Pacôme Dadiet":               "G-F",
    "Paolo Banchero":              "F",
    "Pascal Siakam":               "F",
    "Pat Connaughton":             "G",
    "Pat Spencer":                 "G",
    "Patrick Baldwin Jr.":         "G-F",
    "Patrick Williams":            "F",
    "Patty Mills":                 "G",
    "Paul George":                 "F",
    "Paul Reed":                   "F-C",
    "Payton Pritchard":            "G",
    "Pelle Larsson":               "G",
    "Peyton Watson":               "F",
    "Precious Achiuwa":            "F-C",
    # Q
    "Quentin Grimes":              "G",
    "Quenton Jackson":             "G-F",
    "Quinten Post":                "C",
    # R
    "RJ Barrett":                  "G-F",
    "Rayan Rupert":                "G-F",
    "Reece Beekman":               "G",
    "Reed Sheppard":               "G",
    "Reggie Jackson":              "G",
    "Richaun Holmes":              "C",
    "Ricky Council IV":            "G-F",
    "Rob Dillingham":              "G",
    "Robert Williams":             "C",
    "Ron Holland":                 "F",
    "Royce O'Neale":               "F",
    "Rudy Gobert":                 "C",
    "Rui Hachimura":               "F",
    "Russell Westbrook":           "G",
    "Ryan Dunn":                   "G-F",
    "Ryan Rollins":                "G",
    # S
    "Sam Hauser":                  "F",
    "Sam Merrill":                 "G",
    "Sandro Mamukelashvili":       "F",
    "Santi Aldama":                "F",
    "Scoot Henderson":             "G",
    "Scottie Barnes":              "F",
    "Scotty Pippen Jr.":           "G",
    "Seth Curry":                  "G",
    "Shaedon Sharpe":              "G",
    "Shai Gilgeous-Alexander":     "G",
    "Shake Milton":                "G",
    "Sidy Cissoko":                "G-F",
    "Simone Fontecchio":           "F",
    "Spencer Dinwiddie":           "G",
    "Spencer Jones":               "G-F",
    "Stanley Umude":               "G-F",
    "Stephen Curry":               "G",
    "Stephon Castle":              "G",
    "Steven Adams":                "C",
    "Svi Mykhailiuk":              "G-F",
    # T
    "T.J. McConnell":              "G",
    "Taj Gibson":                  "C",
    "Talen Horton-Tucker":         "G-F",
    "Tari Eason":                  "F",
    "Taurean Prince":              "F",
    "Terance Mann":                "G-F",
    "Terrence Shannon Jr.":        "G",
    "Terry Rozier":                "G",
    "Thomas Bryant":               "C",
    "Tidjane Salaün":              "F",
    "Tim Hardaway Jr.":            "G",
    "Tobias Harris":               "F",
    "Tony Bradley":                "C",
    "Torrey Craig":                "F",
    "Tosan Evbuomwan":             "F",
    "Toumani Camara":              "F",
    "Trae Young":                  "G",
    "Trayce Jackson-Davis":        "F-C",
    "Tre Jones":                   "G",
    "Trendon Watford":             "F",
    "Trevelin Queen":              "G",
    "Trey Alexander":              "G",
    "Trey Jemison":                "C",
    "Trey Lyles":                  "F",
    "Trey Murphy III":             "F",
    "Tristan Da Silva":            "F",
    "Tristan Thompson":            "C",
    "Tristan Vukcevic":            "C",
    "Ty Jerome":                   "G",
    "Tyler Herro":                 "G",
    "Tyler Kolek":                 "G",
    "Tyler Smith":                 "G-F",
    "Tyrese Haliburton":           "G",
    "Tyrese Martin":               "G-F",
    "Tyrese Maxey":                "G",
    "Tyus Jones":                  "G",
    # V
    "Vasilije Micić":              "G",
    "Victor Wembanyama":           "F-C",
    "Vince Williams Jr.":          "G-F",
    "Vít Krejčí":                  "G",
    # W
    "Walker Kessler":              "C",
    "Wendell Carter Jr.":          "C",
    "Wendell Moore Jr.":           "G",
    # X
    "Xavier Tillman Sr.":          "F-C",
    # Y
    "Yuki Kawamura":               "G",
    "Yves Missi":                  "C",
    # Z
    "Zaccharie Risacher":          "F",
    "Zach Collins":                "C",
    "Zach Edey":                   "C",
    "Zach LaVine":                 "G",
    "Zeke Nnaji":                  "C",
    "Ziaire Williams":             "F",
    "Zion Williamson":             "F",
}

df_players["POS"] = df_players["Player"].map(POS_DICT)

# Verificar jogadores sem posição mapeada
unmapped = df_players[df_players["POS"].isna()]["Player"].tolist()
if unmapped:
    print(f"  → {len(unmapped)} jogador(es) sem posição no dicionário: {unmapped}")
    df_players["POS"] = df_players["POS"].fillna("F")

# --- 2.7 POS_simple ---
pos_simple_map = {
    "G":   "Armador",
    "G-F": "Armador",
    "F":   "Ala",
    "F-C": "Ala",
    "C":   "Pivô",
}
df_players["POS_simple"] = df_players["POS"].map(pos_simple_map).fillna("Ala")

print(f"\nDistribuição por POS_simple:")
print(df_players["POS_simple"].value_counts().to_string())

# =============================================================================
# SEÇÃO 3 — ESTATÍSTICAS DESCRITIVAS
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 3 — ESTATÍSTICAS DESCRITIVAS")
print("=" * 70)

# --- 3.1 Tabela completa ---
print("\n--- 3.1 Medidas Descritivas por Variável ---\n")

VARS = {
    "PTS_mean":     "Pontos (méd./jogo)",
    "REB_mean":     "Rebotes (méd./jogo)",
    "AST_mean":     "Assistências (méd./jogo)",
    "MP_mean":      "Minutos (méd./jogo)",
    "FG_pct_mean":  "FG% (média)",
    "P3_pct_mean":  "3P% (média)",
}

rows_desc = []
for col, label in VARS.items():
    s = df_players[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    mean_v = s.mean()
    std_v  = s.std()
    rows_desc.append({
        "Variável":     label,
        "Média":        round(mean_v, 3),
        "Mediana":      round(s.median(), 3),
        "Desv.Pad.":    round(std_v, 3),
        "CV (%)":       round(std_v / mean_v * 100, 2) if mean_v != 0 else np.nan,
        "Mínimo":       round(s.min(), 3),
        "Máximo":       round(s.max(), 3),
        "Q1":           round(q1, 3),
        "Q2(Med.)":     round(s.median(), 3),
        "Q3":           round(q3, 3),
        "IQR":          round(q3 - q1, 3),
        "Assimetria":   round(stats.skew(s), 4),
        "Curtose":      round(stats.kurtosis(s), 4),
    })

desc_df = pd.DataFrame(rows_desc).set_index("Variável")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(desc_df.to_string())

# --- 3.2 Tabela de frequências de GP (6 classes) ---
print("\n\n--- 3.2 Tabela de Frequências de Jogos Disputados — GP (6 classes) ---\n")

n_total = len(df_players)
bins_gp = pd.cut(df_players["GP"], bins=6, include_lowest=True)
freq_gp  = bins_gp.value_counts(sort=False)
fi_gp    = freq_gp.values
fr_gp    = (fi_gp / n_total * 100).round(2)
Fi_gp    = fi_gp.cumsum()
Fr_gp    = np.round(Fi_gp / n_total * 100, 2)

def formatar_intervalo_gp(intervalo):
    inicio = int(np.floor(intervalo.left)) + 1
    fim = int(np.floor(intervalo.right))
    return f"{inicio} a {fim}"

freq_table_gp = pd.DataFrame({
    "Intervalo (GP)": [formatar_intervalo_gp(i) for i in freq_gp.index],
    "fi":             fi_gp,
    "fr (%)":         fr_gp,
    "Fi":             Fi_gp,
    "Fr (%)":         Fr_gp,
})
print(freq_table_gp.to_string(index=False))

# --- 3.3 Tabelas de frequências qualitativas ---
print("\n\n--- 3.3.1 Tabela de Frequências — Posição Simplificada (POS_simple) ---\n")
ps_freq = df_players["POS_simple"].value_counts()
ps_tbl  = pd.DataFrame({
    "Posição": ps_freq.index,
    "fi":      ps_freq.values,
    "fr (%)":  (ps_freq.values / n_total * 100).round(2),
})
print(ps_tbl.to_string(index=False))

print("\n\n--- 3.3.2 Tabela de Frequências — Posição Detalhada (POS) ---\n")
pos_freq = df_players["POS"].value_counts()
pos_tbl  = pd.DataFrame({
    "Posição": pos_freq.index,
    "fi":      pos_freq.values,
    "fr (%)":  (pos_freq.values / n_total * 100).round(2),
})
print(pos_tbl.to_string(index=False))

print("\n\n--- 3.3.3 Tabela de Frequências — Top 10 Times (mais jogadores) ---\n")
tm_freq = df_players["Tm"].value_counts().head(10)
tm_tbl  = pd.DataFrame({
    "Time":   tm_freq.index,
    "fi":     tm_freq.values,
    "fr (%)": (tm_freq.values / n_total * 100).round(2),
})
print(tm_tbl.to_string(index=False))

# =============================================================================
# SEÇÃO 4 — ANÁLISE POR GRUPOS
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 4 — ANÁLISE POR GRUPOS (POS_simple)")
print("=" * 70)

ORDEM_POS = ["Armador", "Ala", "Pivô"]

def q1f(x): return x.quantile(0.25)
def q3f(x): return x.quantile(0.75)

for col, label in [("PTS_mean", "Pontos"), ("REB_mean", "Rebotes"), ("AST_mean", "Assistências")]:
    print(f"\n--- {label} por Posição ---")
    grp = (
        df_players.groupby("POS_simple")[col]
        .agg(Media="mean", Mediana="median", DesvPad="std",
             Minimo="min", Maximo="max", Q1=q1f, Q3=q3f)
        .reindex(ORDEM_POS)
        .round(3)
    )
    grp.index.name = "Posição"
    grp.columns = ["Média", "Mediana", "Desv.Pad.", "Mínimo", "Máximo", "Q1", "Q3"]
    print(grp.to_string())

# =============================================================================
# SEÇÃO 5 — GRÁFICOS
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 5 — GRÁFICOS")
print("=" * 70)

sns.set_style("whitegrid")
FONTE = "Fonte: Kaggle — NBA Daily Leaders 2024-25"

def add_footer(fig, texto=FONTE):
    fig.text(0.5, 0.005, texto, ha="center", fontsize=9, color="gray", style="italic")

def salvar(fig, nome):
    caminho = GRAFICOS_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Gráfico salvo: {nome}")
    return str(caminho)

# ── Gráfico 1 — Histograma + KDE de PTS_mean ──────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["PTS_mean"].dropna(), kde=True, bins=25,
             color="#2196F3", ax=ax, line_kws={"lw": 2})
m_pts  = df_players["PTS_mean"].mean()
md_pts = df_players["PTS_mean"].median()
ax.axvline(m_pts,  color="red",    ls="--", lw=2, label=f"Média: {m_pts:.2f} pts")
ax.axvline(md_pts, color="orange", ls="-.", lw=2, label=f"Mediana: {md_pts:.2f} pts")
ax.set_title("Distribuição de Pontos por Jogo — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Pontos por Jogo (média)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "01_hist_pts.png")

# ── Gráfico 2 — Histograma + KDE de MP_mean ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["MP_mean"].dropna(), kde=True, bins=25,
             color="#4CAF50", ax=ax, line_kws={"lw": 2})
m_mp  = df_players["MP_mean"].mean()
md_mp = df_players["MP_mean"].median()
ax.axvline(m_mp,  color="red",    ls="--", lw=2, label=f"Média: {m_mp:.2f} min")
ax.axvline(md_mp, color="orange", ls="-.", lw=2, label=f"Mediana: {md_mp:.2f} min")
ax.set_title("Distribuição de Minutos por Jogo — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Minutos por Jogo (média)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "02_hist_mp.png")

# ── Gráfico 3 — Box plot horizontal PTS_mean × POS_simple ─────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.boxplot(data=df_players, y="POS_simple", x="PTS_mean",
            order=ORDEM_POS, palette="Set2", orient="h", ax=ax)
ax.set_title("Box Plot — Pontos por Jogo por Posição  (NBA 2024-25)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Pontos por Jogo (média)", fontsize=11)
ax.set_ylabel("Posição", fontsize=11)
add_footer(fig)
plt.tight_layout()
salvar(fig, "03_box_pts.png")

# ── Gráfico 4 — Box plot horizontal REB_mean × POS_simple ─────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.boxplot(data=df_players, y="POS_simple", x="REB_mean",
            order=ORDEM_POS, palette="Set2", orient="h", ax=ax)
ax.set_title("Box Plot — Rebotes por Jogo por Posição  (NBA 2024-25)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Rebotes por Jogo (média)", fontsize=11)
ax.set_ylabel("Posição", fontsize=11)
add_footer(fig)
plt.tight_layout()
salvar(fig, "04_box_reb.png")

# ── Gráfico 5 — Box plot horizontal AST_mean × POS_simple ─────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.boxplot(data=df_players, y="POS_simple", x="AST_mean",
            order=ORDEM_POS, palette="Set2", orient="h", ax=ax)
ax.set_title("Box Plot — Assistências por Jogo por Posição  (NBA 2024-25)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Assistências por Jogo (média)", fontsize=11)
ax.set_ylabel("Posição", fontsize=11)
add_footer(fig)
plt.tight_layout()
salvar(fig, "05_box_ast.png")

# ── Gráfico 6 — Barras verticais: contagem por POS_simple ─────────────────
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
pos_counts = df_players["POS_simple"].value_counts().reindex(ORDEM_POS)
cores6 = ["#2196F3", "#4CAF50", "#FF9800"]
barras = ax.bar(pos_counts.index, pos_counts.values, color=cores6,
                edgecolor="white", linewidth=0.8)
for bar, cnt in zip(barras, pos_counts.values):
    pct = cnt / n_total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Quantidade de Jogadores por Posição — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Posição", fontsize=11)
ax.set_ylabel("Número de Jogadores", fontsize=11)
ax.set_ylim(0, pos_counts.max() * 1.22)
add_footer(fig)
plt.tight_layout()
salvar(fig, "06_bar_posicao.png")

# ── Gráfico 7 — Barras horizontais: top 10 times ──────────────────────────
fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
top10 = df_players["Tm"].value_counts().head(10).sort_values()
cores7 = sns.color_palette("Blues_r", len(top10))
barras7 = ax.barh(top10.index, top10.values, color=cores7, edgecolor="white")
for bar, cnt in zip(barras7, top10.values):
    ax.text(bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left", fontsize=10, fontweight="bold")
ax.set_title("Top 10 Times com Mais Jogadores  (GP ≥ 20) — NBA 2024-25",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Número de Jogadores", fontsize=11)
ax.set_ylabel("Time", fontsize=11)
ax.set_xlim(0, top10.max() * 1.18)
add_footer(fig)
plt.tight_layout()
salvar(fig, "07_bar_times.png")

# ── Gráfico 8 — Barras agrupadas: PTS, REB, AST por POS_simple ────────────
medias_grp = (
    df_players.groupby("POS_simple")[["PTS_mean", "REB_mean", "AST_mean"]]
    .mean()
    .reindex(ORDEM_POS)
)

x8    = np.arange(len(ORDEM_POS))
w8    = 0.25
cores8 = ["#2196F3", "#4CAF50", "#FF9800"]
labs8  = ["Pontos", "Rebotes", "Assistências"]
cols8  = ["PTS_mean", "REB_mean", "AST_mean"]

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
for i, (col, lab, cor) in enumerate(zip(cols8, labs8, cores8)):
    offset = (i - 1) * w8
    barras8 = ax.bar(x8 + offset, medias_grp[col], w8,
                     label=lab, color=cor, edgecolor="white")
    for bar in barras8:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x8)
ax.set_xticklabels(ORDEM_POS, fontsize=11)
ax.set_title("Média de Pontos, Rebotes e Assistências por Posição — NBA 2024-25",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Posição", fontsize=11)
ax.set_ylabel("Média por Jogo", fontsize=11)
ax.legend(title="Estatística", fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "08_bar_agrupado.png")

# =============================================================================
# SEÇÃO 6 — TABELAS DE FREQUÊNCIAS DAS VARIÁVEIS CONTÍNUAS (STURGES) + MODA
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 6 — TABELAS DE FREQUÊNCIAS DAS VARIÁVEIS CONTÍNUAS (STURGES)")
print("=" * 70)

import math

def tabela_frequencias_sturges(series, nome_var):
    """Tabela de frequências via regra de Sturges. Retorna (tbl, k, h, fi_arr, li_list)."""
    s  = series.dropna().reset_index(drop=True)
    n  = len(s)
    k  = math.ceil(1 + 3.322 * math.log10(n))
    mn, mx = s.min(), s.max()
    edges = np.linspace(mn, mx, k + 1)
    edges[-1] = mx
    h  = edges[1] - edges[0]

    fi_list, xi_list, labels, li_list = [], [], [], []
    for i in range(k):
        li = edges[i]
        ls = edges[i + 1]
        xi = (li + ls) / 2
        mask = (s >= li) & (s <= ls) if i == k - 1 else (s >= li) & (s < ls)
        fi_list.append(int(mask.sum()))
        xi_list.append(round(xi, 4))
        fechamento = "]" if i == k - 1 else ")"
        labels.append(f"[{li:.4f}; {ls:.4f}{fechamento}")
        li_list.append(li)

    fi_arr = np.array(fi_list)
    fr_arr = np.round(fi_arr / n * 100, 2)
    Fi_arr = fi_arr.cumsum()
    Fr_arr = np.round(Fi_arr / n * 100, 2)

    tbl = pd.DataFrame({
        "Classe":   labels,
        "xi":       xi_list,
        "fi":       fi_arr,
        "fr (%)":   fr_arr,
        "Fi":       Fi_arr,
        "Fr (%)":   Fr_arr,
    })

    print(f"\n{'─'*72}")
    print(f"Variável: {nome_var}  |  n={n}  |  k={k} classes  |  h={h:.4f}")
    print(f"{'─'*72}")
    print(tbl.to_string(index=False))
    print(f"{'─'*72}")
    print(f"Total  fi={fi_arr.sum()}  |  Σfr={fi_arr.sum() / n * 100:.2f}%")
    return tbl, k, h, fi_arr, li_list


def moda_czuber(fi_arr, li_list, h):
    """Moda pelo método de Czuber (fórmula da classe modal)."""
    idx    = int(np.argmax(fi_arr))
    Li     = li_list[idx]
    f_mod  = fi_arr[idx]
    f_ant  = fi_arr[idx - 1] if idx > 0           else 0
    f_pos  = fi_arr[idx + 1] if idx < len(fi_arr) - 1 else 0
    denom  = 2 * f_mod - f_ant - f_pos
    return (Li + h / 2) if denom == 0 else Li + (f_mod - f_ant) / denom * h


print("\n--- 6.1 Tabelas de Frequências com Moda (Czuber) ---")
modas          = {}
freq_tables_ct = {}

for col, label in VARS.items():
    tbl, k, h, fi_arr, li_list = tabela_frequencias_sturges(df_players[col].dropna(), label)
    mo = moda_czuber(fi_arr, li_list, h)
    modas[col]          = round(mo, 4)
    freq_tables_ct[col] = (tbl, k, h, fi_arr, li_list)
    print(f"  → Moda de {label}: {mo:.4f}")

# --- 6.2 Tabela descritiva COMPLETA (com Moda) ---
print("\n\n--- 6.2 Medidas Descritivas COMPLETAS (incluindo Moda) ---\n")
rows_full = []
for col, label in VARS.items():
    s      = df_players[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    mean_v, std_v = s.mean(), s.std()
    rows_full.append({
        "Variável": label,
        "Média":    round(mean_v, 3),
        "Mediana":  round(s.median(), 3),
        "Moda":     round(modas[col], 3),
        "DP":       round(std_v, 3),
        "CV(%)":    round(std_v / mean_v * 100, 2) if mean_v != 0 else np.nan,
        "Q1":       round(q1, 3),
        "Q3":       round(q3, 3),
        "IQR":      round(q3 - q1, 3),
        "Assim.":   round(stats.skew(s), 4),
        "Curtose":  round(stats.kurtosis(s), 4),
    })
desc_full = pd.DataFrame(rows_full).set_index("Variável")
print(desc_full.to_string())

# =============================================================================
# SEÇÃO 7 — ANÁLISE COMPLETA DE GP (VARIÁVEL DISCRETA)
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 7 — ANÁLISE COMPLETA DE GP (VARIÁVEL DISCRETA)")
print("=" * 70)

gp       = df_players["GP"]
gp_n     = len(gp)
gp_mean  = gp.mean()
gp_med   = gp.median()
gp_mode  = int(gp.mode().iloc[0])
gp_std   = gp.std()
gp_var   = gp.var()
gp_cv    = gp_std / gp_mean * 100
gp_min   = int(gp.min())
gp_max   = int(gp.max())
gp_q1    = gp.quantile(0.25)
gp_q3    = gp.quantile(0.75)
gp_iqr   = gp_q3 - gp_q1
gp_li    = gp_q1 - 1.5 * gp_iqr
gp_ls    = gp_q3 + 1.5 * gp_iqr
gp_assim = round(stats.skew(gp), 4)
gp_kurt  = round(stats.kurtosis(gp), 4)

print(f"\n  Medidas Descritivas de GP (Jogos Disputados) — n = {gp_n}")
print(f"  {'─'*44}")
print(f"  {'Média:':<24} {gp_mean:.3f}")
print(f"  {'Mediana (Q2):':<24} {gp_med:.1f}")
print(f"  {'Moda:':<24} {gp_mode}")
print(f"  {'Desvio Padrão:':<24} {gp_std:.3f}")
print(f"  {'Variância:':<24} {gp_var:.3f}")
print(f"  {'CV (%):':<24} {gp_cv:.2f}%")
print(f"  {'Mínimo:':<24} {gp_min}")
print(f"  {'Máximo:':<24} {gp_max}")
print(f"  {'Q1:':<24} {gp_q1:.1f}")
print(f"  {'Q2 (Mediana):':<24} {gp_med:.1f}")
print(f"  {'Q3:':<24} {gp_q3:.1f}")
print(f"  {'IQR:':<24} {gp_iqr:.1f}")
print(f"  {'Assimetria:':<24} {gp_assim}")
print(f"  {'Curtose:':<24} {gp_kurt}")
print(f"  {'Limite Inferior:':<24} {gp_li:.1f}  (Q1 − 1,5·IQR)")
print(f"  {'Limite Superior:':<24} {gp_ls:.1f}  (Q3 + 1,5·IQR)")

gp_out_lo = df_players[gp < gp_li][["Player", "Tm", "POS_simple", "GP"]].sort_values("GP")
gp_out_hi = df_players[gp > gp_ls][["Player", "Tm", "POS_simple", "GP"]].sort_values("GP", ascending=False)

print(f"\n  Outliers GP < {gp_li:.1f}  →  {len(gp_out_lo)} jogador(es):")
print(gp_out_lo.to_string(index=False) if len(gp_out_lo) else "  Nenhum.")
print(f"\n  Outliers GP > {gp_ls:.1f}  →  {len(gp_out_hi)} jogador(es):")
print(gp_out_hi.to_string(index=False) if len(gp_out_hi) else "  Nenhum.")

# =============================================================================
# SEÇÃO 8 — OUTLIERS DAS VARIÁVEIS CONTÍNUAS PRINCIPAIS (GLOBAL)
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 8 — OUTLIERS GLOBAIS — PTS_mean, REB_mean, AST_mean")
print("=" * 70)

VARS_OUT = {
    "PTS_mean": "Pontos (pts/jogo)",
    "REB_mean": "Rebotes (reb/jogo)",
    "AST_mean": "Assistências (ast/jogo)",
}
outliers_data = {}

for col, label in VARS_OUT.items():
    s   = df_players[col].dropna()
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    iqr = q3 - q1
    li  = q1 - 1.5 * iqr
    ls  = q3 + 1.5 * iqr
    ol  = df_players[df_players[col] < li][["Player", "Tm", "POS_simple", col]].sort_values(col)
    oh  = df_players[df_players[col] > ls][["Player", "Tm", "POS_simple", col]].sort_values(col, ascending=False)
    outliers_data[col] = {"q1": q1, "q3": q3, "iqr": iqr, "li": li, "ls": ls, "low": ol, "high": oh}

    print(f"\n  ── {label} ({col}) ──")
    print(f"  Q1={q1:.3f}  Q3={q3:.3f}  IQR={iqr:.3f}")
    print(f"  LI={li:.3f}  LS={ls:.3f}")
    print(f"\n  Outliers superiores  ({len(oh)} jogadores)  {col} > {ls:.3f}:")
    print(oh.to_string(index=False) if len(oh) else "  Nenhum.")
    print(f"\n  Outliers inferiores  ({len(ol)} jogadores)  {col} < {li:.3f}:")
    print(ol.to_string(index=False) if len(ol) else "  Nenhum.")

# =============================================================================
# SEÇÃO 9 — GRÁFICOS ADICIONAIS (09 a 17)
# =============================================================================
print("\n" + "=" * 70)
print("SEÇÃO 9 — GRÁFICOS ADICIONAIS (09 a 17)")
print("=" * 70)

# ── Gráfico 9 — Histograma + KDE REB_mean ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["REB_mean"].dropna(), kde=True, bins=25,
             color="#FF9800", ax=ax, line_kws={"lw": 2})
m_reb, md_reb = df_players["REB_mean"].mean(), df_players["REB_mean"].median()
ax.axvline(m_reb,  color="red",    ls="--", lw=2, label=f"Média: {m_reb:.2f}")
ax.axvline(md_reb, color="orange", ls="-.", lw=2, label=f"Mediana: {md_reb:.2f}")
ax.set_title("Distribuição de Rebotes por Jogo — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Rebotes por Jogo (média)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "09_hist_reb.png")

# ── Gráfico 10 — Histograma + KDE AST_mean ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["AST_mean"].dropna(), kde=True, bins=25,
             color="#9C27B0", ax=ax, line_kws={"lw": 2})
m_ast, md_ast = df_players["AST_mean"].mean(), df_players["AST_mean"].median()
ax.axvline(m_ast,  color="red",    ls="--", lw=2, label=f"Média: {m_ast:.2f}")
ax.axvline(md_ast, color="orange", ls="-.", lw=2, label=f"Mediana: {md_ast:.2f}")
ax.set_title("Distribuição de Assistências por Jogo — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Assistências por Jogo (média)", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "10_hist_ast.png")

# ── Gráfico 11 — Histograma + KDE FG_pct_mean ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["FG_pct_mean"].dropna(), kde=True, bins=25,
             color="#009688", ax=ax, line_kws={"lw": 2})
m_fg, md_fg = df_players["FG_pct_mean"].mean(), df_players["FG_pct_mean"].median()
ax.axvline(m_fg,  color="red",    ls="--", lw=2, label=f"Média: {m_fg:.3f}")
ax.axvline(md_fg, color="orange", ls="-.", lw=2, label=f"Mediana: {md_fg:.3f}")
ax.set_title("Distribuição de FG% (Arremessos de Campo) — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Percentual de Acertos — FG%", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "11_hist_fg.png")

# ── Gráfico 12 — Histograma + KDE P3_pct_mean ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
sns.histplot(df_players["P3_pct_mean"].dropna(), kde=True, bins=25,
             color="#F44336", ax=ax, line_kws={"lw": 2})
m_3p, md_3p = df_players["P3_pct_mean"].mean(), df_players["P3_pct_mean"].median()
ax.axvline(m_3p,  color="red",    ls="--", lw=2, label=f"Média: {m_3p:.3f}")
ax.axvline(md_3p, color="orange", ls="-.", lw=2, label=f"Mediana: {md_3p:.3f}")
ax.set_title("Distribuição de 3P% (Arremessos de 3 Pontos) — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Percentual de Acertos — 3P%", fontsize=11)
ax.set_ylabel("Frequência", fontsize=11)
ax.legend(fontsize=10)
add_footer(fig)
plt.tight_layout()
salvar(fig, "12_hist_3p.png")

# ── Gráfico 13 — Histograma de GP (6 classes) ─────────────────────────────
gp_bins_6  = pd.cut(df_players["GP"], bins=6, include_lowest=True)
gp_freq6   = gp_bins_6.value_counts(sort=False)
intervalos = [formatar_intervalo_gp(iv) for iv in gp_freq6.index]
cores_gp   = sns.color_palette("Blues", len(intervalos))

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
barras_gp = ax.bar(range(len(intervalos)), gp_freq6.values,
                   color=cores_gp, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(barras_gp, gp_freq6.values):
    pct = cnt / n_total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(intervalos)))
ax.set_xticklabels(intervalos, rotation=20, ha="right", fontsize=9)
ax.set_title("Distribuição de Jogos Disputados (GP) — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Intervalo de Jogos Disputados", fontsize=11)
ax.set_ylabel("Número de Jogadores (fi)", fontsize=11)
ax.set_ylim(0, gp_freq6.max() * 1.22)
add_footer(fig)
plt.tight_layout()
salvar(fig, "13_hist_gp.png")

# ── Gráfico 14 — Box plot individual de GP ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.boxplot(df_players["GP"].dropna(), vert=False, patch_artist=True,
           boxprops=dict(facecolor="#2196F3", alpha=0.6),
           medianprops=dict(color="red", lw=2),
           whiskerprops=dict(lw=1.5),
           capprops=dict(lw=1.5),
           flierprops=dict(marker="o", color="#FF5722", alpha=0.6, markersize=7))
ax.set_title("Box Plot — Jogos Disputados (GP) — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Jogos Disputados (GP)", fontsize=11)
ax.set_yticks([])
add_footer(fig)
plt.tight_layout()
salvar(fig, "14_box_gp.png")

# ── Gráfico 15 — Box plots individuais PTS, REB, AST (1×3 subplots) ──────
fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=150)
for ax, (col, label, cor) in zip(axes, [
    ("PTS_mean", "Pontos (pts/jogo)",        "#2196F3"),
    ("REB_mean", "Rebotes (reb/jogo)",        "#FF9800"),
    ("AST_mean", "Assistências (ast/jogo)",   "#9C27B0"),
]):
    data = df_players[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor=cor, alpha=0.5),
               medianprops=dict(color="red", lw=2),
               whiskerprops=dict(lw=1.5),
               capprops=dict(lw=1.5),
               flierprops=dict(marker="o", color="#FF5722", alpha=0.5, markersize=5))
    q1_, q2_, q3_ = data.quantile(0.25), data.median(), data.quantile(0.75)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Valor (média por jogo)", fontsize=10)
    ax.annotate(f"Q1={q1_:.2f}", xy=(1.06, q1_), xycoords=("axes fraction", "data"),
                fontsize=8, color="navy", va="center")
    ax.annotate(f"Q2={q2_:.2f}", xy=(1.06, q2_), xycoords=("axes fraction", "data"),
                fontsize=8, color="red", va="center")
    ax.annotate(f"Q3={q3_:.2f}", xy=(1.06, q3_), xycoords=("axes fraction", "data"),
                fontsize=8, color="navy", va="center")
    ax.set_xticks([])

fig.suptitle("Box Plots — Distribuição das Variáveis Contínuas Principais — NBA 2024-25",
             fontsize=13, fontweight="bold", y=1.01)
add_footer(fig)
plt.tight_layout()
salvar(fig, "15_box_continuas.png")

# ── Gráfico 16 — Barras verticais POS (5 categorias) ─────────────────────
ORDEM_POS5  = ["G", "F", "G-F", "C", "F-C"]
pos5_counts = df_players["POS"].value_counts().reindex(ORDEM_POS5).fillna(0).astype(int)
LABELS_POS5 = [
    "Guard\n(G)", "Forward\n(F)", "Guard-Forward\n(G-F)",
    "Center\n(C)", "Forward-Center\n(F-C)",
]
cores16 = sns.color_palette("Set2", 5)

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
barras16 = ax.bar(LABELS_POS5, pos5_counts.values,
                  color=cores16, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(barras16, pos5_counts.values):
    pct = cnt / n_total * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{cnt}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Distribuição por Posição Detalhada (POS — 5 categorias) — NBA 2024-25",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Posição em Quadra", fontsize=11)
ax.set_ylabel("Número de Jogadores", fontsize=11)
ax.set_ylim(0, pos5_counts.max() * 1.28)
add_footer(fig)
plt.tight_layout()
salvar(fig, "16_bar_pos5.png")

# ── Gráfico 17 — Barras horizontais: TODOS os 30 times ────────────────────
all_times  = df_players["Tm"].value_counts().sort_values()
n_times    = len(all_times)
cores17    = sns.color_palette("Blues_r", n_times)
fig_height = max(9, n_times * 0.37)

fig, ax = plt.subplots(figsize=(12, fig_height), dpi=150)
barras17 = ax.barh(all_times.index, all_times.values,
                   color=cores17, edgecolor="white", linewidth=0.5, height=0.72)
for bar, cnt in zip(barras17, all_times.values):
    pct = cnt / n_total * 100
    ax.text(bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt} ({pct:.1f}%)", va="center", ha="left", fontsize=9)
ax.set_title("Jogadores Qualificados por Franquia — NBA 2024-25  (GP ≥ 20)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Número de Jogadores (fi)", fontsize=11)
ax.set_ylabel("Franquia (sigla)", fontsize=11)
ax.set_xlim(0, all_times.max() * 1.38)
add_footer(fig)
plt.tight_layout()
salvar(fig, "17_bar_todos_times.png")

# =============================================================================
# CONFIRMAÇÃO FINAL
# =============================================================================
graficos_salvos = sorted([f for f in os.listdir(GRAFICOS_DIR) if f.endswith(".png")])

print(f"\n{'=' * 70}")
print("Análise concluída com sucesso.")
print(f"\nTotal de gráficos em graficos/: {len(graficos_salvos)}")
for g in graficos_salvos:
    print(f"  {g}")
print(f"{'=' * 70}")

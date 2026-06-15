# -*- coding: utf-8 -*-
"""Dashboard interativo em estilo BI para a análise da NBA 2024-25."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="NBA 2024-25 | Dashboard BI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "dados" / "nba_dailyleaders_full_24_25.csv"
COLOR_MAP = {"Guard": "#1F77B4", "Forward": "#2CA58D", "Center": "#FF7F11"}
ORDERED_POSITIONS = ["Guard", "Forward", "Center"]
METRIC_LABELS = {
    "PTS_mean": "Pontos",
    "REB_mean": "Rebotes",
    "AST_mean": "Assistências",
    "MP_mean": "Minutos",
    "FG_pct_mean": "FG%",
    "P3_pct_mean": "3P%",
    "STL_mean": "Roubadas",
    "BLK_mean": "Tocos",
    "TOV_mean": "Turnovers",
    "PF_mean": "Faltas",
}


POS_DICT = {
    "A.J. Green": "G", "A.J. Lawson": "G-F", "AJ Johnson": "F",
    "Aaron Gordon": "F", "Aaron Holiday": "G", "Aaron Nesmith": "G-F",
    "Aaron Wiggins": "G-F", "Adam Flagler": "G", "Adem Bona": "C",
    "Ajay Mitchell": "G", "Al Horford": "F-C", "Alec Burks": "G",
    "Alex Caruso": "G", "Alex Ducas": "G-F", "Alex Len": "C",
    "Alex Sarr": "F-C", "Alperen Şengün": "C", "Amen Thompson": "G-F",
    "Amir Coffey": "G-F", "Andre Drummond": "C", "Andre Jackson Jr.": "G-F",
    "Andrew Nembhard": "G", "Andrew Wiggins": "F", "Anfernee Simons": "G",
    "Anthony Black": "G", "Anthony Davis": "C", "Anthony Edwards": "G",
    "Anthony Gill": "F", "Antonio Reeves": "G", "Ariel Hukporti": "C",
    "Ausar Thompson": "F", "Austin Reaves": "G", "Ayo Dosunmu": "G",
    "Bam Adebayo": "C", "Baylor Scheierman": "G-F", "Ben Sheppard": "G-F",
    "Ben Simmons": "G-F", "Bennedict Mathurin": "G-F", "Bilal Coulibaly": "G-F",
    "Bismack Biyombo": "C", "Blake Wesley": "G", "Bobby Portis": "F-C",
    "Bogdan Bogdanović": "G", "Bol Bol": "F-C", "Bones Hyland": "G",
    "Bradley Beal": "G", "Branden Carlson": "C", "Brandin Podziemski": "G",
    "Brandon Boston Jr.": "G-F", "Brandon Clarke": "F", "Brandon Miller": "F",
    "Brandon Williams": "G", "Brice Sensabaugh": "G-F", "Bronny James": "G",
    "Brook Lopez": "C", "Bruce Brown": "G-F", "Bub Carrington": "G",
    "Buddy Hield": "G", "CJ McCollum": "G", "Cade Cunningham": "G",
    "Caleb Houstan": "G-F", "Caleb Martin": "F", "Cam Reddish": "F",
    "Cam Spencer": "G", "Cam Thomas": "G", "Cam Whitmore": "F",
    "Cameron Johnson": "F", "Cameron Payne": "G", "Caris LeVert": "G-F",
    "Cason Wallace": "G", "Charles Bassey": "C", "Chet Holmgren": "F-C",
    "Chris Boucher": "F-C", "Chris Livingston": "G-F", "Chris Paul": "G",
    "Christian Braun": "G", "Christian Koloko": "C", "Clint Capela": "C",
    "Coby White": "G", "Cody Martin": "G-F", "Cody Williams": "G",
    "Colby Jones": "G-F", "Cole Anthony": "G", "Colin Castleton": "C",
    "Collin Gillespie": "G", "Collin Sexton": "G", "Corey Kispert": "G-F",
    "Cory Joseph": "G", "Craig Porter Jr.": "G", "D'Angelo Russell": "G",
    "DaQuan Jeffries": "G-F", "Dalano Banton": "G-F", "Dalen Terry": "G",
    "Dalton Knecht": "G-F", "Damian Lillard": "G", "Damion Lee": "G",
    "Daniel Gafford": "C", "Daniel Theis": "F-C", "Danté Exum": "G",
    "Dariq Whitehead": "G-F", "Darius Garland": "G", "David Roddy": "F",
    "Davion Mitchell": "G", "Day'Ron Sharpe": "C", "De'Aaron Fox": "G",
    "De'Andre Hunter": "F", "DeAndre Jordan": "C", "DeMar DeRozan": "G-F",
    "Dean Wade": "F", "Deandre Ayton": "C", "Dejounte Murray": "G",
    "Delon Wright": "G", "Deni Avdija": "F", "Dennis Schröder": "G",
    "Dereck Lively II": "C", "Derrick Jones Jr.": "F", "Derrick White": "G",
    "Desmond Bane": "G-F", "Devin Booker": "G", "Devin Carter": "G",
    "Devin Vassell": "G", "Dillon Brooks": "G-F", "Dillon Jones": "F",
    "Domantas Sabonis": "C", "Dominick Barlow": "F", "Donovan Clingan": "C",
    "Donovan Mitchell": "G", "Donte DiVincenzo": "G", "Dorian Finney-Smith": "F",
    "Doug McDermott": "F", "Draymond Green": "F", "Drew Eubanks": "C",
    "Drew Peterson": "F", "Duncan Robinson": "G-F", "Duop Reath": "C",
    "Dwight Powell": "C", "Dyson Daniels": "G", "Elfrid Payton": "G",
    "Enrique Freeman": "F-C", "Eric Gordon": "G", "Evan Mobley": "F-C",
    "Franz Wagner": "G-F", "Fred VanVleet": "G", "GG Jackson II": "F",
    "Gabe Vincent": "G", "Garrett Temple": "G", "Garrison Mathews": "G",
    "Gary Harris": "G", "Gary Payton II": "G", "Gary Trent Jr.": "G",
    "Georges Niang": "F", "Giannis Antetokounmpo": "F", "Goga Bitadze": "C",
    "Gradey Dick": "G-F", "Grayson Allen": "G", "Guerschon Yabusele": "F",
    "Gui Santos": "F", "Harrison Barnes": "F", "Haywood Highsmith": "F",
    "Herbert Jones": "G-F", "Hunter Tyson": "F", "Immanuel Quickley": "G",
    "Isaac Jones": "F", "Isaac Okoro": "G-F", "Isaiah Collier": "G",
    "Isaiah Hartenstein": "C", "Isaiah Joe": "G", "Isaiah Stewart": "F-C",
    "Isaiah Wong": "G", "Ivica Zubac": "C", "JD Davison": "G",
    "JT Thor": "F", "Ja Morant": "G", "Ja'Kobe Walter": "G",
    "Jabari Smith Jr.": "F", "Jabari Walker": "F", "Jaden Hardy": "G",
    "Jaden Ivey": "G", "Jaden McDaniels": "F", "Jaden Springer": "G",
    "Jae'Sean Tate": "F", "Jaime Jaquez Jr.": "G-F", "Jake LaRavia": "F",
    "Jakob Poeltl": "C", "Jalen Brunson": "G", "Jalen Duren": "C",
    "Jalen Green": "G", "Jalen Johnson": "F", "Jalen Pickett": "G",
    "Jalen Smith": "F-C", "Jalen Suggs": "G", "Jalen Williams": "G",
    "Jalen Wilson": "F", "Jamal Cain": "F", "Jamal Murray": "G",
    "Jamal Shead": "G", "James Harden": "G", "Jamison Battle": "F",
    "Jarace Walker": "F-C", "Jared Butler": "G", "Jared McCain": "G",
    "Jaren Jackson Jr.": "F-C", "Jarred Vanderbilt": "F", "Jarrett Allen": "C",
    "Javonte Green": "G-F", "Jaxson Hayes": "C", "Jay Huff": "C",
    "Jaylen Brown": "F", "Jaylen Clark": "G", "Jaylen Wells": "G-F",
    "Jaylin Williams": "F", "Jaylon Tyson": "G-F", "Jayson Tatum": "F",
    "Jeff Dowtin Jr.": "G", "Jeff Green": "F", "Jerami Grant": "F",
    "Jeremiah Robinson-Earl": "F", "Jeremy Sochan": "F", "Jericho Sims": "C",
    "Jett Howard": "G-F", "Jevon Carter": "G", "Jimmy Butler": "F",
    "Jock Landale": "C", "John Collins": "F", "John Konchar": "G-F",
    "Johnny Davis": "G", "Johnny Furphy": "G-F", "Johnny Juzang": "G",
    "Jonas Valančiūnas": "C", "Jonathan Isaac": "F", "Jonathan Kuminga": "F",
    "Jonathan Mogbo": "F-C", "Jordan Clarkson": "G", "Jordan Goodwin": "G",
    "Jordan Hawkins": "G", "Jordan McLaughlin": "G", "Jordan Miller": "G-F",
    "Jordan Poole": "G", "Jordan Walsh": "G-F", "Jose Alvarado": "G",
    "Josh Giddey": "G-F", "Josh Green": "G-F", "Josh Hart": "G-F",
    "Josh Minott": "F", "Josh Okogie": "G-F", "Jrue Holiday": "G",
    "Julian Champagnie": "F", "Julian Phillips": "F", "Julian Strawther": "G-F",
    "Julius Randle": "F", "Justin Champagnie": "F", "Justin Edwards": "G-F",
    "Jusuf Nurkić": "C", "KJ Martin": "F", "KJ Simpson": "G",
    "Kai Jones": "C", "Karl-Anthony Towns": "C", "Karlo Matković": "F-C",
    "Kawhi Leonard": "F", "Keaton Wallace": "G", "Keegan Murray": "F",
    "Kel'el Ware": "C", "Keldon Johnson": "F", "Kelly Olynyk": "F-C",
    "Kelly Oubre Jr.": "G-F", "Kenrich Williams": "G-F",
    "Kentavious Caldwell-Pope": "G", "Keon Ellis": "G", "Keon Johnson": "G",
    "Kessler Edwards": "F", "Kevin Durant": "F", "Kevin Huerter": "G-F",
    "Kevin Love": "F-C", "Kevin Porter Jr.": "G", "Kevon Looney": "C",
    "Keyonte George": "G", "Khris Middleton": "G-F", "Klay Thompson": "G",
    "Kobe Brown": "F", "Kris Dunn": "G", "Kris Murray": "F",
    "Kristaps Porziņģis": "C", "Kyle Anderson": "G-F", "Kyle Filipowski": "F-C",
    "Kyle Kuzma": "F", "Kyle Lowry": "G", "Kyrie Irving": "G",
    "Kyshawn George": "G", "LaMelo Ball": "G", "Lamar Stevens": "F",
    "Landry Shamet": "G", "Larry Nance Jr.": "F", "Lauri Markkanen": "F",
    "LeBron James": "F", "Lindy Waters III": "G-F", "Lonnie Walker IV": "G",
    "Lonzo Ball": "G", "Luguentz Dort": "G-F", "Luka Dončić": "G",
    "Luka Garza": "C", "Luke Kennard": "G", "Luke Kornet": "C",
    "Malaki Branham": "G", "Malcolm Brogdon": "G", "Malik Beasley": "G",
    "Malik Monk": "G", "MarJon Beauchamp": "G-F", "Marcus Sasser": "G",
    "Marcus Smart": "G", "Mark Williams": "C", "Markelle Fultz": "G",
    "Marvin Bagley III": "F-C", "Mason Plumlee": "C", "Matas Buzelis": "F",
    "Max Christie": "G", "Max Strus": "G-F", "Maxi Kleber": "F-C",
    "Maxwell Lewis": "G-F", "Micah Potter": "C", "Michael Porter Jr.": "F",
    "Mikal Bridges": "F", "Mike Conley": "G", "Miles Bridges": "F",
    "Miles McBride": "G", "Mitchell Robinson": "C", "Mo Bamba": "C",
    "Monte Morris": "G", "Moritz Wagner": "C", "Moses Moody": "G-F",
    "Mouhamed Gueye": "F", "Moussa Diabaté": "F-C", "Myles Turner": "C",
    "Naji Marshall": "F", "Nate Williams": "G-F", "Naz Reid": "F-C",
    "Neemias Queta": "C", "Nic Claxton": "C", "Nick Richards": "C",
    "Nick Smith Jr.": "G", "Nickeil Alexander-Walker": "G", "Nicolas Batum": "F",
    "Nikola Jokić": "C", "Nikola Jović": "F", "Nikola Vučević": "C",
    "Noah Clowney": "F", "Norman Powell": "G-F", "OG Anunoby": "F",
    "Obi Toppin": "F", "Ochai Agbaji": "G-F", "Olivier-Maxence Prosper": "F",
    "Onyeka Okongwu": "C", "Orlando Robinson": "C", "Oso Ighodaro": "C",
    "Ousmane Dieng": "G-F", "P.J. Washington": "F", "Pacôme Dadiet": "G-F",
    "Paolo Banchero": "F", "Pascal Siakam": "F", "Pat Connaughton": "G",
    "Pat Spencer": "G", "Patrick Baldwin Jr.": "G-F", "Patrick Williams": "F",
    "Patty Mills": "G", "Paul George": "F", "Paul Reed": "F-C",
    "Payton Pritchard": "G", "Pelle Larsson": "G", "Peyton Watson": "F",
    "Precious Achiuwa": "F-C", "Quentin Grimes": "G", "Quenton Jackson": "G-F",
    "Quinten Post": "C", "RJ Barrett": "G-F", "Rayan Rupert": "G-F",
    "Reece Beekman": "G", "Reed Sheppard": "G", "Reggie Jackson": "G",
    "Richaun Holmes": "C", "Ricky Council IV": "G-F", "Rob Dillingham": "G",
    "Robert Williams": "C", "Ron Holland": "F", "Royce O'Neale": "F",
    "Rudy Gobert": "C", "Rui Hachimura": "F", "Russell Westbrook": "G",
    "Ryan Dunn": "G-F", "Ryan Rollins": "G", "Sam Hauser": "F",
    "Sam Merrill": "G", "Sandro Mamukelashvili": "F", "Santi Aldama": "F",
    "Scoot Henderson": "G", "Scottie Barnes": "F", "Scotty Pippen Jr.": "G",
    "Seth Curry": "G", "Shaedon Sharpe": "G", "Shai Gilgeous-Alexander": "G",
    "Shake Milton": "G", "Sidy Cissoko": "G-F", "Simone Fontecchio": "F",
    "Spencer Dinwiddie": "G", "Spencer Jones": "G-F", "Stanley Umude": "G-F",
    "Stephen Curry": "G", "Stephon Castle": "G", "Steven Adams": "C",
    "Svi Mykhailiuk": "G-F", "T.J. McConnell": "G", "Taj Gibson": "C",
    "Talen Horton-Tucker": "G-F", "Tari Eason": "F", "Taurean Prince": "F",
    "Terance Mann": "G-F", "Terrence Shannon Jr.": "G", "Terry Rozier": "G",
    "Thomas Bryant": "C", "Tidjane Salaün": "F", "Tim Hardaway Jr.": "G",
    "Tobias Harris": "F", "Tony Bradley": "C", "Torrey Craig": "F",
    "Tosan Evbuomwan": "F", "Toumani Camara": "F", "Trae Young": "G",
    "Trayce Jackson-Davis": "F-C", "Tre Jones": "G", "Trendon Watford": "F",
    "Trevelin Queen": "G", "Trey Alexander": "G", "Trey Jemison": "C",
    "Trey Lyles": "F", "Trey Murphy III": "F", "Tristan Da Silva": "F",
    "Tristan Thompson": "C", "Tristan Vukcevic": "C", "Ty Jerome": "G",
    "Tyler Herro": "G", "Tyler Kolek": "G", "Tyler Smith": "G-F",
    "Tyrese Haliburton": "G", "Tyrese Martin": "G-F", "Tyrese Maxey": "G",
    "Tyus Jones": "G", "Vasilije Micić": "G", "Victor Wembanyama": "F-C",
    "Vince Williams Jr.": "G-F", "Vít Krejčí": "G", "Walker Kessler": "C",
    "Wendell Carter Jr.": "C", "Wendell Moore Jr.": "G",
    "Xavier Tillman Sr.": "F-C", "Yuki Kawamura": "G", "Yves Missi": "C",
    "Zaccharie Risacher": "F", "Zach Collins": "C", "Zach Edey": "C",
    "Zach LaVine": "G", "Zeke Nnaji": "C", "Ziaire Williams": "F",
    "Zion Williamson": "F",
}


@st.cache_data(show_spinner="Carregando dados da temporada 2024-25...")
def load_players() -> pd.DataFrame:
    raw = pd.read_csv(CSV_PATH, encoding="utf-8")
    raw = raw.rename(columns={"Unnamed: 3": "Home_Away"})
    raw["Home_Away"] = raw["Home_Away"].fillna("Home").replace("@", "Away")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")

    def mp_to_decimal(value):
        try:
            text = str(value).strip()
            if ":" in text:
                minutes, seconds = text.split(":")
                return int(minutes) + int(seconds) / 60
            return float(text)
        except Exception:
            return np.nan

    raw["MP_decimal"] = raw["MP"].apply(mp_to_decimal)

    numeric_columns = [
        "FG%", "3P%", "FT%", "PTS", "TRB", "AST", "STL",
        "BLK", "TOV", "PF", "GmSc",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    def safe_mode(series: pd.Series):
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else series.iloc[0]

    players = (
        raw.groupby("Player")
        .agg(
            PTS_mean=("PTS", "mean"),
            REB_mean=("TRB", "mean"),
            AST_mean=("AST", "mean"),
            STL_mean=("STL", "mean"),
            BLK_mean=("BLK", "mean"),
            TOV_mean=("TOV", "mean"),
            PF_mean=("PF", "mean"),
            MP_mean=("MP_decimal", "mean"),
            FG_pct_mean=("FG%", "mean"),
            P3_pct_mean=("3P%", "mean"),
            GmSc_mean=("GmSc", "mean"),
            Tm=("Tm", safe_mode),
        )
        .reset_index()
    )

    games_played = raw.groupby("Player").size().rename("GP").reset_index()
    players = players.merge(games_played, on="Player", how="left")
    players = players[players["GP"] >= 20].copy()

    players["POS"] = players["Player"].map(POS_DICT).fillna("F")
    simple_map = {
        "G": "Guard",
        "G-F": "Guard",
        "F": "Forward",
        "F-C": "Forward",
        "C": "Center",
    }
    players["POS_simple"] = players["POS"].map(simple_map).fillna("Forward")

    team_counts = players.groupby("Tm")["Player"].transform("count")
    players["share_team_filtered"] = (1 / team_counts) * 100
    return players.sort_values("PTS_mean", ascending=False).reset_index(drop=True)


def inject_css():
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(31,119,180,0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(255,127,17,0.16), transparent 24%),
                    linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
            }
            .hero-card {
                background: linear-gradient(120deg, #0f2748 0%, #1f4e79 55%, #2ca58d 100%);
                border-radius: 22px;
                padding: 26px 30px;
                color: #ffffff;
                margin-bottom: 18px;
                box-shadow: 0 18px 40px rgba(15, 39, 72, 0.18);
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 6px;
            }
            .hero-subtitle {
                font-size: 0.98rem;
                opacity: 0.9;
                margin-bottom: 0;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(15, 39, 72, 0.08);
                border-radius: 18px;
                padding: 18px 18px 14px 18px;
                box-shadow: 0 8px 24px rgba(30, 55, 90, 0.08);
                min-height: 126px;
            }
            .metric-label {
                color: #4f6178;
                font-size: 0.88rem;
                font-weight: 600;
                letter-spacing: 0.02em;
                margin-bottom: 8px;
            }
            .metric-value {
                color: #10233f;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.0;
                margin-bottom: 8px;
            }
            .metric-foot {
                color: #61748b;
                font-size: 0.84rem;
                margin-bottom: 0;
            }
            .section-note {
                background: rgba(255, 255, 255, 0.82);
                border-left: 5px solid #1f77b4;
                padding: 14px 16px;
                border-radius: 10px;
                color: #21364d;
                margin: 6px 0 8px 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def make_histogram(frame: pd.DataFrame, column: str, title: str, color: str):
    mean_value = frame[column].mean()
    median_value = frame[column].median()
    fig = px.histogram(
        frame,
        x=column,
        nbins=24,
        opacity=0.84,
        color_discrete_sequence=[color],
        title=title,
    )
    fig.add_vline(
        x=mean_value,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Média: {mean_value:.2f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_value,
        line_dash="dot",
        line_color="#ff7f0e",
        annotation_text=f"Mediana: {median_value:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=56, b=20),
        xaxis_title=None,
        yaxis_title="Frequência",
    )
    return fig


def make_boxplot(frame: pd.DataFrame, column: str, label: str):
    fig = px.box(
        frame,
        x=column,
        y="POS_simple",
        color="POS_simple",
        category_orders={"POS_simple": ORDERED_POSITIONS},
        color_discrete_map=COLOR_MAP,
        points="outliers",
        title=f"{label} por posição",
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=56, b=20),
        xaxis_title=label,
        yaxis_title="Posição",
    )
    return fig


def prepare_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Player", "Tm", "POS_simple", "POS", "GP", "PTS_mean", "REB_mean",
        "AST_mean", "MP_mean", "FG_pct_mean", "P3_pct_mean", "GmSc_mean",
    ]
    table = frame[columns].rename(
        columns={
            "Player": "Jogador",
            "Tm": "Time",
            "POS_simple": "Posição",
            "POS": "Pos. Detalhada",
            "GP": "GP",
            "PTS_mean": "PTS",
            "REB_mean": "REB",
            "AST_mean": "AST",
            "MP_mean": "MIN",
            "FG_pct_mean": "FG%",
            "P3_pct_mean": "3P%",
            "GmSc_mean": "GmSc",
        }
    ).copy()
    return table


inject_css()
df_players = load_players()


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Dashboard BI — Análise NBA 2024-25</div>
        <p class="hero-subtitle">
            Painel interativo complementar, em estilo executivo, para reforçar a apresentação
            estatística do trabalho com filtros, comparações e visuais exploratórios.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-note">
        Este painel não substitui o relatório escrito manualmente. Ele funciona como um
        apoio visual no estilo Power BI para valorizar a exploração dos resultados.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Filtros do painel")
    selected_positions = st.multiselect(
        "Posições simplificadas",
        options=ORDERED_POSITIONS,
        default=ORDERED_POSITIONS,
    )
    teams = sorted(df_players["Tm"].dropna().unique().tolist())
    selected_teams = st.multiselect(
        "Franquias",
        options=teams,
        default=teams,
    )
    gp_range = st.slider(
        "Faixa de jogos (GP)",
        min_value=int(df_players["GP"].min()),
        max_value=int(df_players["GP"].max()),
        value=(20, int(df_players["GP"].max())),
    )
    fg_threshold = st.slider(
        "FG% mínimo",
        min_value=0.15,
        max_value=0.80,
        value=0.15,
        step=0.01,
    )
    top_n = st.slider("Tamanho dos rankings", 5, 30, 12)
    focus_player = st.selectbox(
        "Jogador em destaque",
        options=df_players["Player"].tolist(),
        index=0,
    )
    st.caption("Base: Kaggle — NBA Daily Leaders 2024-25")


filtered = df_players[
    df_players["POS_simple"].isin(selected_positions)
    & df_players["Tm"].isin(selected_teams)
    & df_players["GP"].between(gp_range[0], gp_range[1])
    & (df_players["FG_pct_mean"] >= fg_threshold)
].copy()

filtered = filtered.sort_values(["PTS_mean", "AST_mean"], ascending=False).reset_index(drop=True)

if filtered.empty:
    st.error("Nenhum jogador atende aos filtros atuais. Ajuste os filtros para continuar.")
    st.stop()


period_start = "2024-10-22"
period_end = "2025-06-22"
league_average_pts = filtered["PTS_mean"].mean()
league_average_reb = filtered["REB_mean"].mean()
league_average_ast = filtered["AST_mean"].mean()

tabs = st.tabs(
    [
        "Resumo Executivo",
        "Exploração Interativa",
        "Distribuições",
        "Rankings e Tabela",
    ]
)


with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Jogadores filtrados", f"{len(filtered)}", f"Período da base: {period_start} a {period_end}")
    with c2:
        metric_card("Pontos médios", f"{league_average_pts:.1f}", "Média de pontos por jogo")
    with c3:
        metric_card("Rebotes médios", f"{league_average_reb:.1f}", "Média de rebotes por jogo")
    with c4:
        metric_card("Assistências médias", f"{league_average_ast:.1f}", "Média de assistências por jogo")

    left, right = st.columns([1.1, 1])
    with left:
        scatter = px.scatter(
            filtered,
            x="PTS_mean",
            y="AST_mean",
            size="REB_mean",
            color="POS_simple",
            hover_name="Player",
            hover_data={"Tm": True, "GP": True, "MP_mean": ":.1f"},
            color_discrete_map=COLOR_MAP,
            title="Perfil ofensivo: pontos x assistências",
        )
        scatter.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=56, b=20),
            xaxis_title="Pontos por jogo",
            yaxis_title="Assistências por jogo",
        )
        st.plotly_chart(scatter, use_container_width=True)
    with right:
        pos_summary = (
            filtered.groupby("POS_simple")
            .size()
            .reindex(ORDERED_POSITIONS)
            .dropna()
            .reset_index(name="Jogadores")
        )
        donut = px.pie(
            pos_summary,
            names="POS_simple",
            values="Jogadores",
            hole=0.58,
            color="POS_simple",
            color_discrete_map=COLOR_MAP,
            title="Composição da amostra por posição",
        )
        donut.update_traces(textposition="inside", textinfo="percent+label")
        donut.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=56, b=20))
        st.plotly_chart(donut, use_container_width=True)

    heatmap_source = (
        filtered.groupby("POS_simple")[["PTS_mean", "REB_mean", "AST_mean", "MP_mean", "FG_pct_mean", "P3_pct_mean"]]
        .mean()
        .reindex(ORDERED_POSITIONS)
        .rename(columns={
            "PTS_mean": "PTS",
            "REB_mean": "REB",
            "AST_mean": "AST",
            "MP_mean": "MIN",
            "FG_pct_mean": "FG%",
            "P3_pct_mean": "3P%",
        })
    )
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_source.values,
            x=heatmap_source.columns.tolist(),
            y=heatmap_source.index.tolist(),
            colorscale="Blues",
            text=np.round(heatmap_source.values, 2),
            texttemplate="%{text}",
        )
    )
    heatmap.update_layout(
        title="Médias por posição",
        template="plotly_white",
        margin=dict(l=20, r=20, t=56, b=20),
    )
    st.plotly_chart(heatmap, use_container_width=True)

    top_scorer = filtered.iloc[0]
    best_rebounder = filtered.sort_values("REB_mean", ascending=False).iloc[0]
    best_playmaker = filtered.sort_values("AST_mean", ascending=False).iloc[0]
    st.markdown(
        f"""
        **Leituras rápidas do painel**

        - O jogador com maior média de pontos no filtro atual é **{top_scorer['Player']}**
          ({top_scorer['Tm']}) com **{top_scorer['PTS_mean']:.1f} pts/jogo**.
        - O principal reboteiro no recorte atual é **{best_rebounder['Player']}**
          com **{best_rebounder['REB_mean']:.1f} reb/jogo**.
        - O maior destaque em criação de jogadas é **{best_playmaker['Player']}**
          com **{best_playmaker['AST_mean']:.1f} ast/jogo**.
        """
    )


with tabs[1]:
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        x_metric = st.selectbox(
            "Eixo X",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda key: METRIC_LABELS[key],
            index=0,
        )
        y_metric = st.selectbox(
            "Eixo Y",
            options=list(METRIC_LABELS.keys()),
            format_func=lambda key: METRIC_LABELS[key],
            index=2,
        )
        size_metric = st.selectbox(
            "Tamanho da bolha",
            options=["GP", "REB_mean", "AST_mean", "MP_mean"],
            format_func=lambda key: "Jogos" if key == "GP" else METRIC_LABELS.get(key, key),
            index=0,
        )
        dynamic_scatter = px.scatter(
            filtered,
            x=x_metric,
            y=y_metric,
            size=size_metric,
            color="POS_simple",
            hover_name="Player",
            hover_data={"Tm": True, "GP": True},
            color_discrete_map=COLOR_MAP,
            title="Painel exploratório customizável",
        )
        dynamic_scatter.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=56, b=20),
            xaxis_title=METRIC_LABELS.get(x_metric, x_metric),
            yaxis_title=METRIC_LABELS.get(y_metric, y_metric),
        )
        st.plotly_chart(dynamic_scatter, use_container_width=True)

    with col_b:
        player_row = filtered[filtered["Player"] == focus_player]
        if player_row.empty:
            player_row = filtered.iloc[[0]]
        player_row = player_row.iloc[0]

        position_average = (
            filtered[filtered["POS_simple"] == player_row["POS_simple"]][["PTS_mean", "REB_mean", "AST_mean", "MP_mean"]]
            .mean()
        )
        comparison = pd.DataFrame(
            {
                "Indicador": ["PTS", "REB", "AST", "MIN"],
                "Jogador": [
                    player_row["PTS_mean"],
                    player_row["REB_mean"],
                    player_row["AST_mean"],
                    player_row["MP_mean"],
                ],
                "Média da posição": position_average.values,
            }
        )
        compare_fig = go.Figure()
        compare_fig.add_trace(
            go.Bar(
                x=comparison["Indicador"],
                y=comparison["Jogador"],
                name=player_row["Player"],
                marker_color="#1f77b4",
            )
        )
        compare_fig.add_trace(
            go.Bar(
                x=comparison["Indicador"],
                y=comparison["Média da posição"],
                name=f"Média dos {player_row['POS_simple']}",
                marker_color="#ff7f11",
            )
        )
        compare_fig.update_layout(
            barmode="group",
            template="plotly_white",
            title=f"Jogador em destaque: {player_row['Player']}",
            margin=dict(l=20, r=20, t=56, b=20),
            yaxis_title="Valor médio",
        )
        st.plotly_chart(compare_fig, use_container_width=True)

        st.dataframe(
            pd.DataFrame(
                {
                    "Indicador": ["Time", "Posição", "GP", "PTS", "REB", "AST", "MIN", "FG%", "3P%"],
                    "Valor": [
                        player_row["Tm"],
                        player_row["POS_simple"],
                        int(player_row["GP"]),
                        round(player_row["PTS_mean"], 2),
                        round(player_row["REB_mean"], 2),
                        round(player_row["AST_mean"], 2),
                        round(player_row["MP_mean"], 2),
                        format_pct(player_row["FG_pct_mean"]),
                        format_pct(player_row["P3_pct_mean"]),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with tabs[2]:
    metric_choice = st.selectbox(
        "Variável para distribuição",
        options=["PTS_mean", "REB_mean", "AST_mean", "MP_mean", "FG_pct_mean", "P3_pct_mean", "GP"],
        format_func=lambda key: "Jogos (GP)" if key == "GP" else METRIC_LABELS[key],
    )
    dist_left, dist_right = st.columns([1.05, 0.95])
    with dist_left:
        hist = make_histogram(
            filtered,
            metric_choice,
            f"Distribuição — {'Jogos (GP)' if metric_choice == 'GP' else METRIC_LABELS[metric_choice]}",
            "#4C78A8",
        )
        st.plotly_chart(hist, use_container_width=True)
    with dist_right:
        if metric_choice != "GP":
            box = make_boxplot(filtered, metric_choice, METRIC_LABELS[metric_choice])
            st.plotly_chart(box, use_container_width=True)
        else:
            team_box = px.box(
                filtered,
                x="GP",
                color="POS_simple",
                color_discrete_map=COLOR_MAP,
                title="Distribuição de jogos disputados",
            )
            team_box.update_layout(
                template="plotly_white",
                showlegend=True,
                margin=dict(l=20, r=20, t=56, b=20),
                xaxis_title="Jogos disputados",
            )
            st.plotly_chart(team_box, use_container_width=True)

    distribution_series = filtered[metric_choice].dropna()
    stat1, stat2, stat3, stat4, stat5 = st.columns(5)
    stat1.metric("Média", f"{distribution_series.mean():.2f}")
    stat2.metric("Mediana", f"{distribution_series.median():.2f}")
    stat3.metric("Desvio padrão", f"{distribution_series.std():.2f}")
    stat4.metric("Q1", f"{distribution_series.quantile(0.25):.2f}")
    stat5.metric("Q3", f"{distribution_series.quantile(0.75):.2f}")


with tabs[3]:
    ranking_metric = st.selectbox(
        "Ranking principal",
        options=["PTS_mean", "REB_mean", "AST_mean", "MP_mean", "FG_pct_mean", "P3_pct_mean", "GmSc_mean"],
        format_func=lambda key: "GmSc" if key == "GmSc_mean" else METRIC_LABELS.get(key, key),
    )
    ranking = filtered.nlargest(top_n, ranking_metric)[
        ["Player", "Tm", "POS_simple", "GP", ranking_metric, "PTS_mean", "REB_mean", "AST_mean", "MP_mean"]
    ].copy()
    ranking_label = "GmSc" if ranking_metric == "GmSc_mean" else METRIC_LABELS[ranking_metric]
    ranking = ranking.rename(
        columns={
            "Player": "Jogador",
            "Tm": "Time",
            "POS_simple": "Posição",
            "GP": "GP",
            ranking_metric: ranking_label,
            "PTS_mean": "PTS",
            "REB_mean": "REB",
            "AST_mean": "AST",
            "MP_mean": "MIN",
        }
    )

    rank_left, rank_right = st.columns([1.15, 0.85])
    with rank_left:
        chart_source = ranking.sort_values(ranking_label, ascending=True)
        bar = px.bar(
            chart_source,
            x=ranking_label,
            y="Jogador",
            color="Posição",
            orientation="h",
            color_discrete_map=COLOR_MAP,
            title=f"Top {top_n} jogadores por {ranking_label}",
            text_auto=".2f",
        )
        bar.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=56, b=20),
            xaxis_title=ranking_label,
            yaxis_title=None,
        )
        st.plotly_chart(bar, use_container_width=True)
    with rank_right:
        team_panel = (
            filtered.groupby("Tm")
            .size()
            .sort_values(ascending=False)
            .head(12)
            .reset_index(name="Jogadores")
        )
        team_fig = px.bar(
            team_panel.sort_values("Jogadores", ascending=True),
            x="Jogadores",
            y="Tm",
            orientation="h",
            color="Jogadores",
            color_continuous_scale="Blues",
            title="Franquias com mais jogadores no recorte",
        )
        team_fig.update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=56, b=20),
            xaxis_title="Jogadores",
            yaxis_title=None,
        )
        st.plotly_chart(team_fig, use_container_width=True)

    st.subheader("Tabela analítica completa")
    table = prepare_table(filtered)
    st.dataframe(
        table.style.format(
            {
                "PTS": "{:.2f}",
                "REB": "{:.2f}",
                "AST": "{:.2f}",
                "MIN": "{:.2f}",
                "FG%": "{:.3f}",
                "3P%": "{:.3f}",
                "GmSc": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar tabela filtrada em CSV",
        data=csv_data,
        file_name="painel_bi_nba_filtrado.csv",
        mime="text/csv",
    )


st.divider()
st.caption(
    "Projeto de Probabilidade e Estatística A — UFG | "
    "Painel complementar em Streamlit + Plotly | "
    "Objetivo: dar suporte visual ao relatório manual do grupo."
)

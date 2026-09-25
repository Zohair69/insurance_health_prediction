# =============================================================
#  Aegis Insurance - Simulateur de frais médicaux
#  Projet Simplon "Machine Learning & Régression" - Zohair & Brice
#
#  L'application se lit comme un parcours en 4 étapes :
#  1. Le constat  ->  2. Les facteurs de risque  ->  3. Le choix du modèle  ->  4. Le simulateur
#
#  Tous les chiffres affichés sont calculés par le code à partir de insurance-data.csv,
#  du modèle (modele_aegis.joblib) ou des résultats de la section 9 du notebook.
# =============================================================

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Aegis - Frais médicaux", page_icon="🩺", layout="wide")

# Deux couleurs utilisées dans tous les graphiques : bleu = non-fumeurs, orange = fumeurs
COULEURS = {"Non-fumeur": "#1F6F8B", "Fumeur": "#E9A23B"}

# Noms des colonnes en français pour les bulles d'information des graphiques
LIBELLES = {"age": "Âge", "bmi": "IMC", "children": "Enfants", "sex": "Sexe",
            "region": "Région", "expenses": "Frais annuels ($)", "statut": "Statut"}


# ---------- Chargement des données et du modèle ----------
# @st.cache_... : Streamlit ne recharge pas les fichiers à chaque clic

@st.cache_data
def charger_donnees():
    df = pd.read_csv("insurance-data.csv").drop_duplicates()
    # Colonne en français pour la légende des graphiques
    df["statut"] = df["smoker"].map({"no": "Non-fumeur", "yes": "Fumeur"})
    return df


@st.cache_resource
def charger_modele():
    # Fichier créé à la section 10 du notebook : modèle + scaler + ordre des colonnes
    return joblib.load("modele_aegis.joblib")


df = charger_donnees()
artefacts = charger_modele()
modele = artefacts["modele"]
scaler = artefacts["scaler"]
colonnes = artefacts["colonnes"]

# Résultats des 5 modèles sur le jeu de test : valeurs issues de la section 9 du notebook
resultats = pd.DataFrame({
    "Modèle": ["Régression Linéaire", "KNN", "Arbre de Décision",
               "Random Forest", "Random Forest optimisé"],
    "MAE ($)": [4177, 3472, 2917, 2665, 2440],
    "MSE": [35481472, 29900869, 38918076, 22227850, 18137957],
    "R²": [0.807, 0.837, 0.788, 0.879, 0.901],
})
meilleur = resultats.loc[resultats["R²"].idxmax()]


def predire_frais(age, sex, bmi, children, smoker, region):
    """Même encodage que dans le notebook (pd.get_dummies avec drop_first=True)."""
    client = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": sex == "male",
        "smoker_yes": smoker == "yes",
        "region_northwest": region == "northwest",
        "region_southeast": region == "southeast",
        "region_southwest": region == "southwest",
    }])[colonnes]
    return modele.predict(scaler.transform(client))[0]


def dollars(montant):
    """Affiche un montant avec un espace comme séparateur : 12 345 $"""
    return f"{montant:,.0f} $".replace(",", " ")


def dollars_texte(montant):
    """Même chose pour les phrases (st.write, st.caption).
    Le \\ devant $ évite que Streamlit lise le texte entre deux $ comme une formule mathématique."""
    return dollars(montant).replace("$", "\\$")


# ---------- Bandeau en haut de chaque page ----------
# Un bloc HTML avec un dégradé de bleus (CSS), affiché avant le contenu de la page
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1B4B7A, #2B97BF);
                padding: 22px 30px; border-radius: 6px; margin-bottom: 20px;">
        <div style="color: white; font-size: 30px; font-weight: 700;">Aegis Insurance</div>
        <div style="color: #D6EAF5; font-size: 16px;">Simulateur de frais médicaux</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Menu de navigation (barre latérale) ----------
st.sidebar.title("🩺 Aegis Insurance")
st.sidebar.write("Estimer les frais médicaux annuels d'un client.")

etape = st.sidebar.radio(
    "Parcours",
    ["1. Le constat", "2. Les facteurs de risque", "3. Le choix du modèle", "4. Le simulateur"],
)

st.sidebar.divider()
st.sidebar.caption("Projet Simplon - Machine Learning & Régression\n\nZohair & Brice")


# =============================================================
# ÉTAPE 1 : LE CONSTAT
# =============================================================
if etape == "1. Le constat":
    moy_fumeur = df[df["smoker"] == "yes"]["expenses"].mean()
    moy_non_fumeur = df[df["smoker"] == "no"]["expenses"].mean()
    ratio = moy_fumeur / moy_non_fumeur  # frais moyens fumeurs / frais moyens non-fumeurs

    st.title(f"Un fumeur coûte en moyenne {ratio:.1f} fois plus cher".replace(".", ","))
    st.write(
        f"Sur les {len(df)} clients étudiés, les frais médicaux annuels d'un fumeur atteignent "
        f"{dollars_texte(moy_fumeur)} en moyenne, contre {dollars_texte(moy_non_fumeur)} pour un non-fumeur."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Clients étudiés", len(df))
    k2.metric("Frais moyens", dollars(df["expenses"].mean()))
    k3.metric("Part de fumeurs", f"{(df['smoker'] == 'yes').mean():.0%}")

    g1, g2 = st.columns(2)

    with g1:
        fig = px.histogram(df, x="expenses", nbins=30, color="statut", color_discrete_map=COULEURS,
                           labels=LIBELLES, title="Répartition des frais médicaux")
        fig.update_layout(yaxis_title="Nombre de clients", legend_title="")
        st.plotly_chart(fig)
        st.caption(
            f"Frais médians : {dollars_texte(df['expenses'].median())}. "
            f"10 % des clients dépassent {dollars_texte(df['expenses'].quantile(0.9))}."
        )

    with g2:
        fig = px.box(df, x="statut", y="expenses", color="statut", color_discrete_map=COULEURS,
                     labels=LIBELLES, title="Frais selon le statut fumeur")
        fig.update_layout(xaxis_title="", showlegend=False)
        st.plotly_chart(fig)
        st.caption(
            f"Frais médians : {dollars_texte(df[df['smoker'] == 'no']['expenses'].median())} pour les non-fumeurs, "
            f"{dollars_texte(df[df['smoker'] == 'yes']['expenses'].median())} pour les fumeurs."
        )

    st.info("➡️ Étape suivante : les frais selon l'IMC et l'âge.")


# =============================================================
# ÉTAPE 2 : LES FACTEURS DE RISQUE
# =============================================================
elif etape == "2. Les facteurs de risque":
    st.title("Les frais selon l'IMC et l'âge")

    choix_regions = st.multiselect(
        "Filtrer par région", sorted(df["region"].unique()), default=sorted(df["region"].unique())
    )
    df_filtre = df[df["region"].isin(choix_regions)]
    fumeurs = df_filtre[df_filtre["smoker"] == "yes"]

    g1, g2 = st.columns(2)

    with g1:
        # hover_data : informations affichées quand on passe la souris sur un point
        fig = px.scatter(df_filtre, x="bmi", y="expenses", color="statut", color_discrete_map=COULEURS,
                         hover_data=["age", "children", "region"], labels=LIBELLES,
                         opacity=0.7, title="Frais selon l'IMC")
        fig.add_vline(x=30, line_dash="dash", line_color="grey")  # IMC 30 = seuil de l'obésité (OMS)
        fig.update_layout(legend_title="")
        st.plotly_chart(fig)
        if len(fumeurs) > 0:
            st.caption(
                f"Fumeurs avec un IMC ≥ 30 : {dollars_texte(fumeurs[fumeurs['bmi'] >= 30]['expenses'].mean())} en moyenne. "
                f"Fumeurs avec un IMC < 30 : {dollars_texte(fumeurs[fumeurs['bmi'] < 30]['expenses'].mean())}."
            )

    with g2:
        fig = px.scatter(df_filtre, x="age", y="expenses", color="statut", color_discrete_map=COULEURS,
                         hover_data=["bmi", "children", "region"], labels=LIBELLES,
                         opacity=0.7, title="Frais selon l'âge")
        fig.update_layout(legend_title="")
        st.plotly_chart(fig)
        st.caption(
            f"Frais moyens des 18-29 ans : {dollars_texte(df_filtre[df_filtre['age'] < 30]['expenses'].mean())}. "
            f"Des 50-64 ans : {dollars_texte(df_filtre[df_filtre['age'] >= 50]['expenses'].mean())}."
        )

    st.info("➡️ Étape suivante : le choix du modèle.")


# =============================================================
# ÉTAPE 3 : LE CHOIX DU MODÈLE
# =============================================================
elif etape == "3. Le choix du modèle":
    st.title(f"{meilleur['Modèle']} : R² = {meilleur['R²']:.3f}".replace(".", ","))
    st.write("5 modèles comparés sur le même jeu de test (20 % des clients, jamais vus pendant l'entraînement).")

    st.dataframe(resultats, hide_index=True)
    st.write(
        f"**Modèle retenu : {meilleur['Modèle']}**, avec le R² le plus élevé et la MAE la plus faible "
        f"({dollars_texte(meilleur['MAE ($)'])})."
    )

    g1, g2 = st.columns(2)

    with g1:
        tri = resultats.sort_values("R²")
        tri["Retenu"] = tri["Modèle"] == meilleur["Modèle"]
        fig = px.bar(tri, x="R²", y="Modèle", orientation="h", color="Retenu",
                     color_discrete_map={True: "#1F6F8B", False: "#C9D6DF"},
                     hover_data=["MAE ($)", "MSE"], title="R² par modèle (plus haut = meilleur)")
        fig.update_layout(xaxis_range=[0.7, 1], showlegend=False, yaxis_title="")
        st.plotly_chart(fig)

    with g2:
        noms = {"age": "Âge", "bmi": "IMC", "children": "Enfants", "sex_male": "Sexe",
                "smoker_yes": "Fumeur", "region_northwest": "Région NO",
                "region_southeast": "Région SE", "region_southwest": "Région SO"}
        importances = pd.Series(modele.feature_importances_, index=[noms[c] for c in colonnes]).sort_values()
        fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                     labels={"x": "Importance", "y": ""}, title="Importance des variables (Random Forest)")
        fig.update_traces(marker_color="#1F6F8B")
        st.plotly_chart(fig)
        top3 = importances.sort_values(ascending=False).head(3)
        st.caption(" · ".join(f"{nom} : {valeur:.0%}" for nom, valeur in top3.items()))

    st.info("➡️ Étape suivante : le simulateur.")


# =============================================================
# ÉTAPE 4 : LE SIMULATEUR
# =============================================================
else:
    st.title("Simulez les frais d'un client")
    st.write("Renseignez le profil, puis découvrez l'estimation et comment elle évoluerait.")

    # st.form : le calcul se lance seulement quand on clique sur le bouton
    with st.form("formulaire_client"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Âge", min_value=18, max_value=64, value=35)
            sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        with col2:
            imc = st.number_input("IMC", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
            enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=5, value=0)
        with col3:
            fumeur = st.radio("Fumeur ?", ["Non", "Oui"], horizontal=True)
            region = st.selectbox("Région", ["Nord-Est", "Nord-Ouest", "Sud-Est", "Sud-Ouest"])

        valider = st.form_submit_button("Estimer les frais", type="primary")

    if valider:
        # Traduction des choix du formulaire vers les valeurs du dataset
        sex = "male" if sexe == "Homme" else "female"
        smoker = "yes" if fumeur == "Oui" else "no"
        reg = {"Nord-Est": "northeast", "Nord-Ouest": "northwest",
               "Sud-Est": "southeast", "Sud-Ouest": "southwest"}[region]

        frais = predire_frais(age, sex, imc, enfants, smoker, reg)

        # Part des clients du dataset dont les frais sont inférieurs à l'estimation
        position = (df["expenses"] < frais).mean()

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Frais médicaux estimés", dollars(frais) + " / an")
            st.write(f"Plus élevés que ceux de **{position:.0%}** des clients du dataset.")

        with r2:
            # Où se situe ce client par rapport à tous les autres ?
            fig = px.histogram(df, x="expenses", nbins=30, labels=LIBELLES,
                               title="Position du client parmi tous les clients")
            fig.update_traces(marker_color="#C9D6DF")
            fig.add_vline(x=frais, line_width=3, line_color="#1F6F8B",
                          annotation_text="ce client", annotation_font_color="#1F6F8B")
            fig.update_layout(yaxis_title="Nombre de clients", height=320)
            st.plotly_chart(fig)

        # ---------- Et si... ? ----------
        st.subheader("Et si… ?")
        st.write("Nouvelle estimation du modèle si un seul élément du profil change :")

        scenarios = []
        if smoker == "yes":
            scenarios.append(("Non-fumeur", predire_frais(age, sex, imc, enfants, "no", reg)))
        if imc >= 25:
            scenarios.append(("IMC à 24,9", predire_frais(age, sex, 24.9, enfants, smoker, reg)))
        scenarios.append(("10 ans de plus", predire_frais(min(age + 10, 64), sex, imc, enfants, smoker, reg)))

        colonnes_scenarios = st.columns(len(scenarios))
        for col, (titre, montant) in zip(colonnes_scenarios, scenarios):
            ecart = montant - frais
            signe = "+" if ecart >= 0 else "-"
            # delta_color="off" : écart affiché en gris (pas de rouge / vert)
            col.metric(titre, dollars(montant), delta=f"{signe}{dollars(abs(ecart))}", delta_color="off")

        st.caption(f"Erreur moyenne du modèle sur le jeu de test (MAE) : {dollars_texte(meilleur['MAE ($)'])}.")

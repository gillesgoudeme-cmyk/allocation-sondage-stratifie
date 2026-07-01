from __future__ import annotations

from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
except ImportError as exc:  # pragma: no cover - friendly runtime error
    raise SystemExit(
        "Une ou plusieurs dépendances Python manquent. Installe les paquets avec "
        "'pip install -r requirements.txt' avant de lancer l'application."
    ) from exc

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - friendly runtime error
    raise SystemExit(
        "La dépendance 'streamlit' est manquante. Installe les paquets avec "
        "'pip install -r requirements.txt', puis lance 'streamlit run app.py'."
    ) from exc


REQUIRED_COLUMNS = ["strate", "Nh", "S2_T", "S2_Y", "S2_Z"]
VAR_NAMES = ["T", "Y", "Z"]


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the uploaded table."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")

    cleaned = df[REQUIRED_COLUMNS].copy()
    cleaned["strate"] = cleaned["strate"].astype(str).str.strip()

    for col in ["Nh", "S2_T", "S2_Y", "S2_Z"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if cleaned[["Nh", "S2_T", "S2_Y", "S2_Z"]].isna().any().any():
        raise ValueError("Le fichier contient des valeurs non numériques ou manquantes.")

    if (cleaned["Nh"] <= 0).any():
        raise ValueError("Toutes les valeurs Nh doivent être strictement positives.")

    if (cleaned[["S2_T", "S2_Y", "S2_Z"]] < 0).any().any():
        raise ValueError("Les variances S2_T, S2_Y et S2_Z doivent être positives ou nulles.")

    if cleaned["strate"].eq("").any():
        raise ValueError("La colonne 'strate' ne doit pas contenir de valeurs vides.")

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def variance_components(df: pd.DataFrame, n: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-stratum contributions for each target variable."""
    N = float(df["Nh"].sum())
    Nh = df["Nh"].to_numpy(dtype=float)
    fh = n / Nh
    base = (Nh / N) ** 2
    components = {}
    for var in VAR_NAMES:
        s2 = df[f"S2_{var}"].to_numpy(dtype=float)
        components[var] = base * ((1.0 - fh) / n) * s2
    return components


def final_variances(df: pd.DataFrame, n: np.ndarray) -> dict[str, float]:
    comps = variance_components(df, n)
    return {var: float(np.sum(values)) for var, values in comps.items()}


def objective(x: np.ndarray) -> float:
    return float(np.sum(x))


def continuous_allocation(df: pd.DataFrame, alphas: dict[str, float]) -> np.ndarray:
    """Solve the continuous relaxation with SLSQP."""
    Nh = df["Nh"].to_numpy(dtype=float)
    bounds = [(1.0, float(nh)) for nh in Nh]
    x0 = np.maximum(1.0, np.minimum(Nh, np.ceil(np.sqrt(Nh))))

    N = float(Nh.sum())
    coeffs = {}
    consts = {}
    for var in VAR_NAMES:
        s2 = df[f"S2_{var}"].to_numpy(dtype=float)
        c = (Nh / N) ** 2 * s2
        coeffs[var] = c
        consts[var] = float(np.sum(c / Nh))

    def constraint_factory(var_name: str):
        c = coeffs[var_name]
        rhs = alphas[var_name] + consts[var_name]

        def _constraint(x: np.ndarray) -> float:
            return rhs - float(np.sum(c / x))

        return _constraint

    constraints = [{"type": "ineq", "fun": constraint_factory(var)} for var in VAR_NAMES]

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10, "disp": False},
    )

    if result.success and np.all(np.isfinite(result.x)):
        x = np.clip(result.x, 1.0, Nh)
    else:
        # Fallback to the lower bound if the solver struggles.
        x = np.ones_like(Nh, dtype=float)

    return x


def is_feasible(df: pd.DataFrame, n: np.ndarray, alphas: dict[str, float]) -> bool:
    vars_ = final_variances(df, n)
    return all(vars_[var] <= alphas[var] + 1e-12 for var in VAR_NAMES)


def greedy_integer_allocation(df: pd.DataFrame, alphas: dict[str, float], start: np.ndarray) -> np.ndarray:
    """Repair a continuous solution into a feasible integer one.

    The problem is nonlinear because of the reciprocal terms in the variance
    expression. SciPy's MILP solver only handles linear constraints, so we use a
    nonlinear relaxation plus a monotone integer repair.
    """
    Nh = df["Nh"].to_numpy(dtype=int)
    n = np.clip(np.ceil(start).astype(int), 1, Nh)

    # If the rounded solution is not feasible, increase sample sizes greedily
    # until the constraints are satisfied.
    while not is_feasible(df, n.astype(float), alphas):
        current = final_variances(df, n.astype(float))
        deficits = {var: max(0.0, current[var] - alphas[var]) for var in VAR_NAMES}
        violated = [var for var, deficit in deficits.items() if deficit > 0]
        if not violated:
            break

        best_idx = None
        best_score = -1.0
        comps = variance_components(df, n.astype(float))

        for idx in range(len(n)):
            if n[idx] >= Nh[idx]:
                continue
            # Marginal reduction in each variance if we increase n_h by 1.
            improvement = 0.0
            for var in violated:
                c_h = comps[var][idx]
                before = c_h
                after = (df["Nh"].iloc[idx] / df["Nh"].sum()) ** 2 * (
                    (1.0 - (n[idx] + 1) / Nh[idx]) / (n[idx] + 1)
                ) * df[f"S2_{var}"].iloc[idx]
                improvement += deficits[var] * max(0.0, before - after)

            if improvement > best_score:
                best_score = improvement
                best_idx = idx

        if best_idx is None:
            # As a last resort, fill up all remaining strata.
            n = Nh.copy()
            break

        n[best_idx] += 1

    # Local descent: try to remove redundant units while keeping feasibility.
    improved = True
    while improved:
        improved = False
        for idx in range(len(n)):
            if n[idx] <= 1:
                continue
            candidate = n.copy()
            candidate[idx] -= 1
            if is_feasible(df, candidate.astype(float), alphas):
                n = candidate
                improved = True

    return n


def solve_allocation(df: pd.DataFrame, alphas: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    continuous = continuous_allocation(df, alphas)
    n_opt = greedy_integer_allocation(df, alphas, continuous)

    result = df.copy()
    result["nh_optimal"] = n_opt.astype(int)
    result["fh"] = result["nh_optimal"] / result["Nh"]

    comps = variance_components(result, result["nh_optimal"].to_numpy(dtype=float))
    for var in VAR_NAMES:
        result[f"contrib_{var}"] = comps[var]

    variances = final_variances(result, result["nh_optimal"].to_numpy(dtype=float))
    return result, variances, n_opt


def export_to_excel(result_df: pd.DataFrame, variances: dict[str, float], alphas: dict[str, float]) -> BytesIO:
    buffer = BytesIO()
    summary = pd.DataFrame(
        {
            "indicateur": [
                "n_total",
                "Var(T)",
                "Var(Y)",
                "Var(Z)",
                "alpha_T",
                "alpha_Y",
                "alpha_Z",
            ],
            "valeur": [
                int(result_df["nh_optimal"].sum()),
                variances["T"],
                variances["Y"],
                variances["Z"],
                alphas["T"],
                alphas["Y"],
                alphas["Z"],
            ],
        }
    )

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="allocation")
        summary.to_excel(writer, index=False, sheet_name="summary")

    buffer.seek(0)
    return buffer


def make_bar_chart(labels: pd.Series, values: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels.astype(str), values.astype(float), color="#2E86AB")
    ax.set_title(title)
    ax.set_xlabel("Strate")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def make_comparison_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["Nh"], width, label="Nh", color="#8ECAE6")
    ax.bar(x + width / 2, df["nh_optimal"], width, label="nh optimal", color="#FB8500")
    ax.set_xticks(x)
    ax.set_xticklabels(df["strate"].astype(str), rotation=45, ha="right")
    ax.set_title("Comparaison Nh vs nh optimal")
    ax.set_ylabel("Effectif")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def render_stat_card(container, title_html: str, value: str):
    container.markdown(
        f"""
        <div style="
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            min-height: 138px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.6rem;
        ">
            <div style="font-size: 1.2rem; font-weight: 700; line-height: 1.1; color: rgba(255,255,255,0.95); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {title_html}
            </div>
            <div style="font-size: 2.15rem; font-weight: 700; line-height: 0.95; color: rgba(255,255,255,0.98); white-space: nowrap;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strate": [1, 2, 3],
            "Nh": [500, 800, 300],
            "S2_T": [12.5, 15.1, 10.2],
            "S2_Y": [8.3, 7.9, 9.1],
            "S2_Z": [5.2, 6.4, 4.8],
        }
    )


st.set_page_config(
    page_title="Allocation optimale en sondage stratifié",
    page_icon="📊",
    layout="wide",
)

st.title("Allocation optimale en sondage stratifié")
st.markdown(
    r"**Minimisation de $\sum_h n_h$ sous contraintes de variance pour $T$, $Y$ et $Z$.**"
)
st.caption("Réalisé par GOUDEME Hamade et DATONDJI Mario.")

with st.expander("Principe du code", expanded=False):
    st.markdown(
        """
        Cette application suit les étapes suivantes :

        1. elle lit les données des strates à partir d’un fichier Excel ou de l’exemple intégré ;
        2. elle vérifie que les colonnes attendues sont bien présentes et que les valeurs sont valides ;
        3. elle calcule une allocation initiale en essayant de minimiser la taille totale de l’échantillon ;
        4. elle ajuste ensuite les tailles `n_h` pour respecter les contraintes de variance ;
        5. elle affiche les résultats, les graphiques, puis permet d’exporter le tout en Excel.

        En résumé, le code cherche le plus petit échantillon possible tout en gardant les variances sous les seuils demandés.
        """
    )

with st.sidebar:
    st.header("Paramètres")
    st.markdown(r"$\alpha_T$")
    alpha_T = st.number_input(
        "alpha_T",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.4f",
        label_visibility="collapsed",
    )
    st.markdown(r"$\alpha_Y$")
    alpha_Y = st.number_input(
        "alpha_Y",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.4f",
        label_visibility="collapsed",
    )
    st.markdown(r"$\alpha_Z$")
    alpha_Z = st.number_input(
        "alpha_Z",
        min_value=0.0,
        value=10.0,
        step=0.1,
        format="%.4f",
        label_visibility="collapsed",
    )

    st.divider()
    st.write("L'objectif est de minimiser la taille totale de l'échantillon tout en respectant les seuils de variance.")
    use_sample = st.checkbox("Charger l'exemple intégré", value=True)

uploaded = None if use_sample else st.file_uploader("Importer un fichier Excel (.xlsx)", type=["xlsx"])

if use_sample:
    input_df = sample_data()
    st.info("Exemple intégré chargé. Décoche la case pour importer ton propre fichier Excel.")
elif uploaded is not None:
    try:
        input_df = pd.read_excel(uploaded)
    except Exception as exc:  # pragma: no cover - Streamlit feedback
        st.error(f"Impossible de lire le fichier Excel: {exc}")
        st.stop()
else:
    st.warning("Charge un fichier Excel ou active l'exemple intégré pour commencer.")
    st.stop()

st.subheader("Données brutes")
st.caption("Tu peux modifier manuellement les valeurs ci-dessous avant de lancer le calcul.")
st.markdown("**Notations utilisées**")
st.latex(r"N_h,\; n_h,\; f_h,\; S_h^2,\; \alpha_T,\; \alpha_Y,\; \alpha_Z")
manual_edit = st.checkbox("Modifier les données manuellement", value=True)
if manual_edit:
    editor_df = input_df.copy()
    editor_df["strate"] = editor_df["strate"].astype(str)
    input_df = st.data_editor(
        editor_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "strate": st.column_config.TextColumn("strate"),
            "Nh": st.column_config.NumberColumn("Nₕ", min_value=1, step=1),
            "S2_T": st.column_config.NumberColumn("S²_T", min_value=0.0, step=0.1),
            "S2_Y": st.column_config.NumberColumn("S²_Y", min_value=0.0, step=0.1),
            "S2_Z": st.column_config.NumberColumn("S²_Z", min_value=0.0, step=0.1),
        },
        key="input_editor",
    )

try:
    cleaned_df = validate_dataframe(input_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Données d'entrée")
pretty_cleaned_df = cleaned_df.rename(
    columns={
        "Nh": "Nₕ",
        "S2_T": "S²_T",
        "S2_Y": "S²_Y",
        "S2_Z": "S²_Z",
    }
)
st.dataframe(pretty_cleaned_df, use_container_width=True)

with st.expander("Aperçu des formules"):
    st.markdown("**Objectif**")
    st.latex(r"\min \; n = \sum_h n_h")

    st.markdown("**Contraintes de variance**")
    st.latex(r"V(\bar t_s) \leq \alpha_T")
    st.latex(r"V(\bar y_s) \leq \alpha_Y")
    st.latex(r"V(\bar z_s) \leq \alpha_Z")

    st.markdown("**Formule générale**")
    st.latex(
        r"V(\bar x_s) = \sum_h \left(\frac{N_h}{N}\right)^2 \left(\frac{1-f_h}{n_h}\right) S_h^2(X)"
    )
    st.latex(r"f_h = \frac{n_h}{N_h}")
    st.latex(r"N = \sum_h N_h")
    st.latex(r"n = \sum_h n_h")

    st.markdown("**Remarque**")
    st.write(
        "La présence de `1 / n_h` rend le problème non linéaire. "
        "L'application utilise donc une relaxation continue puis une réparation entière."
    )

if st.button("Calculer l'allocation optimale", type="primary"):
    alphas = {"T": float(alpha_T), "Y": float(alpha_Y), "Z": float(alpha_Z)}

    try:
        result_df, variances, n_opt = solve_allocation(cleaned_df, alphas)
    except Exception as exc:  # pragma: no cover - Streamlit feedback
        st.error(f"Erreur pendant l'optimisation: {exc}")
        st.stop()

    st.success("Allocation calculée avec succès.")

    total_n = int(result_df["nh_optimal"].sum())
    col1, col2, col3, col4 = st.columns(4)
    render_stat_card(col1, "Taille totale optimale", str(total_n))
    render_stat_card(col2, "V( t̄<sub>s</sub> )", f"{variances['T']:.6f}")
    render_stat_card(col3, "V( ȳ<sub>s</sub> )", f"{variances['Y']:.6f}")
    render_stat_card(col4, "V( z̄<sub>s</sub> )", f"{variances['Z']:.6f}")

    st.subheader("Résultats par strate")
    pretty_result_df = result_df[["strate", "Nh", "nh_optimal", "fh", "contrib_T", "contrib_Y", "contrib_Z"]].rename(
        columns={
            "Nh": "Nₕ",
            "nh_optimal": "nₕ*",
            "fh": "fₕ",
            "contrib_T": "cₕ(T)",
            "contrib_Y": "cₕ(Y)",
            "contrib_Z": "cₕ(Z)",
        }
    )
    st.dataframe(pretty_result_df, use_container_width=True)

    st.subheader("Visualisations")
    fig1 = make_bar_chart(result_df["strate"], result_df["Nh"], "Barres des Nh", "Nh")
    fig2 = make_bar_chart(result_df["strate"], result_df["nh_optimal"], "Barres des nh optimaux", "nh optimal")
    fig3 = make_comparison_chart(result_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(fig1, clear_figure=True)
    with c2:
        st.pyplot(fig2, clear_figure=True)
    with c3:
        st.pyplot(fig3, clear_figure=True)

    st.subheader("Exportation")
    export_buffer = export_to_excel(result_df, variances, alphas)
    st.download_button(
        label="Télécharger les résultats Excel",
        data=export_buffer,
        file_name="allocation_optimale_sondage_stratifie.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption(
        "Le fichier exporté contient les feuilles 'allocation' et 'summary' avec les tailles optimales, "
        "les fractions de sondage et les contributions à la variance."
    )

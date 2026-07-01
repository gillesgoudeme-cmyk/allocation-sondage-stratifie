# Allocation optimale en sondage stratifié

Application web Streamlit en Python pour calculer une allocation optimale par strate sous contraintes de variance.

## Principe

L'application résout le problème:

```text
min Σ nh
sous contraintes:
Var(T) ≤ αT
Var(Y) ≤ αY
Var(Z) ≤ αZ
```

avec:

```text
Var(X) = Σ[(Nh/N)^2 × ((1-fh)/nh) × S²h(X)]
fh = nh / Nh
```

Les `nh` doivent être des entiers positifs.

## Fichiers attendus

Le fichier Excel importé doit contenir au minimum les colonnes:

- `strate`
- `Nh`
- `S2_T`
- `S2_Y`
- `S2_Z`

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Fonctionnement

L'application:

1. lit le fichier Excel,
2. valide les colonnes et les valeurs,
3. calcule une solution continue avec `scipy.optimize.minimize`,
4. convertit la solution en entiers positifs,
5. vérifie les contraintes de variance,
6. affiche les résultats et permet l'export Excel.

## Export

Le fichier exporté contient deux feuilles:

- `allocation`
- `summary`

La feuille `allocation` inclut:

- `strate`
- `Nh`
- `nh_optimal`
- `fh`
- `contrib_T`
- `contrib_Y`
- `contrib_Z`

## Remarque importante

Le problème contient des termes en `1/nh`, donc il est non linéaire. `scipy.optimize.milp` ne peut pas le traiter directement sans linéarisation supplémentaire. L'application utilise donc une relaxation continue suivie d'une réparation entière monotone pour produire des `nh` entiers et positifs.


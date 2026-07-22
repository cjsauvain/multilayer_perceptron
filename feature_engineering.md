Mandatory:
    - standardization (why?)

Optional:
    - Feature Selection — ta sélection par corrélation (pertinence au diagnostic + élagage des redondances), en cours de construction. Utile pour réduire la variance, mais pas indispensable pour un MLP (contrairement à un modèle linéaire).
    - Mutual Information — complément à Pearson pour juger la pertinence des features, capte le non-linéaire. On avait vu qu'à la main, c'est faisable pour des features discrétisées (binning + numpy), mais plus complexe pour du continu rigoureux (méthode k-NN, plutôt via sklearn si autorisé — point qu'on n'a pas pu vérifier faute d'accès au sujet).
    - Robust Scaling / Log Transform / Box-Cox — alternatives ou compléments à la standardisation si tu observes beaucoup d'outliers ou des distributions très asymétriques dans le WDBC.
    - PCA — utile pour visualiser la séparabilité (tu l'as déjà fait), pas nécessaire pour l'entraînement lui-même.
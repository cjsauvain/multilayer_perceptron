# TODO

Les bonus du sujet ne sont notés que si la partie obligatoire est complète et sans dysfonctionnement. Early stopping, precision / recall / F1, historique des métriques et courbes train + validation sont déjà en place.

## Obligatoire

- [ ] **Binary cross-entropy à la prédiction.** Le sujet demande que `predict` évalue le jeu avec la BCE. Aujourd'hui `src/predict.py` n'écrit que les labels, et l'accuracy seulement avec `--display-accuracy`. Afficher la BCE dès que la colonne `diagnosis` est complète, avec la fonction de `src/model/loss.py`.
- [ ] **Le résumé de fin d'entraînement doit décrire le modèle sauvé.** `src/model/network.py` restaure les poids de la meilleure validation loss, puis `src/train.py` imprime `acc_train[-1]` et `acc_valid[-1]`, c'est-à-dire le dernier epoch. Avec l'early stopping (patience 5 par défaut), ces chiffres ne sont pas ceux de `predict` sur le fichier de poids. Mémoriser l'epoch du meilleur loss et afficher ses métriques après la restauration.
- [ ] **Mesurer le train et la validation sur les mêmes poids.** Dans `Network.fit`, la loss train réutilise la prédiction d'avant la mise à jour des poids, la loss validation est calculée après. Recalculer les deux après le pas de gradient.
- [ ] **Architecture par défaut à deux couches cachées, présente dans le dépôt.** Le sujet l'exige. `model.json` est dans `.gitignore` et le schéma (`minItems: 1`) n'applique pas son `default` `[24, 24, 2]`. Versionner un exemple (`model.example.json`, deux couches cachées + sortie à 2 neurones) et le citer dans le README. L'entraînement réécrit `model.json` avec les stats de normalisation : ce fichier-là peut rester ignoré.

## Bonus

- [ ] **Adam**, en plus de la descente de gradient actuelle, qui reste le défaut.
  - Flag `--optimizer {gradient-descent,adam}` dans `src/parsing/mlp/train_parsing.py`. Défaut : `gradient-descent`.
  - État par couche (moment du gradient et moment du gradient au carré), mis à jour dans `src/model/layer.py` au lieu du seul `W -= lr * dW`.
  - Learning rate propre à Adam, autour de `0.001`. Le `0.1` actuel correspond à la descente de gradient pleine.
  - Documenter la commande dans le README.

## Propreté

- [ ] **`requirements.txt`** avec les versions utilisées (numpy, pandas, matplotlib, jsonschema, tqdm, argcomplete), à la place de la seule ligne `pip install` du README.
- [ ] **Glorot normal.** Dans `src/model/initializer.py`, l'écart-type vaut `sqrt(4 / (n_in + n_out))`. La formule de Glorot normal est `sqrt(2 / (n_in + n_out))`. He normal est déjà correct.
- [ ] **Shapes affichées.** `src/train.py` imprime `(features, samples)`, par exemple `(4, 454)`. L'exemple du sujet est `(samples, features)`. Afficher cette convention, le calcul interne en colonnes peut rester tel quel.
- [ ] **Log d'epoch.** Le `print` de `Network.fit` passe deux chaînes et produit un double espace (`epoch 1/70 -  loss:`).
- [ ] **Message du ratio de split.** `src/parsing/mlp/split_parsing.py` accepte `1.0` et annonce l'intervalle `]0, 1[`.
- [ ] **Chemin du schéma.** `src/parsing/mlp/schema_validation.py` ouvre `model.schema.json` depuis le répertoire courant. Le résoudre à partir de l'emplacement du projet, pour que la commande marche aussi lancée d'ailleurs.
- [ ] **Encodage des labels explicite.** `get_target` dans `src/preprocessing.py` s'appuie sur l'ordre alphabétique de `pandas.get_dummies` (`B` puis `M`). Poser les deux colonnes dans cet ordre, sans dépendre du tri.
- [ ] **Quelques tests numériques** : softmax qui somme à 1, BCE sur un cas connu, échelle de Glorot, et early stopping qui sauve bien l'epoch de meilleure validation loss.
- [ ] **README** une fois les points ci-dessus faits : BCE en prédiction, exemple de `model.json` versionné, option Adam et son learning rate.

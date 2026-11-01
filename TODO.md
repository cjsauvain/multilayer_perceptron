# TODO

## Bonus

- [ ] **Adam**, en plus de la descente de gradient actuelle, qui reste le défaut.
  - Flag `--optimizer {gradient-descent,adam}` dans `src/parsing/mlp/train_parsing.py`. Défaut : `gradient-descent`.
  - État par couche (moment du gradient et moment du gradient au carré), mis à jour dans `src/model/layer.py` au lieu du seul `W -= lr * dW`.
  - Learning rate propre à Adam, autour de `0.001`. Le `0.1` actuel correspond à la descente de gradient pleine.
  - Documenter la commande dans le README.

## Propreté

- [ ] **Quelques tests numériques** : softmax qui somme à 1, BCE sur un cas connu, échelle de Glorot, et early stopping qui sauve bien l'epoch de meilleure validation loss.
- [ ] **README** une fois les points ci-dessus faits : BCE en prédiction, exemple de `model.json` versionné, option Adam et son learning rate.

# TODO

Mis à jour après le cinquième passage.

**État : 17 points réglés, 1 partiel, 2 non commencés, aucun bug connu.**

Premier passage sans défaut trouvé : les trois sous-commandes fonctionnent,
`feature_analysis` s'importe, flake8 est à 0.

Ordre de traitement conseillé : **22 → 6 → 13 → mineur**

---

## Reste à faire

### 22. Aucun test — NON COMMENCÉ

C'est le point qui plafonne le projet. Rappel des quatre défauts des passages
précédents, tous corrigés, tous attrapables par un test trivial :

| Passage | Défaut | Test qui l'attrapait |
|---|---|---|
| 2 | `feature_analysis.py` : import cassé | smoke test d'import |
| 2 | `load_dataset` : une ligne perdue | comptage de lignes |
| 3 | `_restore_best_parameters` : `b = W` | shape des biais après `fit` |
| 4 | `predict.py` : call site orphelin | smoke test / intégration |

Aucun nouveau défaut ce passage-ci, mais le point 6 qui reste à faire est le
refactor le plus large du projet — c'est le moment d'avoir un filet.

**Correction :** créer `tests/` avec pytest. Par ordre de valeur :

1. **Smoke test d'import** — 3 lignes, attrape les passages 2 et 4 :

```python
@pytest.mark.parametrize("module", ["mlp", "feature_analysis"])
def test_entrypoints_import(module):
    importlib.import_module(module)
```

2. **Test d'intégration bout en bout** — attrape ce qu'un import ne voit pas :

```python
def test_full_pipeline(tmp_path):
    """split -> train (5 epochs) -> predict sur une petite fixture."""
    run("split-dataset", fixture_csv, ...)
    run("train", ..., "--epochs", "5")
    run("predict", ...)
    assert (tmp_path / "predictions.csv").exists()
```

   Lancer les sous-commandes via `subprocess` plutôt qu'en important les
   fonctions : c'est le seul moyen de couvrir aussi le parsing et le dispatch
   de `mlp.py`.

3. **Intégrité du réseau après `fit`** (non-régression du bug du passage 3) :

```python
def test_biases_keep_their_shape_after_fit():
    for layer in fitted_network.layers:
        assert layer.b.shape == (layer.n_neurons, 1)


def test_network_still_predicts_after_fit():
    assert fitted_network.predict(X_valid).shape == y_valid.shape
```

4. **`load_dataset` conserve toutes les lignes**, dans les deux formats
   (non-régression du point 14) :

```python
def test_load_dataset_keeps_every_row():
    raw = pd.read_csv("datasets/data.csv", header=None)
    assert len(load_dataset("datasets/data.csv")) == len(raw)
```

5. **Gradient checking** — le plus important sur le fond. Comparer les
   gradients analytiques de `DenseLayer.backward` à une approximation
   numérique `(J(W + eps) - J(W - eps)) / 2 eps` et vérifier une erreur
   relative `< 1e-7`. Utiliser `sigmoid` et non `relu` : la dérivée de ReLU
   est discontinue en 0 et fait échouer les différences finies.

   **À faire après le point 6** : le test devient beaucoup plus simple quand
   `backward` expose `layer.dW` sans appliquer la mise à jour.

6. **Tests unitaires** sur les fonctions pures :
   - `softmax` : colonnes qui somment à 1, pas d'overflow sur `Z = [1000, ...]`
   - `binary_cross_entropy` : pas de `nan` sur `y_pred = 0` (non-régression
     du clipping)
   - `z_score` / `min_max` : moyenne ≈ 0 et std ≈ 1 ; pas de `nan` sur une
     colonne constante (non-régression du point 17)
   - `z_score(df, means, stds)` : applique bien les stats passées et ne les
     recalcule pas — c'est le cœur de la cohérence train/predict
   - `get_target` : encodage stable même si une seule classe est présente
   - `--seed` : deux `split-dataset` avec la même graine donnent des fichiers
     identiques (verrouille le point 18)

### 6. Extraire un `Optimizer` hors de `DenseLayer` — NON COMMENCÉ

`src/model/layer.py:76-77` applique toujours `self.W -= learning_rate * dW`.
C'est le **seul** écart structurant qui reste, et le bonus le plus classique
du sujet.

**Correction :**

1. Dans `DenseLayer.backward`, **stocker** les gradients au lieu de les
   appliquer, et retirer `learning_rate` de la signature :

```python
def backward(self, dA, output_layer=False) -> np.ndarray:
    m = self.prev_A.shape[1]

    dZ = dA if output_layer else dA * self.activation_deriv(self.A)
    dA_prev = self.W.T.dot(dZ)

    self.dW = (1 / m) * dZ.dot(self.prev_A.T)
    self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    return dA_prev
```

2. Créer `src/model/optimizer.py` :

```python
class SGD:
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def update(self, layers):
        for layer in layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db


class Momentum:
    ...  # garde v_W / v_b par couche


class Adam:
    ...  # garde m/v par couche + compteur de pas t
```

3. `Network.__init__` reçoit l'optimiseur ; `Network.backward` appelle
   `self.optimizer.update(self.layers)` après la rétropropagation et perd son
   paramètre `learning_rate`, qui remonte dans l'optimiseur. `fit` perd lui
   aussi le sien.
4. Ajouter `--optimizer {sgd,momentum,adam}` dans `train_parsing.py` et le
   passer à `network_builder` — la factory existe déjà, il suffit d'un
   paramètre de plus.

Pour Adam/Momentum, initialiser l'état par couche à la première `update`.

Attention : ce refactor touche `layer.py`, `network.py`, `builder.py`,
`train.py` et `train_parsing.py` d'un coup — exactement le profil des call
sites cassés des passages 2 à 4, en plus large. Faire le point 22 **avant**.

### 13. Erreurs typées — PARTIEL

**Fait, et c'est la moitié qui comptait :** le `except Exception` fourre-tout
a disparu des deux entrypoints, remplacé par des `except` typés, et les
erreurs partent sur `stderr` avec un code de sortie 1. Un bug de
programmation remonte désormais avec son traceback complet au lieu d'être
déguisé en erreur utilisateur.

**a) Alignement des deux entrypoints — FAIT.** Le groupe
`(KeyError, ValueError)` a été ajouté à `mlp.py`, qui gère désormais les
mêmes exceptions que `feature_analysis.py`. Retesté :

| Cas | Sortie actuelle | |
|---|---|---|
| Fichier absent | `FileNotFoundError: ...` | ok |
| CSV sans les bonnes colonnes | `KeyError: ...` | ok |
| CSV vide | `EmptyDataError: No columns to parse from file` | ok |
| `params.npz` incompatible avec `model.json` | traceback brut de 14 lignes | à traiter en **c** |

Bon à savoir : `pd.errors.EmptyDataError` et `pd.errors.ParserError` héritent
tous deux de `ValueError`, ils sont donc déjà couverts — rien à ajouter à la
liste.

Reste une duplication : les deux entrypoints ont maintenant le **même** bloc
`try/except` de 7 lignes, mot pour mot. À extraire dans une fonction
partagée, par exemple `src/errors.py::run_safely(fn, args)`, pour qu'ils ne
puissent plus redivereger comme ils l'ont fait au passage précédent.

**b) `KeyboardInterrupt` n'est pas géré.** Un Ctrl-C pendant un `train` de
10000 epochs affiche un traceback au lieu de sortir proprement :

```python
except KeyboardInterrupt:
    print("\ninterrupted", file=sys.stderr)
    sys.exit(130)
```

**c) Les validations explicites manquent** — c'est le seul cas du tableau qui
sort encore en traceback brut. `IndexError: tuple index out of range`
n'indique absolument pas que le problème vient d'un couple `params.npz` /
`model.json` dépareillé, et `IndexError` n'hérite pas de `ValueError` : il ne
sera jamais rattrapé par la liste actuelle. C'est une validation qu'il faut,
pas un `except` de plus.

Ajouter dans `predict` :

- les features de `model.json` existent-elles dans le CSV ?
- `len(W) == len(model["layers"])` et `W[0].shape[1] == len(features)` ?

et dans `train` : dataset non vide, au moins 2 classes présentes.

Lever une exception maison `MLPError` (dans le même `src/errors.py`) avec un
message explicite, et l'ajouter aux `except` — ça distingue proprement les
erreurs métier des bugs.

### Mineur

- **`load_dataset` n'a pas d'annotation de retour** (`load_dataset.py:10`) :
  ajouter `-> pd.DataFrame`. C'est la seule fonction publique du projet à ne
  pas être typée.

---

## Réglé (conservé pour mémoire)

- **1. Dossiers mono-fichier** — aplatis en `src/train.py`, `src/predict.py`,
  `src/split.py`, `src/preprocessing.py`, `src/visualization.py`.
  `src/model/` et `src/parsing/` restent des packages justifiés.
- **4. `src/utils.py`** — remplacé par `src/constants.py` exposant
  `BASE_FEATURES` / `SUFFIXES` / `FEATURES` / `COLUMNS`, calculés une seule
  fois.
- **5. Nommage** — `parse_command_line` unifié, `weights_initializer`
  cohérent, champ mort `Z` retiré de `DenseLayer` et des activations,
  `cross_entropy` → `binary_cross_entropy`.
- **7. `fit` faisait trop** — `params_path` mort supprimé, `EarlyStopping`
  extraite en classe, tous les `print` sortis du modèle (`Network` expose
  `stopping_reason`, `train.py` affiche).
- **8. 5 forwards par epoch** — `_evaluate` accepte un `y_pred` optionnel et
  réutilise la passe d'entraînement pour la loss train. **5 → 2 forwards.**
- **9. Duplication** — `network_builder` supprime la construction dupliquée
  train/predict, `ft_accuracy` celle du calcul d'accuracy.
- **10. Encapsulation** — méthode rendue publique, renommée `load_parameters`
  (elle ne lit aucun fichier), et le call site de `predict.py:124` est à jour.
  Vérifié : `predict` renvoie 93.04 %.
- **11. Loss** — **annulé, pas un défaut.** Le sujet impose
  `binaryCrossEntropy` : l'implémentation binaire est conforme. Le clipping
  (`epsilon = 1e-15`) règle le risque de `log(0)` → `nan`, et le renommage en
  `binary_cross_entropy` lève l'ambiguïté du nom.
- **12. Early stopping** — `EarlyStopping` extraite en classe, `--patience`
  exposée (défaut 5, `positive_int`, sans défaut dans le constructeur pour
  forcer l'injection), `_restore_best_parameters` correcte, et
  `_save_best_parameters` renommée `_update_best_parameters` — initiative non
  suggérée qui lève la confusion avec `save_parameters_to_file`.
- **14. Chargement unifié** — mieux que ce qui était suggéré : le paramètre
  `header` **disparaît** de la signature ; `load_dataset` sonde le fichier
  avec `nrows=1` puis lit au bon format. Les 5 appelants se réduisent à
  `load_dataset(path)`. Vérifié : `predict` sur `data.csv` brut renvoie
  **569 prédictions pour 569 lignes** (568 avant).
- **15. `predict` exigeait les labels** — réglé avec un flag
  `--display-accuracy` explicite en plus du garde `"diagnosis" in df.columns`.
- **17. Divisions non protégées** — `safe_stds` et `safe_diff` en place, et
  les stats d'origine sont bien renvoyées telles quelles pour `model.json`.
- **18. Graine aléatoire** — complète. `seed` est un paramètre de
  `load_dataset`, propagé depuis `split.py` : les trois mélanges de
  `split-dataset` obéissent à `--seed`, et `train` applique le sien avant
  l'initialisation des poids.
- **A. Import cassé de `feature_analysis.py`** — corrigé.
- **C. `--seed` de `split-dataset`** — corrigé, y compris le texte d'aide.
- **D. flake8** — 0 erreur, maintenu sur cinq passages.
- **Mineurs des passages 3 et 4** — `load_dataset` sonde avec `nrows=1` au
  lieu de relire le fichier entier ; le `reversed(params.files)` sans effet
  a été retiré de `_retrieve_params`.

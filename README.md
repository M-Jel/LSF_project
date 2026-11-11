# Reconnaissance de signes isolés en temps réel (ASL, puis LSF)

> Projet en cours. Ce système **reconnaît des signes isolés** et **propose** une phrase.
> Il ne traduit pas la langue des signes et ne remplace en aucun cas un interprète.

## Architecture (cascade)

1. **Palier 1 — classifieur de signes isolés** : points clés MediaPipe → Transformer léger → top-5 avec probabilités. C'est la seule partie entraînée.
2. **Palier 2 — mode « signe par signe »** : découpage aux pauses (mains absentes ou immobiles), puis classification de chaque segment.
3. **Palier 3 — proposition de phrase** : un LLM reçoit la suite des top-5 et propose une phrase. Les gloses restent toujours affichées.

## Données

- Phase 1 : Google ISLR (Kaggle « asl-signs », ASL, 250 signes, 21 signeurs). Entraînement sur Kaggle.
- Phase 2 : LSF, sous réserve d'un accès légitime aux corpus.
- **Aucune donnée ni vidéo n'est incluse dans ce dépôt.**

## Évaluation

Séparation par signeur (aucun signeur commun entre l'entraînement et le test), top-1 / top-5, matrice de confusion, analyse des erreurs, latence de bout en bout mesurée sur CPU.

## Limites connues

Vocabulaire réduit, peu de signeurs, perte d'informations essentielles (expressions du visage, espace de signation, classificateurs).

## Structure

```
src/lsf/
  data/       islr.py (lecture parquet, cache .npz), split.py (separation par signeur), dataset.py
  features/   landmarks.py (131 points : 2 mains, haut du corps, levres, contour du visage ; normalisation)
              torch_features.py (positions + vitesses, augmentation affine)
  models/     baseline.py (statistiques temporelles + MLP), transformer.py (token CLS, padding masque)
  train.py    entrainement (seeds fixees, configuration sauvegardee)
  evaluate.py top-1, top-5, par signeur, matrice de confusion, paires confondues, latence du modele
  demo/       extract.py (MediaPipe Tasks Holistic), segmenter.py (decoupage aux pauses), webcam.py
  llm/        prompt.py (construction du prompt, aucun appel reseau)
notebooks/    01 exploration ISLR, 02 entrainement et evaluation (Kaggle)
tests/        tests unitaires et test de bout en bout sur donnees synthetiques
```

## Installation

Python 3.12 :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; python -m pytest -q
```

## Utilisation

1. Sur Kaggle : `notebooks/01_exploration_islr.ipynb`, puis `notebooks/02_train_eval_islr.ipynb`.
2. En local, demo webcam avec un checkpoint entraine copie dans `models/` :

```powershell
$env:PYTHONPATH = "src"; python -m lsf.demo.webcam --ckpt models/best.pt
```

## Licence

Code sous licence MIT (voir `LICENSE`). Les poids entraînés suivront la licence du corpus utilisé.

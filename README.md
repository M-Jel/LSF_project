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

## Installation

Python 3.12 :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

## Licence

Code sous licence MIT (voir `LICENSE`). Les poids entraînés suivront la licence du corpus utilisé.

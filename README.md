# Predicting Student Performance

Group project for **CSC311 (Introduction to Machine Learning), University of Toronto, Summer 2025**.
Forked from [nataliatabjab/Predicting-Student-Performance](https://github.com/nataliatabjab/Predicting-Student-Performance).

**Task:** predict whether a student will answer a multiple-choice question correctly, using a
sparse student × question response matrix from the Eedi education platform. We implemented and
compared four families of models, then designed an extension of the strongest one.

## Results

Accuracy on the held-out validation and public test sets (binary correct / incorrect):

| Model | Idea | Validation | Test |
|---|---|---:|---:|
| kNN, user-based (k = 11) | Students who answered alike before will answer alike again | 0.690 | 0.685 |
| kNN, item-based (k = 21) | Similar questions get similar answers | 0.692 | 0.683 |
| **Item Response Theory (1PL)** | Logistic model of student ability θ vs. question difficulty β | **0.708** | **0.713** |
| Autoencoder (k = 50) | Compress each student's responses to a latent vector, reconstruct the missing ones | 0.688 | 0.691 |
| Bagged logistic regression (3 models) | Bootstrap ensemble on one-hot student and question features | 0.705 | 0.706 |

**Takeaways**
- The simplest structured model, IRT with one parameter per student and per question, beat the
  more flexible ones. With a matrix this sparse, a strong prior on the structure matters more than
  model capacity.
- kNN struggles with sparsity and cold starts: new students or questions have no overlap to
  compare against, and the all-pairs similarity search is expensive.
- Bagging improved stability more than raw accuracy (+0.3 to 0.4 points over single models).
- L2 regularization on the autoencoder only hurt accuracy, a sign the model was not overfitting.

## Extension: 2PL IRT with clustering priors

The 1PL model assumes every question is equally informative. We added a per-question
discrimination parameter α (2PL) and kept it from overfitting by shrinking each question's
parameters toward the mean of its cluster:

1. Build a feature matrix from question metadata (subjects).
2. Reduce it with PCA (about 90% of variance kept).
3. Group questions with k-means.
4. Fit the 2PL model with Gaussian priors centered on each cluster's mean (α, β), updating the
   cluster means as training goes.

| Experiment (lr = 0.001) | 1PL test | 2PL test |
|---|---:|---:|
| 200 iterations, full training set | 0.678 | **0.707** |
| 50 iterations, full training set | 0.596 | **0.706** |
| 50 iterations, 50% of training set | 0.546 | **0.692** |

The 2PL model loses far less accuracy when half the training data is removed, which supports
the hypothesis that the cluster priors act as regularization. The full derivations, plots and
limitations are in [final_report.pdf](final_report.pdf).

## Team contributions

- **Luis Calderon**: Item Response Theory model (derivation of the log-likelihood and gradients,
  implementation, analysis); formal description of the 2PL-with-clustering extension.
- **Natalia Tabja**: neural-network autoencoder; extension schematic and limitations.
- **Swetha Poneasan**: kNN experiments and bagging ensemble; extension comparison experiments.

## Repository layout

| File | Contents |
|---|---|
| `knn.py` | User- and item-based kNN imputation |
| `item_response.py` | 1PL IRT trained by gradient ascent on the log-likelihood |
| `neural_network.py` | Autoencoder in PyTorch with masked MSE loss |
| `ensemble.py` | Bagged logistic regression |
| `PartB.py` | 2PL IRT with PCA + k-means priors |
| `matrix_factorization.py`, `majority_vote.py` | Additional baselines (SVD / ALS, majority vote), not covered in the report |
| `utils.py` | Data loading and evaluation helpers |
| `data/` | Training, validation and test splits and metadata |

## Running

```bash
pip install numpy scipy scikit-learn torch matplotlib
python item_response.py      # or knn.py, neural_network.py, ensemble.py, PartB.py
```

Scripts expect to be run from the repository root (they read from `./data`).

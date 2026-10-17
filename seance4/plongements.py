# ============================================================
#  Séance 4 --- le moteur à plongements, phase 2
#  Il remplit le contrat de la séance 3, un token lu entre, une probabilité
#  par token suivant sort. La table de comptes est remplacée par deux petites
#  tables apprises, E et U. Ce moteur sera remplacé par le Transformer, qui
#  gardera E en entrée, U en sortie et le softmax.
#  DÉJÀ ÉCRIT, à lire sans le modifier (étape 3.1 du TP)
# ============================================================
import numpy as np


def softmax(z):
    """Transforme un vecteur de scores en distribution de probabilité, des
    nombres positifs de somme 1, le plus grand score gardant la plus grande
    probabilité."""
    exponentielles = np.exp(z)
    return exponentielles / exponentielles.sum()


def distribution(v, E, U, b, numero):
    """Phase 2. La distribution sur le token suivant quand le dernier token
    lu est v, softmax(U E[v] + b). La case numero[w] du résultat est la
    probabilité que w suive v."""
    return softmax(U @ E[numero[v]] + b)

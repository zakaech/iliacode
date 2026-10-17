# ============================================================
#  Séance 4 --- outil fourni, à ne pas modifier
#  apprendre : la phase 1 du moteur à plongements. Les deux tables E et U
#  partent du hasard, puis une descente de gradient les corrige pas à pas
#  pour que le moteur donne une forte probabilité au token réellement écrit.
#  La quantité réduite à chaque pas est la surprise moyenne de la séance 3.
# ============================================================
import numpy as np

DEBUT = "<s>"      # les deux tokens de bord de la séance 3
FIN_SEQ = "</s>"


def numeroter(tokens):
    """Le numéro de chaque token, celui de sa ligne dans les tables E et U.
    Les deux tokens de bord reçoivent le même numéro, le dernier. Dans E
    cette ligne est celle de <s> (lu, jamais prédit), dans U celle de </s>
    (prédit, jamais lu)."""
    numero = {token: i for i, token in enumerate(tokens)}
    numero[DEBUT] = numero[FIN_SEQ] = len(tokens)
    return numero


def compter_en_matrice(sequences, numero):
    """La table de la séance 3 écrite en matrice, C[i, j] = nombre de fois où
    le token numéro j suit le token numéro i."""
    taille = numero[DEBUT] + 1
    C = np.zeros((taille, taille))
    for tokens in sequences:
        bordee = [DEBUT] + list(tokens) + [FIN_SEQ]
        for v, w in zip(bordee, bordee[1:]):
            C[numero[v], numero[w]] += 1
    return C


def log_distributions(E, U, b):
    """Pour chaque token d'entrée, le logarithme de la distribution que le
    moteur donne sur le token suivant, softmax(U E[v] + b), ligne par ligne."""
    Z = E @ U.T + b
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def surprise_moyenne(C, E, U, b):
    """La surprise moyenne du moteur, en bits, sur les couples comptés dans C."""
    return -(C * log_distributions(E, U, b)).sum() / C.sum() / np.log(2)


def apprendre(sequences, tokens, d, pas, taux=0.05, graine=0, a_chaque_pas=None):
    """Phase 1. Renvoie les trois tableaux du moteur, E (une ligne de d
    nombres par token lu), U (une ligne de d nombres par token prédit) et
    b (un nombre par token prédit). La ligne d'un token est son numéro,
    donné par numeroter(tokens).
    a_chaque_pas, s'il est donné, est appelé après chaque pas avec
    (numéro du pas, E, U, b), ce qui permet de suivre l'apprentissage."""
    numero = numeroter(tokens)
    C = compter_en_matrice(sequences, numero)
    total = C.sum()
    vus = C.sum(axis=1, keepdims=True)                 # C(v), une valeur par ligne
    hasard = np.random.default_rng(graine)
    taille = len(tokens) + 1                           # les tokens et un token de bord
    E = hasard.normal(0, 0.1, (taille, d))             # départ au hasard
    U = hasard.normal(0, 0.1, (taille, d))
    b = np.zeros(taille)
    parametres = [E, U, b]
    # Adam, la variante de la descente de gradient qui entraîne les grands
    # modèles, chaque nombre reçoit un pas adapté à l'historique de sa pente
    m = [np.zeros_like(p) for p in parametres]
    s = [np.zeros_like(p) for p in parametres]
    for t in range(1, pas + 1):
        P = np.exp(log_distributions(E, U, b))         # ce que le moteur prédit
        ecart = (P * vus - C) / total                  # prédit moins observé
        pentes = [ecart @ U, ecart.T @ E, ecart.sum(axis=0)]
        for p, g, mi, si in zip(parametres, pentes, m, s):
            mi *= 0.9
            mi += 0.1 * g
            si *= 0.999
            si += 0.001 * g * g
            p -= taux * (mi / (1 - 0.9 ** t)) / (np.sqrt(si / (1 - 0.999 ** t)) + 1e-8)
        if a_chaque_pas is not None:
            a_chaque_pas(t, E, U, b)
    return E, U, b

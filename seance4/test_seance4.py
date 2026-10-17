# ============================================================
#  Séance 4 --- les tests des vecteurs, du moteur à plongements et de la recherche
#  Lancer : pytest -v          (tous)      pytest -v -k p1   (la partie 1)
#  Les tests nommés test_fourni_ vérifient les fonctions déjà écrites
# ============================================================
import math
import numpy as np
import pytest
from apprentissage import numeroter, apprendre
from recherche import cosinus, plus_proches, sac_de_tokens, vecteur_moyen, precision_a_1
from plongements import softmax, distribution

# Le mini-corpus de la séance 3, cinq lignes du dépôt découpées en mots
MINI = [["for", "ticket", "in", "tickets"],
        ["for", "ticket", "in", "tickets"],
        ["for", "user", "in", "users"],
        ["if", "ticket", "is", "None"],
        ["for", "ticket", "in", "queue"]]
# Ses dix mots, dans l'ordre de leur numéro
MOTS = ["for", "if", "ticket", "user", "in", "is", "tickets", "users", "queue", "None"]
NUMERO = {mot: i for i, mot in enumerate(MOTS)}

# La table des plongements apprise sur le mini-corpus (cours, slide 10),
# une ligne de deux nombres par mot, dans l'ordre de MOTS
TABLE = np.array([[-0.91, -2.56],      # for
                  [-1.86, -2.77],      # if
                  [-3.18, -0.41],      # ticket
                  [-3.11, -0.33],      # user
                  [2.01, -1.90],       # in
                  [-4.11, 2.96],       # is
                  [1.98, 1.69],        # tickets
                  [2.02, 1.70],        # users
                  [2.05, 1.73],        # queue
                  [2.11, 1.79]])       # None
LIGNES = {mot: TABLE[NUMERO[mot]] for mot in MOTS}


# --- Partie 1, comparer ---------------------------------------------------------
def test_p1_deux_numeros_n_ont_rien_en_commun():
    ticket, user = np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])     # deux vecteurs one-hot
    assert cosinus(ticket, user) == 0
    assert cosinus(ticket, ticket) == pytest.approx(1)


def test_p1_deux_lignes_de_la_table_de_la_seance_3():
    # les successeurs comptés en séance 3, colonnes in, is, ticket, </s>
    ticket, user, tickets = np.array([3, 1, 0, 0]), np.array([1, 0, 0, 0]), np.array([0, 0, 0, 2])
    assert cosinus(ticket, user) == pytest.approx(0.949, abs=0.001)    # 3 / racine de 10
    assert cosinus(ticket, tickets) == 0


def test_p1_le_cosinus_ignore_la_longueur():
    u = np.array([3.0, 1.0])
    assert cosinus(u, 10 * u) == pytest.approx(1)      # même direction
    assert cosinus(u, -u) == pytest.approx(-1)         # direction opposée
    assert cosinus(u, np.zeros(2)) == 0                # un vecteur nul ne ressemble à rien


# --- Partie 2, chercher les plus proches ----------------------------------------
def test_p2_les_voisins_de_ticket():
    autres = {mot: v for mot, v in LIGNES.items() if mot != "ticket"}
    voisins = plus_proches(LIGNES["ticket"], autres, 3)
    assert [mot for mot, _ in voisins] == ["user", "is", "if"]
    assert voisins[0][1] == pytest.approx(1.00, abs=0.005)


def test_p2_l_ordre_et_le_nombre():
    vecteurs = {"b": np.array([1.0, 0.0]), "a": np.array([2.0, 0.0]), "c": np.array([0.0, 1.0])}
    assert plus_proches(np.array([1.0, 0.0]), vecteurs, 3) == [("a", 1.0), ("b", 1.0), ("c", 0.0)]
    assert len(plus_proches(np.array([1.0, 0.0]), vecteurs, 2)) == 2
    assert len(plus_proches(np.array([1.0, 0.0]), vecteurs)) == 3      # cinq demandés, trois existent


# --- Déjà écrit, prédire ----------------------------------------------------------
def test_fourni_softmax_de_trois_scores():
    p = softmax(np.array([math.log(2), 0.0, 0.0]))     # des scores aux probabilités
    assert p == pytest.approx([0.5, 0.25, 0.25])
    assert softmax(np.zeros(4)) == pytest.approx([0.25] * 4)    # scores égaux, hasard


def test_fourni_softmax_rend_une_distribution():
    scores = np.array([2.0, 1.0, 0.2, -0.5])
    p = softmax(scores)
    assert p == pytest.approx([0.619, 0.228, 0.102, 0.051], abs=0.001)
    assert all(p > 0) and p.sum() == pytest.approx(1)
    assert list(np.argsort(p)) == list(np.argsort(scores))      # l'ordre des scores est gardé


def test_fourni_distribution_d_un_moteur_ecrit_a_la_main():
    # deux tokens a et b, et le token de bord, d = 2
    numero = numeroter(["a", "b"])
    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    U = np.array([[0.0, 0.0], [math.log(2), 0.0], [0.0, 0.0]])
    b = np.zeros(3)
    assert distribution("a", E, U, b, numero) == pytest.approx([0.25, 0.5, 0.25])
    assert distribution("b", E, U, b, numero) == pytest.approx([1 / 3] * 3)
    assert distribution("a", E, U, b, numero)[numero["b"]] == pytest.approx(0.5)


# --- Déjà écrit, apprendre -------------------------------------------------------
def test_fourni_l_apprentissage_rapproche_ticket_et_user():
    numero = numeroter(MOTS)
    E, U, b = apprendre(MINI, MOTS, 2, 80)
    assert cosinus(E[numero["ticket"]], E[numero["user"]]) > 0.95
    apres_user = distribution("user", E, U, b, numero)
    assert apres_user.sum() == pytest.approx(1)
    assert apres_user[numero["is"]] > 0.1              # le zéro du jamais vu a disparu
    E, U, b = apprendre(MINI, MOTS, 2, 500)            # trop de pas, le moteur récite
    assert distribution("user", E, U, b, numero)[numero["is"]] < 0.02


# --- Déjà écrit, écrire un texte en vecteur ---------------------------------------
def test_fourni_sac_de_tokens():
    sac = sac_de_tokens(["for", "ticket", "in", "tickets", "for"], NUMERO)
    assert len(sac) == 10
    assert sac[NUMERO["for"]] == 2 and sac[NUMERO["ticket"]] == 1 and sac[NUMERO["user"]] == 0
    assert sac.sum() == 5


def test_fourni_vecteur_moyen():
    moyen = vecteur_moyen(["ticket", "user"], TABLE, NUMERO)
    assert moyen == pytest.approx([-3.145, -0.37])
    assert vecteur_moyen(["in"], TABLE, NUMERO) == pytest.approx([2.01, -1.90])


def test_fourni_ce_que_chaque_ecriture_voit():
    a, b = ["for", "ticket"], ["if", "user"]           # aucun mot commun, même rôle
    assert cosinus(sac_de_tokens(a, NUMERO), sac_de_tokens(b, NUMERO)) == 0
    assert cosinus(vecteur_moyen(a, TABLE, NUMERO), vecteur_moyen(b, TABLE, NUMERO)) > 0.95
    endroit, envers = ["ticket", "in", "tickets"], ["tickets", "in", "ticket"]
    for ecrire in (lambda t: sac_de_tokens(t, NUMERO), lambda t: vecteur_moyen(t, TABLE, NUMERO)):
        assert cosinus(ecrire(endroit), ecrire(envers)) == pytest.approx(1)     # l'ordre est perdu


# --- Partie 4, mesurer la recherche ---------------------------------------------
INDEX = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.1]),
         "C": np.array([0.0, 1.0]), "D": np.array([0.1, 1.0])}


def test_p4_precision_a_1():
    assert precision_a_1(INDEX, [("A", "B"), ("C", "D")]) == 1
    assert precision_a_1(INDEX, [("A", "B"), ("C", "A")]) == 0.5
    assert precision_a_1(INDEX, [("A", "C")]) == 0


def test_p4_la_requete_est_exclue_de_la_recherche():
    index = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    assert precision_a_1(index, [("A", "B")]) == 1     # B est le seul autre ticket

# ============================================================
#  Séance 4 --- la brique durable, comparer des vecteurs et chercher
#  Ces fonctions ne savent rien de la façon dont les vecteurs ont été
#  obtenus. Elles comparent aujourd'hui les lignes de la table apprise en
#  séance, elles compareront demain celles du moteur adopté, sans qu'une
#  ligne change. Elles resserviront pour l'index des documents internes.
#  Fichier à compléter pendant la séance, trois fonctions à écrire,
#  cosinus, plus_proches et precision_a_1
# ============================================================
import numpy as np


# --- Partie 1, comparer --------------------------------------------------------
def cosinus(u, v):
    """Le cosinus de l'angle entre deux vecteurs, leur produit scalaire
    divisé par le produit de leurs longueurs. Un vecteur nul renvoie 0."""
    longueurs = np.linalg.norm(u) * np.linalg.norm(v)
    if longueurs == 0:
        return 0.0
    return float(np.dot(u, v) / longueurs)


# --- Partie 2, chercher les plus proches ---------------------------------------
def plus_proches(requete, vecteurs, combien=5):
    """vecteurs = dictionnaire {nom : vecteur}. Renvoie les noms dont le
    vecteur est le plus proche de requete, sous la forme d'une liste de
    couples (nom, cosinus), du plus proche au moins proche."""
    candidats = [(nom, cosinus(requete, v)) for nom, v in vecteurs.items()]
    candidats.sort(key=lambda couple: (-couple[1], couple[0]))
    return candidats[:combien]


# --- Partie 4, mesurer la recherche --------------------------------------------
def precision_a_1(vecteurs, paires):
    """paires = liste de couples (requête, attendu). Renvoie la proportion
    des requêtes dont le plus proche voisin, elle-même exclue, est le ticket
    attendu."""
    reussites = 0
    for requete, attendu in paires:
        autres = {nom: v for nom, v in vecteurs.items() if nom != requete}
        trouve = plus_proches(vecteurs[requete], autres, 1)[0][0]
        if trouve == attendu:
            reussites += 1
    return reussites / len(paires)


# ============================================================
#  DÉJÀ ÉCRIT, à lire sans le modifier (étape 4.1 du TP)
#  Les deux façons d'écrire un texte en un seul vecteur
# ============================================================
def sac_de_tokens(tokens, numero):
    """Le texte écrit comme vecteur de comptes, une case par token du
    vocabulaire, la case numero[t] compte les apparitions de t."""
    vecteur = np.zeros(len(numero))
    for t in tokens:
        vecteur[numero[t]] += 1
    return vecteur


def vecteur_moyen(tokens, E, numero):
    """Le texte écrit comme la moyenne des plongements de ses tokens."""
    return np.mean([E[numero[t]] for t in tokens], axis=0)

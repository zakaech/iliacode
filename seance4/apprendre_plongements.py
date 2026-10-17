# ============================================================
#  Séance 4 --- la phase 1 du moteur à plongements sur le dépôt, script fourni
#  Il réutilise le tokeniseur de la séance 2, le comptage, l'hygiène et la
#  perplexité de la séance 3, et vos deux fichiers de la séance 4.
#  Usage, depuis le dossier seance4 :   python apprendre_plongements.py
# ============================================================
import sys

import numpy as np

from depot import ICI, TOKENS, FIN, SEANCE2, lire_lignes, fichiers, lisible
from bigramme import compter, probas_du_texte                    # le moteur de la séance 3
from perplexite import perplexite                                # l'instrument de la séance 3
from apprentissage import DEBUT, FIN_SEQ, numeroter, apprendre, log_distributions
from recherche import plus_proches                               # votre brique de la séance 4

D = 50                                  # le nombre de coordonnées d'un plongement
PAS_MAXIMUM = 100
VALEURS_DE_K = [1, 0.1, 0.01, 0.001]
MOTS_A_REGARDER = ["True", "0", "if", "self", "the", "email"]
VU_AU_MOINS = 10                        # un token rarement vu a une ligne peu corrigée

numero = numeroter(TOKENS)
TAILLE_V = len(TOKENS) + 1              # les 2000 tokens et le token de fin

# Le script réutilise le comptage et la perplexité de la séance 3, qui doivent
# donc être terminés (14 tests au vert dans le dossier seance3).
try:
    compter([["a"]])
    perplexite([0.5])
except NotImplementedError:
    sys.exit("La séance 3 n'est pas terminée, bigramme.py et perplexite.py doivent "
             "être complétés (pytest doit afficher 14 passed dans le dossier seance3).")


def probas_du_moteur(sequences, E, U, b):
    """Les probabilités que le moteur à plongements donne aux tokens
    réellement écrits, le token de fin compris. C'est l'entrée de perplexite()."""
    P = np.exp(log_distributions(E, U, b))
    probas = []
    for tokens in sequences:
        bordee = [DEBUT] + tokens + [FIN_SEQ]
        for v, w in zip(bordee, bordee[1:]):
            probas.append(P[numero[v], numero[w]])
    return probas


# --- Les trois blocs du protocole de la séance 3 --------------------------------
tout = lire_lignes(fichiers("entrainement"))
train = [s for i, s in enumerate(tout) if i % 10 != 9]      # neuf lignes sur dix
dev = [s for i, s in enumerate(tout) if i % 10 == 9]        # une ligne sur dix, pour les réglages
test = {
    "code": lire_lignes(fichiers("test/code")),
    "français": lire_lignes([SEANCE2 / "donnees/test/tickets_fr.txt"]),
    "anglais": lire_lignes([SEANCE2 / "donnees/test/tickets_en.txt"]),
    "darija": lire_lignes([SEANCE2 / "donnees/test/tickets_darija.txt"]),
}

# --- Le moteur de la séance 3, pour la comparaison -------------------------------
paires, contextes = compter(train)
print(f"train : {len(train)} lignes, {sum(contextes.values())} tokens prédits, |V| = {TAILLE_V}")
k_retenu = min(VALEURS_DE_K,
               key=lambda k: perplexite(probas_du_texte(dev, paires, contextes, TAILLE_V, k)))
pp_comptage = perplexite(probas_du_texte(dev, paires, contextes, TAILLE_V, k_retenu))
print(f"\nmoteur de la séance 3, comptage lissé, k = {k_retenu}")
print(f"  nombres stockés : {TAILLE_V * TAILLE_V} cases, dont {len(paires)} non nulles")
print(f"  perplexité sur le dev : {pp_comptage:.1f}")

# --- Phase 1, apprendre E et U sur le bloc train ----------------------------------
print(f"\nmoteur à plongements, d = {D}")
print(f"  nombres stockés : {2 * TAILLE_V * D + TAILLE_V} (E, U et b)")
print("\npas       " + f"{'train':>10}{'dev':>10}")
meilleur = {"pp": float("inf")}


def suivre(pas, E, U, b):
    """Tous les dix pas, mesure le moteur et garde la meilleure version sur le dev."""
    if pas % 10:
        return
    pp_train = perplexite(probas_du_moteur(train, E, U, b))
    pp_dev = perplexite(probas_du_moteur(dev, E, U, b))
    print(f"{pas:<10}{pp_train:10.1f}{pp_dev:10.1f}")
    if pp_dev < meilleur["pp"]:
        meilleur.update(pp=pp_dev, pas=pas, E=E.copy(), U=U.copy(), b=b.copy())


apprendre(train, TOKENS, D, PAS_MAXIMUM, a_chaque_pas=suivre)
E, U, b = meilleur["E"], meilleur["U"], meilleur["b"]
print(f"\nnombre de pas retenu sur le dev : {meilleur['pas']}")

# --- Le chiffre final, une seule lecture du bloc test -----------------------------
print()
print("perplexité sur le test" + f"{'comptage':>12}{'plongements':>14}")
for nom, sequences in test.items():
    pp_c = perplexite(probas_du_texte(sequences, paires, contextes, TAILLE_V, k_retenu))
    pp_p = perplexite(probas_du_moteur(sequences, E, U, b))
    print(f"  {nom:<20}{pp_c:12.1f}{pp_p:14.1f}")

# --- L'artefact figé de la séance --------------------------------------------------
np.savez(ICI / "plongements.npz", E=E, U=U, b=b)
print(f"\ntables écrites dans plongements.npz, E compte {E.shape[0]} lignes de {E.shape[1]} nombres")

# --- Les voisins de quelques tokens dans la table E --------------------------------
# Seuls les tokens qui terminent un mot et que le train a montrés assez souvent
# sont comparés, la ligne d'un token presque jamais vu est restée proche du hasard.
lignes_fiables = {lisible(t): E[numero[t]] for t in TOKENS
                  if t.endswith(FIN) and contextes[t] >= VU_AU_MOINS}
print(f"\nplus proches voisins dans la table E, parmi les {len(lignes_fiables)} mots "
      f"vus au moins {VU_AU_MOINS} fois")
for mot in MOTS_A_REGARDER:
    if mot not in lignes_fiables:
        continue
    autres = {nom: v for nom, v in lignes_fiables.items() if nom != mot}
    voisins = ", ".join(f"{nom} {c:.2f}" for nom, c in plus_proches(lignes_fiables[mot], autres, 4))
    print(f"  {mot:<8}{voisins}")

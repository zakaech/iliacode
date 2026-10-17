# ============================================================
#  Séance 4 --- la recherche du ticket le plus proche, script fourni
#  Il indexe les 48 tickets du dossier donnees de deux façons, le sac de
#  tokens et la moyenne des plongements, puis mesure chaque index sur les
#  8 paires de doublons désignées à la main.
#  Usage, depuis le dossier seance4 :
#     python mesure_recherche.py                      la mesure sur les 8 paires
#     python mesure_recherche.py "texte d'une demande"   une recherche libre
# ============================================================
import sys
from pathlib import Path

import numpy as np

from depot import ICI, TOKENS, en_tokens
from recherche import plus_proches, sac_de_tokens, vecteur_moyen, precision_a_1

if not (ICI / "plongements.npz").exists():
    sys.exit("Introuvable : plongements.npz. Lancez d'abord python apprendre_plongements.py")
E = np.load(ICI / "plongements.npz")["E"]
numero = {token: i for i, token in enumerate(TOKENS)}     # le numéro de chaque token

# --- Les tickets et les paires de doublons ---------------------------------------
tickets, langue = {}, {}
for ligne in open(ICI / "donnees" / "tickets.tsv", encoding="utf-8").read().splitlines():
    identifiant, code_langue, texte = ligne.split("\t")
    tickets[identifiant], langue[identifiant] = texte, code_langue
paires = [tuple(ligne.split("\t"))
          for ligne in open(ICI / "donnees" / "paires_gold.tsv", encoding="utf-8").read().splitlines()]

# --- Phase 1, indexer, chaque ticket devient un vecteur, une fois pour toutes ----
INDEX = {
    "sac de tokens": {i: sac_de_tokens(en_tokens(t), numero) for i, t in tickets.items()},
    "plongements": {i: vecteur_moyen(en_tokens(t), E, numero) for i, t in tickets.items()},
}
ECRIRE = {
    "sac de tokens": lambda texte: sac_de_tokens(en_tokens(texte), numero),
    "plongements": lambda texte: vecteur_moyen(en_tokens(texte), E, numero),
}

# --- Phase 2, une recherche libre --------------------------------------------------
if len(sys.argv) > 1:
    demande = " ".join(sys.argv[1:])
    print("demande :", demande)
    for nom, index in INDEX.items():
        print(f"\nles trois tickets les plus proches, index {nom}")
        for identifiant, c in plus_proches(ECRIRE[nom](demande), index, 3):
            debut = tickets[identifiant][:56] + ("..." if len(tickets[identifiant]) > 56 else "")
            print(f"  {identifiant}  {c:.2f}  {debut}")
    sys.exit()

# --- La mesure sur les paires de doublons ------------------------------------------
print(f"{len(tickets)} tickets indexés, {len(paires)} paires de doublons")
for nom, index in INDEX.items():
    taille = len(next(iter(index.values())))
    print(f"\nindex {nom}, un vecteur de {taille} nombres par ticket")
    print("requête  attendu  langues   trouvé  cosinus  rang de l'attendu")
    for requete, attendu in paires:
        autres = {i: v for i, v in index.items() if i != requete}
        classement = plus_proches(index[requete], autres, len(autres))
        trouve, c = classement[0]
        rang = [i for i, _ in classement].index(attendu) + 1
        print(f"{requete:<9}{attendu:<9}{langue[requete] + '-' + langue[attendu]:<10}"
              f"{trouve:<8}{c:<9.2f}{rang}")
    reussites = round(precision_a_1(index, paires) * len(paires))
    print(f"précision à 1 : {reussites}/{len(paires)} = {precision_a_1(index, paires):.3f}")

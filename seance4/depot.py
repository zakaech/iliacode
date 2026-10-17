# ============================================================
#  Séance 4 --- outil fourni, à ne pas modifier
#  L'accès aux briques des séances précédentes, le tokeniseur figé de la
#  séance 2 (bpe.py, merges.json, donnees) et l'hygiène de la séance 3.
# ============================================================
import glob
import json
import sys
import warnings
from pathlib import Path

# Un motif d'expression régulière écrit dans un commentaire de bpe.py peut
# provoquer un SyntaxWarning sans conséquence, il est masqué ici.
warnings.filterwarnings("ignore", category=SyntaxWarning)

ICI = Path(__file__).resolve().parent
SEANCE2 = ICI.parent / "seance2"
SEANCE3 = ICI.parent / "seance3"
if not (SEANCE2 / "merges.json").exists():
    sys.exit(f"Introuvable : {SEANCE2 / 'merges.json'}. Le dossier seance4 doit être "
             "à côté du dossier seance2, et merges.json doit y avoir été créé.")
if not (SEANCE3 / "perplexite.py").exists():
    sys.exit(f"Introuvable : {SEANCE3 / 'perplexite.py'}. Le dossier seance4 doit être "
             "à côté du dossier seance3.")
sys.path.insert(0, str(SEANCE2))
sys.path.insert(0, str(SEANCE3))

from bpe import BASE, FIN, pretokeniser, encoder_mot, table_des_rangs    # séance 2
from hygiene import masquer_secrets                                      # séance 3

# --- Le tokeniseur figé de la séance 2 ----------------------------------------
FUSIONS = [tuple(p) for p in json.load(open(SEANCE2 / "merges.json", encoding="utf-8"))]
RANGS = table_des_rangs(FUSIONS)
# Les tokens du vocabulaire dans un ordre fixe, les 256 octets, le marqueur de
# fin de mot, puis les fusions dans l'ordre appris. La place d'un token dans
# cette liste est son numéro, donc sa ligne dans la table des plongements.
TOKENS = BASE + [FIN] + [a + b for a, b in FUSIONS]
_memoire = {}                          # un mot déjà encodé n'est pas réencodé

# L'hygiène passe avant tout apprentissage et toute indexation. Si la partie 6
# de la séance 3 n'est pas faite, les lignes sont lues telles quelles.
try:
    masquer_secrets("")
    nettoyer = masquer_secrets
except NotImplementedError:
    print("hygiène de la séance 3 pas encore écrite, les lignes sont lues telles quelles")
    nettoyer = lambda ligne: ligne


def en_tokens(ligne):
    """Une ligne de texte devient la liste de ses tokens BPE, blancs exclus."""
    tokens = []
    for mot in pretokeniser(nettoyer(ligne)):
        if mot.isspace():
            continue
        if mot not in _memoire:
            _memoire[mot] = encoder_mot(mot, RANGS)
        tokens.extend(_memoire[mot])
    return tokens


def lire_lignes(chemins):
    """Toutes les lignes non vides des fichiers, chacune devenue une séquence."""
    sequences = []
    for chemin in sorted(chemins):
        for ligne in open(chemin, encoding="utf-8").read().splitlines():
            if ligne.strip():
                sequences.append(en_tokens(ligne))
    return [s for s in sequences if s]


def fichiers(dossier):
    return glob.glob(str(SEANCE2 / "donnees" / dossier / "**" / "*.*"), recursive=True)


def lisible(token):
    """Un token affiché sans son marqueur de fin de mot."""
    return token.replace(FIN, "")

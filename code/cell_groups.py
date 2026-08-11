# cell_groups.py
from pathlib import Path
import pickle
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

av1a1 = [720575940623041549,720575940622894616,720575940626958878,720575940633984924,720575940611137742,720575940627192337]
TPN1 = [720575940623118029, 720575940624967561]
MN9 = [720575940660219265,720575940618238523]


with open(REPO_ROOT / 'data/sez_neurons.pickle','rb') as f:
    sez = pickle.load(f)

with open(REPO_ROOT / 'data/interneuron_group_info/target_ids_all_v783.pkl','rb') as f:
    target_ids_valid_all = pickle.load(f)
with open(REPO_ROOT / 'data/interneuron_group_info/g_info_all_v783.pkl','rb') as f:
    g_info_all = pickle.load(f)
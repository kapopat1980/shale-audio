import json
codec = open('shale/codec.py').read()
eclq = open('shale/eclq.py').read()
const = json.load(open('kaggle/kaggle_constants.json'))
body = open('kaggle/attack_dl_body.py').read()
cell1 = ("# Cell 1: write the SHALE coder and ECLQ modules (identical to the paper's implementation)\n"
         "import os\nos.makedirs('/kaggle/working/shale_mod', exist_ok=True)\n"
         f"open('/kaggle/working/shale_mod/codec.py','w').write({codec!r})\n"
         f"open('/kaggle/working/shale_mod/eclq.py','w').write({eclq!r})\nprint('modules written')\n")
cell2 = "# Cell 2: bucket tables designed on the public VCTK profile, and the in-domain design actors\nCONST = " + json.dumps(const) + "\n"
cells = [
 ("markdown", "# SHALE paper: deep-learning length side-channel attack\n\n**Before running:** Settings → Accelerator → **GPU T4 x2** (or P100). Add Input → search and add **CREMA-D** (e.g. `ejlok1/cremad`) and **Google Speech Commands** (e.g. `yashdogra/speech-commands` or `neehakurelli/google-speech-commands`).\n\nThen **Run All**. Expected time about 3–5 hours on a T4. Results are saved after every experiment to `/kaggle/working/shale_dl_results.json`. If the session stops, just run again: finished experiments are skipped. Download that JSON file at the end and send it back.\n\nFor a 5-minute smoke test first, set `QUICK = True` in Cell 3."),
 ("code", cell1), ("code", cell2), ("code", body)]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": src, **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells]}
json.dump(nb, open('kaggle/SHALE_deep_attack_kaggle.ipynb', 'w'), indent=1)
open('kaggle/SHALE_deep_attack_kaggle.py', 'w').write(cell1 + "\n" + cell2 + "\n" + body)
print('ok')

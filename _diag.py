import os, sys, importlib, json
from pathlib import Path

def snapshot():
    info = {}
    info["cwd"] = os.getcwd()
    info["python"] = sys.version
    files = ["app.py","model.pkl","requirements.txt","design-00f48579-0d70-4b7b-9c04-7017c5570fbd.png"]
    for p in files:
        info[f"exists:{p}"] = Path(p).exists()
    for m in ["streamlit","pandas","sklearn","joblib","numpy","scipy"]:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, "__version__", "n/a")
        except Exception as e:
            v = f"ERROR: {e}"
        info[f"version:{m}"] = v
    return info

def json_text():
    return json.dumps(snapshot(), indent=2, sort_keys=True)

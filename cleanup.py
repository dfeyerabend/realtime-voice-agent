"""Löscht alle .wav-Dateien aus recordings/ und outputs/."""
import glob
import os
import config

def clean():
    dirs = [config.RECORDING_DIR, config.OUTPUT_DIR]
    count = 0
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.wav")):
            os.remove(f)
            count += 1
    print(f"{count} Dateien gelöscht.")

if __name__ == "__main__":
    clean()
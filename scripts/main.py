import os
import subprocess
import sys

ROOT = (
    os.path.dirname(os.path.dirname(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")


def run_script(script_name):
    """
    Run a script from the scripts directory.
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"Running {script_path}...")
    subprocess.run([sys.executable, script_path], check=True)


def main():
    """
    Execute the full pipeline: preprocessing → training → testing.
    """
    scripts = ["preprocessing.py", "train.py", "test.py"]

    for script in scripts:
        run_script(script)

    print("All scripts executed!")


if __name__ == "__main__":
    main()

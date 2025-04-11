import os
import subprocess
import sys


def run_script(script_name):
    """
    Run a script file.
    """
    script_path = os.path.join("scripts", script_name)
    print(f"Running {script_path}...")
    subprocess.run([sys.executable, script_path], check=True)


def main():
    """
    Execute full pipeline.
    """
    scripts = ["preprocessing.py", "train.py", "test.py"]

    for script in scripts:
        run_script(script)

    print("All scripts executed!")


if __name__ == "__main__":
    main()

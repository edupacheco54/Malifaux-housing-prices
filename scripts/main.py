import os
import subprocess

def run_script(script_name):
    """
    Run a script file.
    """
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)

def main():
    """
    Execute full pipeline.
    """
    scripts = ["preprocessing.py", "train.py", "test.py"]

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"Error: {script} not found.")
    print("All scripts executed!")

if __name__ == "__main__":
    main()
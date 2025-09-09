import subprocess
import yaml

def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for script in cfg.get("pipelines", []):
        subprocess.run(["python", script], check=True)

if __name__ == "__main__":
    main()

import os

import yaml


def load_yaml(file_name: str) -> dict:
    with open(file_name, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None


if os.name == "posix":  # POSIX-compliant systems (Linux, macOS, etc.)
    INPUTDIR = r"C:/Users/denti/Weather_Data_For_Research/G1_weather/DATA/PSSE"
elif os.name == "nt":  # Windows
    INPUTDIR = f"C:/Users/denti/Weather_Data_For_Research/G1_weather/DATA/PSSE"
else:
    raise OSError("Unsupported operating system")

print(f"Input directory is set to: {INPUTDIR}")
HEADERS = load_yaml(f"{INPUTDIR}/AREAS/headers.yaml")
AREAS = load_yaml(f"{INPUTDIR}/AREAS/areas.yaml")

INTERNALS = AREAS["ISONE"] + AREAS["NYISO"] + AREAS["PJM"] + AREAS["DUKE"]
EXTERNALS = list(set(AREAS["ALLAREAS"]) - set(INTERNALS))

import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'config.toml'

with open(CONFIG_PATH, 'rb') as f:
  config = tomllib.load(f)
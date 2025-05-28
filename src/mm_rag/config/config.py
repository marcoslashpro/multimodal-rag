import tomllib

CONFIG_PATH = '/home/marco/Projects/mm-rag/src/mm_rag/config/config.toml'

with open(CONFIG_PATH, 'rb') as f:
  config = tomllib.load(f)
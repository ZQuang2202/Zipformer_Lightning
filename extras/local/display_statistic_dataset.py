import sys
from lhotse import load_manifest


manifest_path = sys.argv[1]
cuts = load_manifest(manifest_path)
cuts.describe()

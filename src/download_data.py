"""Recuperación opcional de la misma instantánea de datos incluida en el proyecto."""

from pathlib import Path
import hashlib
import json
import urllib.request


def main():
    """Descarga de los CSV del manifiesto, comprobados antes de guardarlos."""
    data = Path(__file__).resolve().parents[1] / "data"
    manifest = json.loads((data / "procedencia.json").read_text())
    for name, info in manifest["files"].items():
        content = urllib.request.urlopen(info["url"], timeout=30).read()
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError(
                f"El contenido descargado de {name} no coincide con la instantánea"
            )
        (data / "raw" / name).write_bytes(content)
        print(f"Verificado: {name}")


if __name__ == "__main__":
    main()

import hashlib
import json
import shutil
from pathlib import Path


def cache_key(video: Path, settings: dict) -> str:
    digest = hashlib.sha256()
    with video.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def restore_cache(cache_dir: Path, artifacts: dict[str, Path]) -> dict | None:
    metadata_path = cache_dir / "run.json"
    if not metadata_path.is_file():
        return None
    for name, destination in artifacts.items():
        source = cache_dir / name
        if not source.is_file():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def store_cache(
    cache_dir: Path, artifacts: dict[str, Path], metadata: dict
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, source in artifacts.items():
        shutil.copy2(source, cache_dir / name)
    (cache_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

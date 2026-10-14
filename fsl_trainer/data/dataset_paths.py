"""Named prepared datasets and compatibility with category-level datasets."""
from pathlib import Path


def dataset_name(value):
    if (not isinstance(value, str) or not value or value != value.strip()
            or value.startswith(".") or value.endswith(".")
            or any(c in value for c in '/\\:*?"<>|')
            or any(ord(c) < 32 for c in value)
            or value.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError("Use a single dataset folder name without path separators or reserved characters.")
    return value


def prepared_datasets(category_root):
    root = Path(category_root)
    if not root.is_dir():
        return []
    result = [root] if (root / "manifest.json").is_file() else []
    result.extend(p for p in sorted(root.iterdir())
                  if p.is_dir() and not p.name.startswith(".")
                  and p.resolve().parent == root.resolve()
                  and (p / "manifest.json").is_file())
    return result


def select_dataset(category_root, name=None):
    root = Path(category_root)
    if name == ".":
        return root if (root / "manifest.json").is_file() else None
    if name is not None:
        target = root / dataset_name(name)
        return target if target in prepared_datasets(root) else None
    choices = prepared_datasets(root)
    if len(choices) > 1:
        raise ValueError(f"{root.name}: multiple prepared datasets found. Choose a dataset name with --dataset-name "
                         "(use . for the original category-level dataset).")
    return choices[0] if choices else None

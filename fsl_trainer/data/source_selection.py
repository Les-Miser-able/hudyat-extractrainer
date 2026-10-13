"""Source discovery shared by the dashboard and preparation worker (stdlib only)."""
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SUPPORTED = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | {".npy"}


def folder_selection(value):
    """None selects all sources; a nonempty list selects immediate gesture subfolders."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("Select at least one source folder, or choose All source folders.")
    for name in value:
        if (not isinstance(name, str) or not name.strip() or name.startswith(".")
                or name.endswith((".", " ")) or any(c in name for c in '/\\:*?"<>|')
                or any(ord(c) < 32 for c in name)):
            raise ValueError("Source folders must be individual folder names, not paths.")
    return sorted(set(value))


def source_files(folder, source_folders=None, videos_only=False):
    folder = Path(folder)
    selected = folder_selection(source_folders)
    extensions = VIDEO_EXTENSIONS if videos_only else SUPPORTED
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in extensions
                  and not any(part.startswith(".") for part in p.relative_to(folder).parts)
                  and (selected is None or (len(p.relative_to(folder).parts) > 1
                       and p.relative_to(folder).parts[0] in selected)))


def add_arguments(parser):
    parser.add_argument("--source-folder", action="append", dest="source_folders",
                        help="Include this immediate subfolder of each gesture; repeat to select several")
    parser.add_argument("--videos-only", action="store_true", help="Include only supported video files")

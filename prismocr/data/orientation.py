"""EXIF orientation helpers so annotations and loaded pixels share one frame.

Some source corpora (for example KhmerST scene photos) store pixels in a
different orientation than the one a viewer shows and annotators used. Every
consumer must therefore agree on the *displayed* frame: the dataset loader
transposes pixels on load, and converters record the displayed width and height
so annotation validation and pixels cannot drift apart.
"""

from __future__ import annotations

from PIL import Image, ImageOps

# EXIF orientation values that swap the reported width and height.
_SWAPPED_AXES = frozenset({5, 6, 7, 8})
_ORIENTATION_TAG = 0x0112


def exif_orientation(image: Image.Image) -> int:
    """Return the EXIF orientation tag, defaulting to 1 (upright)."""
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG, 1)
    except Exception:  # pragma: no cover - malformed metadata must not crash loading
        return 1
    return orientation if isinstance(orientation, int) and 1 <= orientation <= 8 else 1


def exif_oriented_size(image: Image.Image) -> tuple[int, int]:
    """Return ``(width, height)`` in the displayed orientation without decoding pixels."""
    width, height = image.size
    return (height, width) if exif_orientation(image) in _SWAPPED_AXES else (width, height)


def exif_oriented(image: Image.Image) -> Image.Image:
    """Return the image's pixels rotated per EXIF so they match ``exif_oriented_size``."""
    if exif_orientation(image) == 1:
        return image
    return ImageOps.exif_transpose(image)

#!/usr/bin/env python3
"""Launch the PrismOCR Gradio demo application."""

from __future__ import annotations

import sys

from prismocr.demo import main

if __name__ == "__main__":
    main(sys.argv[1:])

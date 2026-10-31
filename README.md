# EE513

## ColorChecker detection

Use `uv sync --inexact` to install the locked dependencies without removing
additional installed packages. Restart the notebook kernel after changing OpenCV.

OpenCV contrib is pinned to 4.11.0.86: the installed 5.0.0.93 build gave
inconsistent MCC chart detections on the assignment 4 images. The 4.11 detector
finds all 15 chart images and rejects the five white-sheet images using the
existing detection preview. Detection uses an sRGB-encoded preview only;
patch means are measured from the original linear camera RGB array.

White sheets are the fourth image in each assignment 4 round. Exclude them
from chart extraction rather than treating their detection failures as errors.
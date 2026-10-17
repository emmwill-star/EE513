from collections.abc import Iterable
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


RAW_EXTENSIONS = {".dng", ".raw", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def _load_image(imagePath: Path) -> Image.Image:
	if imagePath.suffix.lower() in RAW_EXTENSIONS:
		import rawpy

		with rawpy.imread(str(imagePath)) as raw:
			return Image.fromarray(raw.postprocess(use_camera_wb=True))
	with Image.open(imagePath) as image:
		return ImageOps.exif_transpose(image).convert("RGB")


def _read_camera_id(imagePath: Path) -> str | None:
	if imagePath.suffix.lower() != ".dng":
		return None

	import tifffile

	with tifffile.TiffFile(imagePath) as dngFile:
		tags = dngFile.pages[0].tags
		cameraId = "Unknown camera"
		for tagCode in (272, 271):
			tag = tags.get(tagCode)
			if tag is not None:
				value = str(tag.value).strip("\x00 ")
				if value:
					cameraId = value
					break

		exifTag = tags.get(34665)
		exif = exifTag.value if exifTag is not None else {}
		cameraDescriptions = [exif.get("LensModel", "")]
		for tagCode in (50708, 42036):
			tag = tags.get(tagCode)
			if tag is not None:
				cameraDescriptions.append(tag.value)
		for description in cameraDescriptions:
			cameraNumber = re.search(
				r"\bcamera\s+(?:#\s*|(?:number|no\.?)\s*)?(\d+)\b(?!\.|\s*(?:mm|cm|m)\b)",
				str(description),
				re.IGNORECASE,
			)
			if cameraNumber is not None:
				return f"{cameraId} | Camera {cameraNumber.group(1)}"
		return cameraId


def create_contact_sheet(
	imagePaths: Iterable[str | Path],
	*,
	columns: int = 5,
	tileWidth: int = 480,
	tileHeight: int = 380,
	labelHeight: int = 40,
	fontSize: int = 24,
	outputPath: str | Path | None = None,
) -> Image.Image:
	"""Create a labeled contact sheet from RAW or Pillow-supported image files.

	Paths are used in the supplied order; rows are calculated automatically.
	Tile and label dimensions are in pixels. Images retain their aspect ratio.
	Filenames appear above photos; DNG camera IDs appear in a bottom label
	band of labelHeight pixels. IDs use Model, falling back to Make or "Unknown camera".
	Explicit camera numbers in LensModel or UniqueCameraModel are appended;
	focal lengths and internal hardware model numbers are not camera numbers.
	Long labels use a smaller font to fit within their tile.
	Pass labelHeight=0 to hide labels. Relative paths use the working directory.
	If outputPath is provided, save there (its parent directory must exist).
	Return the RGB Pillow image without printing or displaying it.

	Raises ValueError for empty inputs or invalid layout dimensions, and
	FileNotFoundError for missing input files. Image decoding errors propagate.
	RAW decoding requires rawpy and uses camera white balance.
    
    Usage: 
        from pathlib import Path
        from utils import create_contact_sheet

        imagePaths = sorted(
            Path("images").glob("cb*.DNG"),
            key=lambda imagePath: int(imagePath.stem[2:]),
        )

        sheet = create_contact_sheet(
            imagePaths,
            columns=5,
            outputPath="images/cb_contact_sheet.png",
        )
        display(sheet)  # Notebook only
 
	"""
	layoutValues = (columns, tileWidth, tileHeight, labelHeight, fontSize)
	if any(not isinstance(value, int) or isinstance(value, bool) for value in layoutValues):
		raise ValueError("Layout dimensions and fontSize must be integers")
	if columns < 1 or fontSize < 1 or labelHeight < 0:
		raise ValueError("columns and fontSize must be positive; labelHeight must be nonnegative")
	if tileWidth <= 24 or tileHeight <= labelHeight + 16:
		raise ValueError("Tiles must leave room for images and padding")
	if labelHeight and labelHeight < fontSize + 8:
		raise ValueError("labelHeight must be at least fontSize + 8, or zero to hide labels")

	paths = [Path(imagePath) for imagePath in imagePaths]
	if not paths:
		raise ValueError("At least one image path is required")
	for imagePath in paths:
		if not imagePath.is_file():
			raise FileNotFoundError(f"Image file not found: {imagePath}")

	rows = (len(paths) + columns - 1) // columns
	sheet = Image.new("RGB", (columns * tileWidth, rows * tileHeight), "white")
	draw = ImageDraw.Draw(sheet)
	font = ImageFont.load_default(size=fontSize)

	for index, imagePath in enumerate(paths):
		with _load_image(imagePath) as image:
			cameraId = _read_camera_id(imagePath) if labelHeight else None
			footerHeight = labelHeight if cameraId is not None else 0
			imageAreaHeight = tileHeight - labelHeight - footerHeight
			if imageAreaHeight <= 16:
				raise ValueError("Tiles must leave room for images, camera labels, and padding")
			image.thumbnail(
				(tileWidth - 24, imageAreaHeight - 16),
				Image.Resampling.LANCZOS,
			)
			left = (index % columns) * tileWidth
			top = (index // columns) * tileHeight
			if labelHeight:
				labels = [(imagePath.stem, top + 8)]
				if cameraId is not None:
					labels.append((cameraId, top + tileHeight - footerHeight + 8))
				for label, labelTop in labels:
					labelFont = font
					labelFontSize = fontSize
					while draw.textlength(label, font=labelFont) > tileWidth - 24 and labelFontSize > 1:
						labelFontSize -= 1
						labelFont = ImageFont.load_default(size=labelFontSize)
					draw.text((left + 12, labelTop), label, fill="black", font=labelFont)
			imageLeft = left + (tileWidth - image.width) // 2
			imageTop = top + labelHeight + (imageAreaHeight - image.height) // 2
			sheet.paste(image, (imageLeft, imageTop))
			draw.rectangle(
				(left, top, left + tileWidth - 1, top + tileHeight - 1),
				outline="#b8b8b8",
				width=2,
			)

	if outputPath is not None:
		sheet.save(outputPath)
	return sheet


def select_largest_square_edge(
	image: np.ndarray,
	*,
	minAreaFraction: float = 0.005,
	roiWidth: int = 48,
	edgeFraction: float = 0.5,
) -> dict:
	"""Find the largest solid dark square-like target and its right-edge ROI.

	Accept a finite 2D green plane. Threshold only a normalized detection copy;
	do not change or resample the input. Allow moderate perspective distortion.
	Return squareCorners, edgeEndpoints, roi=(x, y, width, height), and edgeAngleDeg
	in input-image coordinates. The ROI spans the middle edgeFraction of a
	near-vertical dark-to-light edge, with roiWidth pixels of cross-edge padding.
	Try narrower padding and retain the longest clean run of rows when nearby
	markings contaminate the ROI. Keep at least one third of the requested rows.
	Raise ValueError if no suitable target or unclipped edge ROI is available.
	This is a geometric detector: inspect its overlay before measuring MTF.
	"""
	imageArray = np.asarray(image)
	if imageArray.ndim != 2 or imageArray.size == 0:
		raise ValueError("image must be a non-empty 2D intensity array")
	if not np.issubdtype(imageArray.dtype, np.number) or not np.isfinite(imageArray).all():
		raise ValueError("image must contain finite numeric intensities")
	if not 0 < minAreaFraction < 0.5 or not 0 < edgeFraction < 1:
		raise ValueError("minAreaFraction must be in (0, 0.5); edgeFraction must be in (0, 1)")
	if not isinstance(roiWidth, int) or isinstance(roiWidth, bool) or roiWidth < 8:
		raise ValueError("roiWidth must be an integer of at least 8 pixels")
	lowerLevel, upperLevel = np.percentile(imageArray, (1, 99))
	if upperLevel <= lowerLevel:
		raise ValueError("image has insufficient intensity contrast")
	detectionImage = np.clip(
		(imageArray.astype(np.float64) - lowerLevel) * 255 / (upperLevel - lowerLevel),
		0, 255,
	).astype(np.uint8)
	detectionImage = cv2.GaussianBlur(detectionImage, (5, 5), 0)
	_, mask = cv2.threshold(detectionImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	imageHeight, imageWidth = imageArray.shape
	candidates = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if not minAreaFraction * imageArray.size <= area <= 0.5 * imageArray.size:
			continue
		polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
		if len(polygon) != 4 or not cv2.isContourConvex(polygon):
			continue
		corners = polygon.reshape(4, 2)
		if np.any(corners <= 2) or np.any(corners[:, 0] >= imageWidth - 3) or np.any(corners[:, 1] >= imageHeight - 3):
			continue
		_, (boxWidth, boxHeight), _ = cv2.minAreaRect(polygon)
		if min(boxWidth, boxHeight) == 0 or max(boxWidth, boxHeight) / min(boxWidth, boxHeight) > 1.6:
			continue
		polygonMask = np.zeros_like(mask)
		cv2.fillPoly(polygonMask, [polygon], 255)
		if np.mean(mask[polygonMask != 0] != 0) < 0.9:
			continue
		candidates.append((area, corners))
	if not candidates:
		raise ValueError("No large solid square target found; inspect the capture or minAreaFraction")
	_, squareCorners = max(candidates, key=lambda candidate: candidate[0])
	edges = []
	for index in range(4):
		endpoints = squareCorners[[index, (index + 1) % 4]].astype(np.float64)
		deltaX, deltaY = endpoints[1] - endpoints[0]
		if abs(deltaY) > abs(deltaX):
			edges.append(endpoints)
	if not edges:
		raise ValueError("The square has no near-vertical edge suitable for this MTF function")
	edgeEndpoints = max(edges, key=lambda endpoints: float(np.mean(endpoints[:, 0])))
	midpoint = np.mean(edgeEndpoints, axis=0)
	innerEndpoints = midpoint + (edgeEndpoints - midpoint) * edgeFraction
	top = int(np.ceil(np.min(innerEndpoints[:, 1])))
	bottom = int(np.floor(np.max(innerEndpoints[:, 1]))) + 1
	if top < 0 or bottom > imageHeight or bottom - top < 8:
		raise ValueError("Selected edge ROI is too small or extends outside the image")
	cleanRois = []
	for paddingWidth in dict.fromkeys((roiWidth, max(8, roiWidth // 2), max(8, roiWidth // 4))):
		left = int(np.floor(np.min(innerEndpoints[:, 0]) - paddingWidth / 2))
		right = int(np.ceil(np.max(innerEndpoints[:, 0]) + paddingWidth / 2))
		if left < 0 or right > imageWidth:
			continue
		roiImage = imageArray[top:bottom, left:right]
		plateauWidth = max(1, roiImage.shape[1] // 5)
		darkLevels = np.median(roiImage[:, :plateauWidth], axis=1)
		lightPlateau = roiImage[:, -plateauWidth:]
		lightLevels = np.median(lightPlateau, axis=1)
		cleanRows = (
			(lightLevels > darkLevels)
			& (np.min(lightPlateau, axis=1) > darkLevels + 0.25 * (lightLevels - darkLevels))
		)
		transitions = np.diff(np.concatenate(([False], cleanRows, [False])).astype(np.int8))
		for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
			if stop - start >= max(8, int(np.ceil((bottom - top) / 3))):
				cleanRois.append((int(stop - start), (left, top + int(start), right - left, int(stop - start))))
		if cleanRows.all():
			break
	if not cleanRois:
		raise ValueError("Selected edge has no sufficiently long clean dark-to-light ROI")
	_, roi = max(cleanRois, key=lambda candidate: candidate[0])
	deltaX, deltaY = edgeEndpoints[1] - edgeEndpoints[0]
	return {
		"squareCorners": squareCorners,
		"edgeEndpoints": edgeEndpoints,
		"roi": roi,
		"edgeAngleDeg": float(np.rad2deg(np.arctan(deltaX / deltaY))),
	}

from unittest.mock import MagicMock, patch

import numpy as np
import cv2
from PIL import Image, ImageDraw
import pytest
import tifffile

from utils import _read_camera_id, create_contact_sheet, select_largest_square_edge


def test_layout_order_aspect_ratio_and_save(tmp_path):
    imagePaths = []
    for index, color in enumerate(("red", "green", "blue")):
        imagePath = tmp_path / f"image{index}.png"
        with Image.new("RGB", (200, 100), color) as image:
            image.save(imagePath)
        imagePaths.append(imagePath)

    outputPath = tmp_path / "sheet.png"
    with create_contact_sheet(
        iter(imagePaths), columns=2, tileWidth=100, tileHeight=100, outputPath=outputPath
    ) as sheet:
        assert sheet.size == (200, 200)
        assert sheet.mode == "RGB"
        assert sheet.getpixel((50, 70)) == (255, 0, 0)
        assert sheet.getpixel((150, 70)) == (0, 128, 0)
        assert sheet.getpixel((50, 170)) == (0, 0, 255)
        assert sheet.getpixel((150, 170)) == (255, 255, 255)
        assert sheet.getpixel((50, 45)) == (255, 255, 255)
        assert sheet.getpixel((0, 0)) == (184, 184, 184)
        with Image.open(outputPath) as saved:
            assert saved.size == sheet.size
            assert saved.getpixel((50, 70)) == sheet.getpixel((50, 70))


def test_raw_camera_white_balance(tmp_path):
    imagePath = tmp_path / "camera.DNG"
    imagePath.touch()
    raw = MagicMock()
    raw.postprocess.return_value = np.full((10, 10, 3), 123, dtype=np.uint8)
    with patch("rawpy.imread") as readRaw:
        readRaw.return_value.__enter__.return_value = raw
        with create_contact_sheet([str(imagePath)], columns=1, labelHeight=0) as sheet:
            assert sheet.getpixel((240, 190)) == (123, 123, 123)
        readRaw.assert_called_once_with(str(imagePath))
        raw.postprocess.assert_called_once_with(use_camera_wb=True)
        readRaw.return_value.__exit__.assert_called_once()


def test_missing_and_empty_inputs(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        create_contact_sheet([])
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        create_contact_sheet([tmp_path / "missing.png"])


@pytest.mark.parametrize(
    "options",
    [
        {"columns": 0},
        {"columns": 1.5},
        {"tileWidth": 24},
        {"tileHeight": 56},
        {"labelHeight": -1},
        {"labelHeight": 10},
        {"fontSize": 0},
    ],
)
def test_invalid_layout(options):
    with pytest.raises(ValueError):
        create_contact_sheet([], **options)


@pytest.mark.parametrize(
    "cameraTags, expectedId",
    [
        ([(50708, "s", 0, "iPhone17,3 back camera", False)], "Unknown camera"),
        (
            [(271, "s", 0, "Apple", False), (272, "s", 0, "iPhone 16", False)],
            "iPhone 16",
        ),
        ([(272, "s", 0, "iPhone 16", False)], "iPhone 16"),
        ([(271, "s", 0, "Apple", False)], "Apple"),
        ([], "Unknown camera"),
        (
            [(50708, "s", 0, "", False), (272, "s", 0, "iPhone 16", False)],
            "iPhone 16",
        ),
        (
            [(272, "s", 0, "iPhone 16", False), (50708, "s", 0, "iPhone17,3 back camera", False)],
            "iPhone 16",
        ),
        (
            [(272, "s", 0, "iPhone 16", False), (50708, "s", 0, "iPhone17,3 back camera 2", False)],
            "iPhone 16 | Camera 2",
        ),
        (
            [(272, "s", 0, "iPhone 16", False), (42036, "s", 0, "back camera #1 5.96mm f/1.6", False)],
            "iPhone 16 | Camera 1",
        ),
        (
            [(272, "s", 0, "iPhone 16", False), (42036, "s", 0, "back camera 5.96mm f/1.6", False)],
            "iPhone 16",
        ),
        (
            [(272, "s", 0, "iPhone 16", False), (42036, "s", 0, "back camera 6 mm f/1.6", False)],
            "iPhone 16",
        ),
    ],
)
def test_dng_camera_metadata(tmp_path, cameraTags, expectedId):
    imagePath = tmp_path / "camera.DNG"
    tifffile.imwrite(imagePath, np.zeros((10, 10, 3), dtype=np.uint8), extratags=cameraTags)
    assert _read_camera_id(imagePath) == expectedId


@pytest.mark.parametrize("imageCount", [1, 2])
def test_camera_label_fits_tile(tmp_path, imageCount):
    imagePath = tmp_path / "edge1.DNG"
    tifffile.imwrite(
        imagePath,
        np.zeros((10, 10, 3), dtype=np.uint8),
        extratags=[
            (272, "s", 0, "iPhone 16", False),
            (50708, "s", 0, "iPhone17,3 back camera 2", False),
        ],
    )
    with (
        patch("utils._load_image", side_effect=lambda imagePath: Image.new("RGB", (100, 1000), "red")),
        patch("utils.ImageDraw.ImageDraw.text", autospec=True, side_effect=ImageDraw.ImageDraw.text)
        as drawText,
    ):
        with create_contact_sheet([imagePath] * imageCount, columns=1, tileWidth=180) as sheet:
            assert sheet.size == (180, 380 * imageCount)
            assert drawText.call_count == 2 * imageCount
            for index in range(imageCount):
                top = index * 380
                filenameCall, cameraCall = drawText.call_args_list[index * 2:index * 2 + 2]
                assert filenameCall.args[1:] == ((12, top + 8), "edge1")
                draw, position, label = cameraCall.args
                labelFont = cameraCall.kwargs["font"]
                assert position == (12, top + 348)
                assert label == "iPhone 16 | Camera 2"
                assert draw.textlength(label, font=labelFont) <= 180 - 24
                assert sheet.getpixel((90, top + 331)) == (255, 0, 0)
                assert sheet.getpixel((90, top + 340)) == (255, 255, 255)


def test_camera_footer_needs_image_space(tmp_path):
    imagePath = tmp_path / "camera.DNG"
    tifffile.imwrite(imagePath, np.zeros((10, 10, 3), dtype=np.uint8))
    with patch("utils._load_image", return_value=Image.new("RGB", (10, 10))):
        with pytest.raises(ValueError, match="camera labels"):
            create_contact_sheet([imagePath], tileHeight=90)


def test_other_formats_skip_camera_metadata(tmp_path):
    assert _read_camera_id(tmp_path / "photo.png") is None


def test_largest_square_edge_preserves_data():
    image = np.full((400, 500), 1000, dtype=np.float32)
    largeSquare = np.array([[150, 80], [330, 95], [315, 275], [135, 260]], dtype=np.int32)
    cv2.fillPoly(image, [largeSquare], 20)
    cv2.rectangle(image, (30, 30), (70, 70), 20, -1)
    original = image.copy()
    selection = select_largest_square_edge(image)
    assert cv2.contourArea(selection["squareCorners"]) > 30000
    left, top, width, height = selection["roi"]
    assert 70 < height < 110
    assert left > 270
    crop = image[top:top + height, left:left + width]
    assert np.all(np.median(crop[:, :8], axis=1) == 20)
    assert np.all(np.median(crop[:, -8:], axis=1) == 1000)
    np.testing.assert_array_equal(image, original)


def test_square_detector_rejects_frame_and_rectangle():
    image = np.full((400, 500), 1000, dtype=np.float32)
    cv2.rectangle(image, (0, 0), (499, 399), 20, 12)
    cv2.rectangle(image, (100, 150), (400, 200), 20, -1)
    with pytest.raises(ValueError, match="No large solid square"):
        select_largest_square_edge(image)


def test_square_detector_with_dark_frame_and_adjacent_marking():
    image = np.full((400, 500), 1000, dtype=np.float32)
    cv2.rectangle(image, (5, 5), (494, 394), 20, 15)
    square = np.array([[150, 80], [330, 95], [315, 275], [135, 260]], dtype=np.int32)
    cv2.fillPoly(image, [square], 20)
    cv2.rectangle(image, (337, 160), (345, 180), 0, -1)
    selection = select_largest_square_edge(image)
    assert cv2.contourArea(selection["squareCorners"]) > 30000
    left, top, width, height = selection["roi"]
    crop = image[top:top + height, left:left + width]
    plateauWidth = max(1, width // 5)
    assert np.all(
        np.median(crop[:, -plateauWidth:], axis=1)
        > np.median(crop[:, :plateauWidth], axis=1)
    )
    assert np.all(crop[:, -plateauWidth:] == 1000)
    assert height >= 30


def test_square_detector_trims_contaminated_rows():
    image = np.full((400, 500), 1000, dtype=np.float32)
    cv2.rectangle(image, (150, 80), (330, 280), 20, -1)
    cv2.rectangle(image, (334, 175), (370, 185), 0, -1)
    selection = select_largest_square_edge(image)
    left, top, width, height = selection["roi"]
    assert height < 101
    assert top + height <= 175 or top > 185
    assert left < 330 < left + width


@pytest.mark.parametrize("image", [np.zeros((100, 100)), np.zeros((10, 10, 3)), np.full((20, 20), np.nan)])
def test_square_detector_invalid_images(image):
    with pytest.raises(ValueError):
        select_largest_square_edge(image)
from __future__ import annotations

import math
import unittest

from prismocr.tiling import Tile, TileResult, map_tile_polygon, plan_tiles, stitch_tile_results


def rectangle(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def rotated_rectangle(cx: float, cy: float, width: float, height: float, angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [cx + x * cosine - y * sine, cy + x * sine + y * cosine]
        for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2))
    ]


class TilePlanningTests(unittest.TestCase):
    def test_small_page_uses_one_tile(self) -> None:
        tiles = plan_tiles((40, 30), (128, 96), overlap=16)
        self.assertEqual(tiles, (Tile(index=0, x=0, y=0, width=40, height=30),))

    def test_exact_and_tail_coverage_has_no_gaps(self) -> None:
        tiles = plan_tiles((1001, 701), (400, 300), overlap=(100, 50))
        self.assertEqual(
            [(tile.index, tile.x, tile.y) for tile in tiles],
            [
                (0, 0, 0), (1, 300, 0), (2, 600, 0), (3, 900, 0),
                (4, 0, 250), (5, 300, 250), (6, 600, 250), (7, 900, 250),
                (8, 0, 500), (9, 300, 500), (10, 600, 500), (11, 900, 500),
            ],
        )
        self.assertEqual(tiles[-1].size, (101, 201))
        covered = bytearray(1001 * 701)
        for tile in tiles:
            for y in range(tile.y, tile.y + tile.height):
                start = y * 1001 + tile.x
                covered[start : start + tile.width] = b"\x01" * tile.width
        self.assertTrue(all(covered))
        self.assertEqual(tiles, plan_tiles((1001, 701), (400, 300), overlap=(100, 50)))

    def test_invalid_dimensions_and_overlap_are_rejected(self) -> None:
        invalid_calls = (
            lambda: plan_tiles((0, 10), (5, 5), 1),
            lambda: plan_tiles((10, 10), (0, 5), 1),
            lambda: plan_tiles((10, 10), (5, 5), 5),
            lambda: plan_tiles((10, 10), (5, 5), (-1, 0)),
            lambda: plan_tiles((10, 10), (5,), 1),
            lambda: plan_tiles((True, 10), (5, 5), 1),
            lambda: plan_tiles(10, (5, 5), 1),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_local_polygon_is_translated_by_tile_origin(self) -> None:
        tile = Tile(index=3, x=24, y=50, width=80, height=60)
        self.assertEqual(
            map_tile_polygon(rectangle(1, 2, 5, 6), tile),
            [[25.0, 52.0], [29.0, 52.0], [29.0, 56.0], [25.0, 56.0]],
        )
        with self.assertRaises(ValueError):
            map_tile_polygon([[float("nan"), 0], [1, 0], [0, 1]], tile)


class TileStitchingTests(unittest.TestCase):
    def test_same_line_from_overlapping_tiles_is_deduplicated(self) -> None:
        result = stitch_tile_results(
            [
                TileResult(
                    Tile(0, 0, 0, 100, 80),
                    [{"id": "local-a", "polygon": rectangle(60, 20, 90, 30), "text": "invoice", "detector_score": 0.8}],
                ),
                TileResult(
                    Tile(1, 50, 0, 100, 80),
                    [{"id": "local-b", "polygon": rectangle(10, 20, 40, 30), "text": "invoice", "detector_score": 0.9}],
                ),
            ]
        )
        self.assertEqual(len(result.detections), 1)
        detection = result.detections[0]
        self.assertEqual(detection["id"], "l:1")
        self.assertEqual(detection["polygon"], rectangle(60, 20, 90, 30))
        self.assertEqual(detection["source_tiles"], [0, 1])
        self.assertEqual([source["tile_origin"] for source in detection["source_detections"]], [[0, 0], [50, 0]])
        self.assertEqual([source["detection"]["detector_score"] for source in detection["source_detections"]], [0.8, 0.9])
        self.assertEqual(detection["detector_score"], 0.9)

    def test_calibrated_score_saturation_does_not_change_raw_score_stitch_choice(self) -> None:
        result = stitch_tile_results(
            [
                TileResult(Tile(0, 0, 0, 100, 80), [{
                    "polygon": rectangle(60, 20, 90, 30),
                    "text": "lower raw score",
                    "detector_score": 1.0,
                    "_tile_score": 0.98,
                }]),
                TileResult(Tile(1, 50, 0, 100, 80), [{
                    "polygon": rectangle(10, 20, 40, 30),
                    "text": "higher raw score",
                    "detector_score": 1.0,
                    "_tile_score": 0.99,
                }]),
            ]
        )

        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0]["text"], "higher raw score")
        self.assertEqual(result.detections[0]["source_tiles"], [0, 1])

    def test_adjacent_lines_remain_distinct_while_each_line_stitches(self) -> None:
        left = Tile(0, 0, 0, 100, 80)
        right = Tile(1, 50, 0, 100, 80)
        result = stitch_tile_results(
            [
                TileResult(left, [
                    {"polygon": rectangle(60, 20, 90, 28), "text": "same"},
                    {"polygon": rectangle(60, 38, 90, 46), "text": "next"},
                ]),
                TileResult(right, [
                    {"polygon": rectangle(10, 20, 40, 28), "text": "same"},
                    {"polygon": rectangle(10, 38, 40, 46), "text": "next"},
                ]),
            ]
        )
        self.assertEqual(len(result.detections), 2)
        self.assertEqual([item["source_tiles"] for item in result.detections], [[0, 1], [0, 1]])
        self.assertEqual([item["text"] for item in result.detections], ["same", "next"])

    def test_identical_text_in_different_locations_is_not_a_duplicate(self) -> None:
        result = stitch_tile_results(
            [
                TileResult(Tile(0, 0, 0, 100, 80), [{"polygon": rectangle(55, 10, 75, 20), "text": "repeat"}]),
                TileResult(Tile(1, 50, 0, 100, 80), [{"polygon": rectangle(35, 50, 55, 60), "text": "repeat"}]),
            ]
        )
        self.assertEqual(len(result.detections), 2)
        self.assertEqual([item["source_tiles"] for item in result.detections], [[0], [1]])

    def test_ambiguous_candidates_are_reported_and_retained(self) -> None:
        result = stitch_tile_results(
            [
                TileResult(Tile(0, 0, 0, 100, 80), [{"polygon": rectangle(60, 20, 90, 30), "text": "line"}]),
                TileResult(Tile(1, 50, 0, 100, 80), [
                    {"polygon": rectangle(10, 20, 40, 30), "text": "line-a"},
                    {"polygon": rectangle(11, 20, 41, 30), "text": "line-b"},
                ]),
            ]
        )
        self.assertEqual(len(result.detections), 3)
        self.assertEqual(len(result.ambiguous_merges), 1)
        self.assertEqual(result.warnings[0]["code"], "AMBIGUOUS_TILE_MERGE")

    def test_angle_and_area_mismatches_block_text_assisted_merges(self) -> None:
        left = TileResult(Tile(0, 0, 0, 100, 80), [
            {"polygon": rotated_rectangle(70, 25, 30, 10, 0), "text": "same"},
            {"polygon": rotated_rectangle(70, 55, 30, 10, 0), "text": "same"},
        ])
        right = TileResult(Tile(1, 50, 0, 100, 80), [
            {"polygon": rotated_rectangle(20, 25, 30, 10, math.radians(35)), "text": "same"},
            {"polygon": rotated_rectangle(20, 55, 70, 10, 0), "text": "same"},
        ])
        result = stitch_tile_results(
            [left, right], iou_threshold=0.01, text_iou_threshold=0.01, angle_threshold_degrees=10
        )
        self.assertEqual(len(result.detections), 4)

    def test_low_iou_same_text_merges_only_when_crop_boundary_clipped(self) -> None:
        result = stitch_tile_results(
            [
                TileResult(Tile(0, 0, 0, 100, 80), [
                    {"polygon": rectangle(30, 20, 100, 30), "text": "clipped line"},
                ]),
                TileResult(Tile(1, 50, 0, 100, 80), [
                    {"polygon": rectangle(0, 20, 70, 30), "text": "clipped line"},
                ]),
            ],
            iou_threshold=0.8,
            text_iou_threshold=0.8,
        )
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0]["source_tiles"], [0, 1])

    def test_capacity_overflow_is_reported(self) -> None:
        result = stitch_tile_results(
            [TileResult(Tile(0, 0, 0, 100, 80), [], candidate_count=5, capacity=3)]
        )
        self.assertTrue(result.overflow)
        self.assertEqual(
            result.warnings,
            ({"code": "TILE_CAPACITY_OVERFLOW", "tile_index": 0, "candidate_count": 5, "capacity": 3},),
        )

    def test_ids_order_and_source_metadata_are_independent_of_result_input_order(self) -> None:
        first = TileResult(Tile(0, 0, 0, 100, 80), [
            {"id": "a", "polygon": rectangle(70, 40, 90, 50), "text": "lower", "score": 0.6},
            {"id": "b", "polygon": rectangle(70, 10, 90, 20), "text": "upper", "score": 0.7},
        ])
        second = TileResult(Tile(1, 50, 0, 100, 80), [
            {"id": "c", "polygon": rectangle(20, 40, 40, 50), "text": "lower", "score": 0.8},
            {"id": "d", "polygon": rectangle(20, 10, 40, 20), "text": "upper", "score": 0.9},
        ])
        forward = stitch_tile_results([first, second])
        reverse = stitch_tile_results([second, first])
        self.assertEqual(forward.detections, reverse.detections)
        self.assertEqual([item["id"] for item in forward.detections], ["l:1", "l:2"])
        self.assertEqual([item["text"] for item in forward.detections], ["upper", "lower"])
        self.assertEqual(forward.detections[0]["source_tiles"], [0, 1])
        self.assertEqual(forward.detections[0]["source_detections"][0]["detection"]["id"], "b")


if __name__ == "__main__":
    unittest.main()

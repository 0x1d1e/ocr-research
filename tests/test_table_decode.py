from __future__ import annotations

import unittest

from prismocr.table_decode import reconstruct_tables


def box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    tables = [{"id": "t", "polygon_px": box(0, 0, 30, 20)}]
    cells = [
        {"id": "a", "polygon_px": box(0, 0, 10, 20), "row": 0, "col": 0, "rowspan": 2},
        {"id": "b", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 1, "is_header": True},
        {"id": "c", "polygon_px": box(20, 0, 30, 10), "row": 0, "col": 2},
        {"id": "d", "polygon_px": box(10, 10, 30, 20), "row": 1, "col": 1, "colspan": 2},
    ]
    lines = [
        {"id": "l-a", "polygon_px": box(2, 4, 8, 8), "text": "A", "reading_order": 0},
        {"id": "l-b2", "polygon_px": box(12, 4, 18, 8), "text": "second", "reading_order": 2},
        {"id": "l-b1", "polygon_px": box(12, 1, 18, 3), "text": "first", "reading_order": 1},
        {"id": "l-d", "polygon_px": box(12, 12, 18, 18), "text": "D", "reading_order": 3},
    ]
    return tables, cells, lines


class TableDecodeTests(unittest.TestCase):
    def test_explicit_spans_empty_cell_order_and_headers(self) -> None:
        tables, cells, lines = fixture()
        result = reconstruct_tables(tables, cells, lines)
        self.assertEqual(len(result.tables), 1)
        table = result.tables[0]
        self.assertEqual((table.status, table.rows, table.cols), ("valid", 2, 3))
        self.assertEqual(table.conflicts, ())
        by_id = {cell.id: cell for cell in table.cells}
        self.assertEqual((by_id["a"].rowspan, by_id["a"].colspan), (2, 1))
        self.assertEqual((by_id["d"].rowspan, by_id["d"].colspan), (1, 2))
        self.assertEqual(by_id["b"].text, "first\nsecond")
        self.assertEqual(by_id["b"].line_ids, ("l-b1", "l-b2"))
        self.assertEqual((by_id["c"].text, by_id["c"].line_ids), ("", ()))
        self.assertIn('<td rowspan="2">A</td>', table.html or "")
        self.assertIn("<th>first\nsecond</th>", table.html or "")
        self.assertIn('<td colspan="2">D</td>', table.html or "")
        self.assertEqual((table.html or "").count("<th>"), 1)
        self.assertEqual(result.unassigned_line_ids, ())

    def test_geometric_tracks_infer_rows_columns_and_spans(self) -> None:
        tables, cells, lines = fixture()
        for cell in cells:
            cell["table_id"] = None
            cell["row"] = None
            cell["col"] = None
            cell["rowspan"] = None
            cell["colspan"] = None
            cell["is_header"] = None
        result = reconstruct_tables(tables, cells, lines)
        table = result.tables[0]
        self.assertEqual((table.status, table.rows, table.cols), ("valid", 2, 3))
        by_id = {cell.id: cell for cell in table.cells}
        self.assertEqual((by_id["a"].row, by_id["a"].col, by_id["a"].rowspan), (0, 0, 2))
        self.assertEqual((by_id["d"].row, by_id["d"].col, by_id["d"].colspan), (1, 1, 2))

    def test_geometric_tracks_merge_small_inter_cell_gutters(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 204, 10)}]
        cells = [
            {"id": "left", "polygon_px": box(0, 0, 100, 10)},
            {"id": "right", "polygon_px": box(104, 0, 204, 10)},
        ]

        table = reconstruct_tables(tables, cells, []).tables[0]

        self.assertEqual((table.status, table.cols), ("valid", 2))
        self.assertEqual([(cell.id, cell.col) for cell in table.cells], [("left", 0), ("right", 1)])

    def test_geometric_tracks_preserve_narrow_real_columns(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 205, 10)}]
        cells = [
            {"id": "left", "polygon_px": box(0, 0, 100, 10)},
            {"id": "narrow", "polygon_px": box(100, 0, 105, 10)},
            {"id": "right", "polygon_px": box(105, 0, 205, 10)},
        ]

        table = reconstruct_tables(tables, cells, []).tables[0]

        self.assertEqual((table.status, table.cols), ("valid", 3))
        self.assertEqual(
            [(cell.id, cell.col) for cell in table.cells],
            [("left", 0), ("narrow", 1), ("right", 2)],
        )

    def test_oversized_grid_is_rejected_before_rendering(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 10, 10)}]
        cells = [{"id": "c", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 0, "rowspan": 100_001}]
        table = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual(table.status, "conflict")
        self.assertIsNone(table.html)
        self.assertTrue(any("renderer capacity" in issue for issue in table.conflicts))

    def test_huge_single_dimension_empty_grid_is_bounded(self) -> None:
        tables = [{
            "id": "huge", "polygon_px": box(0, 0, 10, 10),
            "rows": 1_000_000_000, "cols": 0,
        }]
        table = reconstruct_tables(tables, [], []).tables[0]
        self.assertEqual(table.status, "conflict")
        self.assertIsNone(table.html)
        self.assertTrue(any("renderer capacity" in issue for issue in table.conflicts))

    def test_spatially_reversed_learned_columns_are_conflicts(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 10)}]
        cells = [
            {"id": "left", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 1},
            {"id": "right", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 0},
        ]
        table = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual(table.status, "conflict")
        self.assertIsNone(table.html)
        self.assertTrue(any("reversed spatial column assignment" in item for item in table.conflicts))

    def test_calibrated_right_cell_edges_resolve_reversed_column_preferences(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 10)}]
        cells = [
            {"id": "left", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 1},
            {"id": "right", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 0},
        ]
        unresolved = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual(unresolved.status, "conflict")

        solved = reconstruct_tables(
            tables,
            cells,
            [],
            right_cell_scores={"left": {"right": 0.9}, "right": {"left": 0.1}},
            right_cell_null_scores={"left": 0.2, "right": 0.2},
            use_right_cell_constraints=True,
        ).tables[0]
        self.assertEqual(solved.status, "valid")
        by_id = {cell.id: cell for cell in solved.cells}
        self.assertEqual((by_id["left"].col, by_id["right"].col), (0, 1))

    def test_right_cell_affinity_that_reverses_geometry_is_ignored(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 10)}]
        cells = [
            {"id": "a", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 0},
            {"id": "b", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 1},
        ]
        table = reconstruct_tables(
            tables,
            cells,
            [],
            right_cell_scores={"a": {"b": 0.9}, "b": {"a": 0.8}},
            right_cell_null_scores={"a": 0.2, "b": 0.2},
            use_right_cell_constraints=True,
        ).tables[0]
        self.assertEqual(table.status, "valid")
        self.assertEqual([(cell.id, cell.col) for cell in table.cells], [("a", 0), ("b", 1)])

    def test_right_cell_affinity_ties_use_canonical_graph_target_order(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 30, 10)}]
        cells = [
            {"id": "a", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 1},
            {"id": "b", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 0},
            {"id": "c", "polygon_px": box(20, 0, 30, 10), "row": 0, "col": 2},
        ]
        table = reconstruct_tables(
            tables,
            cells,
            [],
            right_cell_scores={"a": {"c": 0.9, "b": 0.9}},
            right_cell_null_scores={"a": 0.2},
            use_right_cell_constraints=True,
        ).tables[0]
        self.assertEqual(table.status, "valid")
        self.assertEqual([(cell.id, cell.col) for cell in table.cells], [("a", 0), ("b", 1), ("c", 2)])

    def test_duplicate_origins_and_overlapping_spans_are_conflicts(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 20)}]
        cells = [
            {"id": "a", "polygon_px": box(0, 0, 10, 20), "row": 0, "col": 0, "rowspan": 2},
            {"id": "b", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 0},
            {"id": "c", "polygon_px": box(10, 10, 20, 20), "row": 1, "col": 0},
        ]
        table = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual(table.status, "conflict")
        self.assertIsNone(table.html)
        self.assertEqual(len(table.cells), 3)
        self.assertTrue(any("share origin" in conflict for conflict in table.conflicts))
        self.assertTrue(any("overlap at" in conflict for conflict in table.conflicts))

    def test_declared_grid_dimensions_are_preserved_and_enforced(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 20), "rows": 2, "cols": 2}]
        cells = [{"id": "c", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 0}]
        table = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual((table.status, table.rows, table.cols), ("valid", 2, 2))
        self.assertIn("<td></td>", table.html or "")

        tables[0]["rows"] = 0
        invalid = reconstruct_tables(tables, cells, []).tables[0]
        self.assertEqual(invalid.status, "conflict")
        self.assertIsNone(invalid.html)
        self.assertTrue(any("declared row count" in issue for issue in invalid.conflicts))

    def test_cell_crossing_two_tables_is_left_unassigned(self) -> None:
        tables = [
            {"id": "left", "polygon_px": box(0, 0, 10, 10)},
            {"id": "right", "polygon_px": box(10, 0, 20, 10)},
        ]
        cells = [{"id": "cross", "polygon_px": box(8, 2, 12, 8)}]
        result = reconstruct_tables(tables, cells, [])
        self.assertEqual(result.unassigned_cells[0].id, "cross")
        self.assertEqual(result.unassigned_cells[0].reason, "ambiguous_table_overlap")
        self.assertEqual(result.unassigned_cells[0].polygon, tuple(map(tuple, box(8, 2, 12, 8))))
        self.assertEqual([table.status for table in result.tables], ["empty", "empty"])
        self.assertTrue(all(table.html is not None for table in result.tables))

    def test_explicit_table_id_resolves_overlap_without_duplicating_cell(self) -> None:
        tables = [
            {"id": "left", "polygon_px": box(0, 0, 12, 10)},
            {"id": "right", "polygon_px": box(8, 0, 20, 10)},
        ]
        cells = [{"id": "c", "table_id": "right", "polygon_px": box(9, 2, 11, 8)}]
        result = reconstruct_tables(tables, cells, [])
        self.assertEqual([len(table.cells) for table in result.tables], [0, 1])
        self.assertEqual(result.unassigned_cells, ())

    def test_line_in_overlapping_cells_is_not_duplicated(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 20, 10)}]
        cells = [
            {"id": "a", "polygon_px": box(0, 0, 15, 10), "row": 0, "col": 0},
            {"id": "b", "polygon_px": box(10, 0, 20, 10), "row": 0, "col": 1},
        ]
        lines = [{"id": "l", "polygon_px": box(11, 2, 14, 8), "text": "once"}]
        result = reconstruct_tables(tables, cells, lines)
        table = result.tables[0]
        self.assertEqual(table.status, "conflict")
        self.assertIsNone(table.html)
        self.assertEqual(sum(cell.line_ids.count("l") for cell in table.cells), 0)
        self.assertEqual(result.unassigned_line_ids, ("l",))

    def test_untrusted_content_and_attribute_values_are_escaped(self) -> None:
        tables = [{"id": 't"><script>', "polygon_px": box(0, 0, 10, 10)}]
        cells = [{"id": "c", "polygon_px": box(0, 0, 10, 10)}]
        lines = [{"id": "l", "polygon_px": box(2, 2, 8, 8), "text": '<script>alert("x")</script>&'}]
        result = reconstruct_tables(tables, cells, lines)
        html = result.tables[0].html or ""
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&quot;x&quot;", html)
        self.assertIn("&amp;", html)

    def test_header_tags_require_explicit_true_without_first_row_assumption(self) -> None:
        tables = [{"id": "t", "polygon_px": box(0, 0, 10, 20)}]
        cells = [
            {"id": "body", "polygon_px": box(0, 0, 10, 10), "row": 0, "col": 0},
            {"id": "header", "polygon_px": box(0, 10, 10, 20), "row": 1, "col": 0, "is_header": True},
        ]
        result = reconstruct_tables(tables, cells, [])
        html = result.tables[0].html or ""
        self.assertIn("<td></td>", html)
        self.assertIn("<th></th>", html)
        self.assertEqual(html.count("<th"), 1)

    def test_empty_table_is_explicit_and_unordered_inputs_are_stable(self) -> None:
        empty = reconstruct_tables([{"id": "empty", "polygon_px": box(0, 0, 10, 10)}], [], [])
        self.assertEqual(empty.tables[0].status, "empty")
        self.assertEqual(empty.tables[0].cells, ())
        self.assertEqual(empty.tables[0].html, "<table></table>")

        tables, cells, lines = fixture()
        forward = reconstruct_tables(tables, cells, lines)
        reverse = reconstruct_tables(list(reversed(tables)), list(reversed(cells)), list(reversed(lines)))
        self.assertEqual(forward, reverse)


if __name__ == "__main__":
    unittest.main()

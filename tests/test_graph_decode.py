from __future__ import annotations

import unittest

from prismocr.graph_decode import RELATION_NAMES, GraphNode, decode_graph, relation_candidate_error
from prismocr.model.network import RELATION_NAMES as MODEL_RELATION_NAMES


def rect(left: float, top: float, right: float, bottom: float) -> tuple[tuple[float, float], ...]:
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def selected(graph, relation: str, source: str):
    return next(
        item for item in graph.predictions
        if item.relation == relation and item.source_id == source
    )


def suppressed(graph, relation: str, source: str, target: str | None, reason: str):
    return any(
        item.relation == relation
        and item.source_id == source
        and item.target_id == target
        and item.reason_code == reason
        for item in graph.suppressed
    )


class GraphDecodeTests(unittest.TestCase):
    def test_channel_order_matches_production_network(self) -> None:
        self.assertEqual(RELATION_NAMES, MODEL_RELATION_NAMES)

    def test_empty_single_and_two_column_order_are_stable(self) -> None:
        empty = decode_graph((), {}, {})
        self.assertEqual(empty.nodes, ())
        self.assertEqual(empty.reading_order, ())

        only = GraphNode("l:only", "line", polygon=rect(2, 2, 8, 5))
        single = decode_graph(
            [only],
            {"NEXT": {"l:only": {}}},
            {"NEXT": {"l:only": 0.75}},
        )
        self.assertEqual(single.reading_order, ("l:only",))
        self.assertIsNone(selected(single, "NEXT", "l:only").target_id)

        nodes = [
            GraphNode("l:right-2", "line", polygon=rect(70, 70, 90, 80), scope_id="column-2"),
            GraphNode("l:left-1", "line", polygon=rect(5, 5, 25, 15), scope_id="column-1"),
            GraphNode("l:right-1", "line", polygon=rect(70, 5, 90, 15), scope_id="column-2"),
            GraphNode("l:left-2", "line", polygon=rect(5, 70, 25, 80), scope_id="column-1"),
        ]
        scores = {"NEXT": {
            "l:left-1": {"l:left-2": 0.9},
            "l:right-1": {"l:right-2": 0.8},
        }}
        nulls = {"NEXT": {
            "l:left-1": 0.1,
            "l:left-2": 0.1,
            "l:right-1": 0.1,
            "l:right-2": 0.1,
        }}
        first = decode_graph(nodes, scores, nulls)
        second = decode_graph(list(reversed(nodes)), scores, nulls)
        self.assertEqual(first.reading_order, ("l:left-1", "l:right-1", "l:left-2", "l:right-2"))
        self.assertEqual(first.reading_order, second.reading_order)
        self.assertEqual(tuple(node.id for node in first.nodes), tuple(node.id for node in second.nodes))

    def test_scope_names_do_not_move_headings_or_later_blocks(self) -> None:
        nodes = [
            GraphNode("after", "line", polygon=rect(5, 80, 25, 90), scope_id="document"),
            GraphNode("inside", "line", polygon=rect(40, 45, 60, 55), scope_id="table:t"),
            GraphNode("before", "line", polygon=rect(5, 5, 25, 15), scope_id="document"),
        ]
        result = decode_graph(nodes, {}, {})
        self.assertEqual(result.reading_order, ("before", "inside", "after"))

    def test_next_cycle_break_is_confidence_aware_and_retains_every_line(self) -> None:
        nodes = [
            GraphNode("a", "line", polygon=rect(0, 0, 10, 4)),
            GraphNode("b", "line", polygon=rect(20, 0, 30, 4)),
            GraphNode("c", "line", polygon=rect(40, 0, 50, 4)),
        ]
        result = decode_graph(
            nodes,
            {"NEXT": {"a": {"b": 0.7}, "b": {"c": 0.9}, "c": {"a": 0.8}}},
            {"NEXT": {"a": 0.1, "b": 0.4, "c": 0.1}},
        )
        self.assertEqual(result.reading_order, ("c", "a", "b"))
        self.assertEqual(set(result.reading_order), {"a", "b", "c"})
        self.assertIsNone(selected(result, "NEXT", "b").target_id)
        self.assertIn("NEXT_CYCLE_BROKEN", selected(result, "NEXT", "b").reason_codes)
        self.assertTrue(suppressed(result, "NEXT", "b", "c", "NEXT_CYCLE_BROKEN"))

    def test_invalid_types_and_self_links_are_reported(self) -> None:
        line = GraphNode("l:key", "line", role="KEY", polygon=rect(1, 1, 4, 3))
        value = GraphNode("l:value", "line", role="VALUE", polygon=rect(1, 5, 4, 7))
        region = GraphNode("r:1", "region", kind="REGION", polygon=rect(0, 0, 10, 10))
        result = decode_graph(
            [line, value, region],
            {
                "NEXT": {"l:key": {"r:1": 0.9, "l:key": 0.8}},
                "KV_VALUE": {"l:value": {"l:key": 0.9}},
                "PARENT": {"l:key": {"l:value": 0.9}},
            },
            {
                "NEXT": {"l:key": 0.1},
                "KV_VALUE": {"l:value": 0.1},
                "PARENT": {"l:key": 0.1},
            },
        )
        self.assertTrue(suppressed(result, "NEXT", "l:key", "r:1", "INVALID_ENDPOINT_TYPES"))
        self.assertTrue(suppressed(result, "NEXT", "l:key", "l:key", "SELF_LINK"))
        self.assertTrue(suppressed(result, "KV_VALUE", "l:value", "l:key", "INVALID_SOURCE_TYPE"))
        self.assertTrue(suppressed(result, "PARENT", "l:key", "l:value", "INVALID_ENDPOINT_TYPES"))

    def test_null_decisions_are_type_checked(self) -> None:
        nodes = [
            GraphNode("r:region", "region", kind="REGION"),
            GraphNode("l:value", "line", role="VALUE"),
            GraphNode("c:cell", "cell"),
        ]
        result = decode_graph(
            nodes,
            {},
            {
                "NEXT": {"r:region": 0.1},
                "KV_VALUE": {"l:value": 0.1},
                "RIGHT_CELL": {"l:value": 0.1},
            },
        )
        self.assertFalse(result.predictions)
        self.assertTrue(suppressed(result, "NEXT", "r:region", None, "INVALID_SOURCE_TYPE"))
        self.assertTrue(suppressed(result, "KV_VALUE", "l:value", None, "INVALID_SOURCE_TYPE"))
        self.assertTrue(suppressed(result, "RIGHT_CELL", "l:value", None, "INVALID_SOURCE_TYPE"))

    def test_parent_prefers_authoritative_table_membership(self) -> None:
        nodes = [
            GraphNode("c:cell", "cell", polygon=rect(1, 1, 2, 2), table_id="r:left"),
            GraphNode("r:left", "region", kind="TABLE", polygon=rect(0, 0, 10, 10)),
            GraphNode("r:right", "region", kind="TABLE", polygon=rect(0, 0, 10, 10)),
        ]
        result = decode_graph(
            nodes,
            {"PARENT": {"c:cell": {"r:left": 0.8, "r:right": 0.9}}},
            {"PARENT": {"c:cell": 0.1}},
        )
        self.assertEqual(selected(result, "PARENT", "c:cell").target_id, "r:left")
        self.assertTrue(suppressed(
            result, "PARENT", "c:cell", "r:right", "AUTHORITATIVE_TABLE_MEMBERSHIP"
        ))
        self.assertEqual(
            next(item for item in result.predictions if item.relation == "CONTAINS").source_id,
            "r:left",
        )

    def test_tiny_disjoint_parent_is_rejected_without_absolute_tolerance(self) -> None:
        tiny = rect(0.8, 0.8, 0.80005, 0.80005)
        parent = rect(0, 0, 0.1, 0.1)
        result = decode_graph(
            [
                GraphNode("l:child", "line", polygon=tiny),
                GraphNode("r:parent", "region", kind="REGION", polygon=parent),
            ],
            {"PARENT": {"l:child": {"r:parent": 0.9}}},
            {"PARENT": {"l:child": 0.1}},
        )
        self.assertIsNone(selected(result, "PARENT", "l:child").target_id)
        self.assertTrue(suppressed(
            result, "PARENT", "l:child", "r:parent", "PARENT_GEOMETRY_MISMATCH"
        ))
        self.assertFalse(any(item.relation == "CONTAINS" for item in result.predictions))

    def test_parent_requires_available_full_containment_geometry(self) -> None:
        missing = GraphNode("l:missing", "line")
        outside = GraphNode("l:outside", "line", polygon=rect(20, 20, 30, 30))
        parent = GraphNode("r:parent", "region", kind="REGION", polygon=rect(0, 0, 10, 10))
        result = decode_graph(
            [missing, outside, parent],
            {"PARENT": {
                "l:missing": {"r:parent": 0.9},
                "l:outside": {"r:parent": 0.8},
            }},
            {"PARENT": {"l:missing": 0.1, "l:outside": 0.1}},
        )
        self.assertTrue(suppressed(result, "PARENT", "l:missing", "r:parent", "PARENT_GEOMETRY_MISSING"))
        self.assertTrue(suppressed(result, "PARENT", "l:outside", "r:parent", "PARENT_GEOMETRY_MISMATCH"))
        self.assertIsNone(selected(result, "PARENT", "l:missing").target_id)

    def test_malformed_parent_geometry_and_table_identity_do_not_crash(self) -> None:
        parent = GraphNode("r:parent", "region", kind="TABLE", polygon=rect(0, 0, 10, 10))
        invalid_polygons = (
            [{}, [2, 1], [2, 3], [1, 3]],
            7,
            ["11", "21", "22", "12"],
            [["1", 1], [2, 1], [2, 3], [1, 3]],
            [[True, 1], [2, 1], [2, 3], [1, 3]],
            [[1, 1, 0], [2, 1, 0], [2, 3, 0], [1, 3, 0]],
            {"points": [[1, 1], [2, 1], [2, 3], [1, 3]]},
        )
        for index, polygon in enumerate(invalid_polygons):
            child = GraphNode(f"l:invalid-{index}", "line", polygon=polygon)  # type: ignore[arg-type]
            self.assertEqual(
                relation_candidate_error("PARENT", child, parent, {parent.id: parent, child.id: child}),
                "PARENT_GEOMETRY_MISSING",
            )
        scalar_child = GraphNode("l:scalar", "line", polygon=7)  # type: ignore[arg-type]
        self.assertEqual(
            decode_graph([scalar_child], {}, {}).reading_order,
            ("l:scalar",),
        )

        cell = GraphNode(
            "c:cell", "cell", polygon=rect(1, 1, 2, 2), table_id={"invalid": "id"}
        )
        self.assertIsNone(relation_candidate_error(
            "PARENT", cell, parent, {parent.id: parent, cell.id: cell}
        ))

    def test_parent_cycle_is_cut_and_contains_is_derived_from_surviving_parent(self) -> None:
        same_shape = rect(0, 0, 20, 20)
        nodes = [
            GraphNode("c:a", "cell", polygon=same_shape, table_id="r:t", row=0, col=0),
            GraphNode("c:b", "cell", polygon=same_shape, table_id="r:t", row=0, col=1),
            GraphNode("r:t", "region", kind="TABLE", polygon=rect(-1, -1, 21, 21)),
        ]
        result = decode_graph(
            nodes,
            {
                "PARENT": {"c:a": {"c:b": 0.9}, "c:b": {"c:a": 0.4}},
                "CONTAINS": {"c:b": {"c:a": 0.99}},
            },
            {"PARENT": {"c:a": 0.1, "c:b": 0.1}},
        )
        self.assertEqual(selected(result, "PARENT", "c:a").target_id, "c:b")
        self.assertIsNone(selected(result, "PARENT", "c:b").target_id)
        self.assertIn("PARENT_CYCLE_BROKEN", selected(result, "PARENT", "c:b").reason_codes)
        derived = next(item for item in result.predictions if item.relation == "CONTAINS")
        self.assertEqual((derived.source_id, derived.target_id, derived.score), ("c:b", "c:a", 0.9))
        self.assertTrue(suppressed(result, "PARENT", "c:b", "c:a", "PARENT_CYCLE_BROKEN"))
        self.assertTrue(suppressed(result, "CONTAINS", "c:b", "c:a", "CONTAINS_DERIVED_ONLY"))
        self.assertEqual(len([item for item in result.predictions if item.relation == "CONTAINS"]), 1)

    def test_kv_selects_one_value_per_key_and_preserves_duplicate_keys(self) -> None:
        nodes = [
            GraphNode("key:1", "line", role="KEY"),
            GraphNode("key:2", "line", role="KEY"),
            GraphNode("key:3", "line", role="KEY"),
            GraphNode("value:1", "line", role="VALUE"),
            GraphNode("value:2", "line", role="VALUE"),
            GraphNode("r:field", "region", kind="FORM_FIELD"),
        ]
        result = decode_graph(
            nodes,
            {"KV_VALUE": {
                "key:1": {"value:1": 0.9, "value:2": 0.9, "r:field": 0.3},
                "key:2": {"value:1": 0.8, "value:2": 0.2},
                "key:3": {"value:1": 0.2, "r:field": 0.3},
            }},
            {"KV_VALUE": {"key:1": 0.1, "key:2": 0.1, "key:3": 0.5}},
        )
        self.assertEqual(selected(result, "KV_VALUE", "key:1").target_id, "value:1")
        self.assertIn("AMBIGUOUS_TIE", selected(result, "KV_VALUE", "key:1").reason_codes)
        self.assertEqual(selected(result, "KV_VALUE", "key:2").target_id, "value:1")
        self.assertIsNone(selected(result, "KV_VALUE", "key:3").target_id)
        selected_values = [
            item.target_id for item in result.predictions
            if item.relation == "KV_VALUE" and item.target_id is not None
        ]
        self.assertEqual(selected_values, ["value:1", "value:1"])
        self.assertTrue(suppressed(result, "KV_VALUE", "key:1", "value:2", "AMBIGUOUS_TIE"))
        self.assertTrue(suppressed(result, "KV_VALUE", "key:3", "value:1", "NULL_PREFERRED"))

    def test_right_cell_affinity_can_repair_reversed_columns_only_with_rightward_geometry(self) -> None:
        nodes = [
            GraphNode("r:t", "region", kind="TABLE"),
            GraphNode("c:left", "cell", polygon=rect(10, 10, 30, 20), table_id="r:t", row=0, col=1),
            GraphNode("c:right", "cell", polygon=rect(40, 10, 60, 20), table_id="r:t", row=0, col=0),
        ]
        source, target = nodes[1], nodes[2]
        nodes_by_id = {node.id: node for node in nodes}
        self.assertEqual(
            relation_candidate_error("RIGHT_CELL", source, target, nodes_by_id),
            "NOT_LATER_COLUMN",
        )
        self.assertIsNone(relation_candidate_error(
            "RIGHT_CELL", source, target, nodes_by_id,
            allow_right_cell_column_repair=True,
        ))
        result = decode_graph(
            nodes,
            {"RIGHT_CELL": {"c:left": {"c:right": 0.9}}},
            {"RIGHT_CELL": {"c:left": 0.1}},
            allow_right_cell_column_repair=True,
        )
        self.assertEqual(selected(result, "RIGHT_CELL", "c:left").target_id, "c:right")

    def test_right_cell_requires_known_same_table_and_later_same_row(self) -> None:
        nodes = [
            GraphNode("r:t1", "region", kind="TABLE", polygon=rect(0, 0, 100, 100)),
            GraphNode("r:t2", "region", kind="TABLE", polygon=rect(110, 0, 210, 100)),
            GraphNode("c:source", "cell", polygon=rect(10, 10, 30, 20), table_id="r:t1", row=0, col=0),
            GraphNode("c:cross", "cell", polygon=rect(120, 10, 140, 20), table_id="r:t2", row=0, col=1),
            GraphNode("c:row", "cell", polygon=rect(40, 40, 60, 50), table_id="r:t1", row=1, col=1),
            GraphNode("c:back", "cell", polygon=rect(5, 10, 8, 20), table_id="r:t1", row=0, col=1),
            GraphNode("c:right", "cell", polygon=rect(40, 10, 60, 20), table_id="r:t1", row=0, col=1),
            GraphNode("c:unknown", "cell", polygon=rect(70, 10, 90, 20), table_id="r:missing", row=0, col=2),
        ]
        result = decode_graph(
            nodes,
            {"RIGHT_CELL": {"c:source": {
                "c:cross": 0.99,
                "c:row": 0.98,
                "c:back": 0.97,
                "c:right": 0.96,
                "c:unknown": 0.95,
            }}},
            {"RIGHT_CELL": {"c:source": 0.1}},
        )
        self.assertEqual(selected(result, "RIGHT_CELL", "c:source").target_id, "c:right")
        self.assertTrue(suppressed(result, "RIGHT_CELL", "c:source", "c:cross", "CROSS_TABLE"))
        self.assertTrue(suppressed(result, "RIGHT_CELL", "c:source", "c:row", "ROW_MISMATCH"))
        self.assertTrue(suppressed(result, "RIGHT_CELL", "c:source", "c:back", "NOT_HORIZONTAL"))
        self.assertTrue(suppressed(result, "RIGHT_CELL", "c:source", "c:unknown", "TABLE_UNKNOWN"))

    def test_suppressed_diagnostics_can_be_bounded_without_losing_counts(self) -> None:
        nodes = [
            GraphNode(f"l:{index:02d}", "line", polygon=rect(index, 0, index + 1, 1), scope_id="page")
            for index in range(30)
        ]
        ids = [node.id for node in nodes]
        result = decode_graph(
            nodes,
            {"NEXT": {source: {target: 0.1 for target in ids} for source in ids}},
            {"NEXT": {source: 0.9 for source in ids}},
            suppressed_limit=7,
        )
        self.assertEqual(len(result.suppressed), 7)
        self.assertEqual(result.suppressed_count, 900)
        self.assertEqual(dict(result.suppressed_reason_counts), {
            "NULL_PREFERRED": 870,
            "SELF_LINK": 30,
        })

    def test_every_node_and_every_line_survives_even_without_edges(self) -> None:
        nodes = [
            GraphNode("l:b", "line", polygon=rect(0, 20, 4, 24)),
            GraphNode("r:a", "region", kind="REGION", polygon=rect(0, 0, 5, 5)),
            GraphNode("l:a", "line", polygon=rect(0, 0, 4, 4)),
            GraphNode("c:a", "cell", polygon=rect(6, 0, 9, 3)),
        ]
        result = decode_graph(nodes, {}, {})
        self.assertEqual({node.id for node in result.nodes}, {node.id for node in nodes})
        self.assertEqual(result.reading_order, ("l:a", "l:b"))
        self.assertEqual(len(result.reading_order), 2)


if __name__ == "__main__":
    unittest.main()

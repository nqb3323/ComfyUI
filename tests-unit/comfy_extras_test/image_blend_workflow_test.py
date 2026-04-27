import json
import pathlib


WORKFLOW_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tests"
    / "inference"
    / "graphs"
    / "image_blend_channel_mismatch.json"
)


def test_workflow_loads():
    with open(WORKFLOW_PATH) as f:
        graph = json.load(f)
    assert isinstance(graph, dict) and graph, "workflow JSON is empty"


def test_workflow_uses_expected_node_types():
    """The workflow uses a fixed, minimal set of nodes. If any are renamed
    or removed upstream, this test fails fast instead of letting the demo
    bitrot silently."""
    expected = {
        "EmptyImage",
        "SolidMask",
        "JoinImageWithAlpha",
        "ImageBlend",
        "SaveImage",
    }
    with open(WORKFLOW_PATH) as f:
        graph = json.load(f)
    actual = {node["class_type"] for node in graph.values()}
    assert expected.issubset(actual), (
        f"workflow is missing required node types: {expected - actual}"
    )


def test_workflow_exercises_imageblend_with_mismatched_channels():
    """Sanity-check that the workflow actually wires an RGB output and an
    RGBA output into ImageBlend (the CORE-103 case). If someone edits the
    JSON and accidentally breaks this guarantee, the demo loses its point."""
    with open(WORKFLOW_PATH) as f:
        graph = json.load(f)
    blend_nodes = [n for n in graph.values() if n["class_type"] == "ImageBlend"]
    assert len(blend_nodes) == 1, "expected exactly one ImageBlend node"
    blend = blend_nodes[0]
    src1_id, _ = blend["inputs"]["image1"]
    src2_id, _ = blend["inputs"]["image2"]
    types = {graph[src1_id]["class_type"], graph[src2_id]["class_type"]}
    assert "JoinImageWithAlpha" in types, (
        "workflow no longer feeds an RGBA image into ImageBlend"
    )
    assert "EmptyImage" in types, (
        "workflow no longer feeds a plain RGB image into ImageBlend"
    )

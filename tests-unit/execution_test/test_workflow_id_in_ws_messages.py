"""Tests that workflow_id is included alongside prompt_id in WebSocket payloads
emitted by the progress handler and the prompt executor.

Frontend stores extra_data["extra_pnginfo"]["workflow"]["id"] when queueing a
prompt; we propagate that as `workflow_id` on every execution event so a
multi-tab UI can scope progress state by workflow even when terminal
WebSocket frames are dropped.
"""

from unittest.mock import MagicMock

import pytest

from comfy_execution.progress import (
    NodeState,
    ProgressRegistry,
    WebUIProgressHandler,
    reset_progress_state,
    get_progress_state,
)


class _DummyDynPrompt:
    def get_display_node_id(self, node_id):
        return node_id

    def get_parent_node_id(self, node_id):
        return None

    def get_real_node_id(self, node_id):
        return node_id


@pytest.fixture
def server():
    s = MagicMock()
    s.client_id = "client-1"
    return s


def _registry(workflow_id):
    return ProgressRegistry(
        prompt_id="prompt-1",
        dynprompt=_DummyDynPrompt(),
        workflow_id=workflow_id,
    )


class TestProgressStatePayload:
    def test_progress_state_includes_workflow_id(self, server):
        registry = _registry("wf-abc")
        registry.nodes["n1"] = {
            "state": NodeState.Running,
            "value": 1.0,
            "max": 5.0,
        }

        handler = WebUIProgressHandler(server)
        handler.set_registry(registry)
        handler._send_progress_state("prompt-1", registry.nodes)

        server.send_sync.assert_called_once()
        event, payload, sid = server.send_sync.call_args.args
        assert event == "progress_state"
        assert payload["prompt_id"] == "prompt-1"
        assert payload["workflow_id"] == "wf-abc"
        assert payload["nodes"]["n1"]["workflow_id"] == "wf-abc"
        assert payload["nodes"]["n1"]["prompt_id"] == "prompt-1"
        assert sid == "client-1"

    def test_progress_state_workflow_id_none_when_missing(self, server):
        registry = _registry(None)
        registry.nodes["n1"] = {
            "state": NodeState.Running,
            "value": 0.5,
            "max": 1.0,
        }

        handler = WebUIProgressHandler(server)
        handler.set_registry(registry)
        handler._send_progress_state("prompt-1", registry.nodes)

        _, payload, _ = server.send_sync.call_args.args
        assert payload["workflow_id"] is None
        assert payload["nodes"]["n1"]["workflow_id"] is None


class TestProgressRegistryConstruction:
    def test_workflow_id_default_is_none(self):
        registry = ProgressRegistry(
            prompt_id="prompt-1", dynprompt=_DummyDynPrompt()
        )
        assert registry.workflow_id is None

    def test_workflow_id_stored_on_registry(self):
        registry = ProgressRegistry(
            prompt_id="prompt-1",
            dynprompt=_DummyDynPrompt(),
            workflow_id="wf-xyz",
        )
        assert registry.workflow_id == "wf-xyz"


class TestResetProgressState:
    def test_reset_threads_workflow_id(self):
        reset_progress_state("prompt-1", _DummyDynPrompt(), "wf-456")
        assert get_progress_state().workflow_id == "wf-456"

    def test_reset_default_workflow_id_none(self):
        reset_progress_state("prompt-2", _DummyDynPrompt())
        assert get_progress_state().workflow_id is None

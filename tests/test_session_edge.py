# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Edge-case tests for session checkpoint/resume module."""

from __future__ import annotations

import pytest

from seclab_taskflow_agent.session import TaskflowSession, session_dir


class TestSessionEdgeCases:
    """Edge-case tests for TaskflowSession."""

    def test_record_task_empty_tool_results(self, tmp_path, monkeypatch):
        """record_task with empty list sets last_tool_results to []."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="t0", success=True, tool_results=[])
        assert s.last_tool_results == []

    def test_record_task_none_tool_results(self, tmp_path, monkeypatch):
        """record_task with None tool_results defaults to []."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="t0", success=True, tool_results=None)
        assert s.last_tool_results == []

    def test_next_task_index_non_sequential(self, tmp_path, monkeypatch):
        """next_task_index uses max(indices) + 1, even if non-sequential."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="t0", success=True)
        s.record_task(index=5, name="t5", success=True)
        assert s.next_task_index == 6

    def test_save_load_roundtrip_preserves_tool_results(self, tmp_path, monkeypatch):
        """save + load roundtrip preserves last_tool_results."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="t0", success=True, tool_results=["res1", "res2"])
        sid = s.session_id

        loaded = TaskflowSession.load(sid)
        assert loaded.last_tool_results == ["res1", "res2"]
        assert loaded.taskflow_path == "test.flow"

    def test_list_sessions_skips_corrupt_files(self, tmp_path, monkeypatch):
        """list_sessions gracefully skips files with invalid JSON."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)

        # Create a valid session
        s = TaskflowSession(taskflow_path="valid.flow")
        s.save()

        # Write a corrupt file
        sdir = session_dir()
        corrupt_path = sdir / "corrupt.json"
        corrupt_path.write_text("{invalid json!!")

        sessions = TaskflowSession.list_sessions()
        # Only the valid session should be returned
        assert len(sessions) == 1
        assert sessions[0].taskflow_path == "valid.flow"

    def test_multiple_record_task_accumulate(self, tmp_path, monkeypatch):
        """Multiple record_task calls accumulate completed_tasks."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.record_task(index=0, name="t0", success=True, tool_results=["r0"])
        s.record_task(index=1, name="t1", success=False, tool_results=["r1"])
        s.record_task(index=2, name="t2", success=True, tool_results=["r2"])

        assert len(s.completed_tasks) == 3
        assert s.completed_tasks[0].name == "t0"
        assert s.completed_tasks[0].result is True
        assert s.completed_tasks[1].result is False
        assert s.completed_tasks[2].name == "t2"
        # last_tool_results reflects the last call
        assert s.last_tool_results == ["r2"]

    def test_mark_failed_then_save_preserves_error(self, tmp_path, monkeypatch):
        """mark_failed persists the error through save/load."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.mark_failed("something went wrong")
        assert s.error == "something went wrong"

        loaded = TaskflowSession.load(s.session_id)
        assert loaded.error == "something went wrong"
        assert loaded.finished is False

    def test_mark_finished_then_load(self, tmp_path, monkeypatch):
        """mark_finished flag persists through save/load."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        s = TaskflowSession(taskflow_path="test.flow")
        s.mark_finished()

        loaded = TaskflowSession.load(s.session_id)
        assert loaded.finished is True

    def test_load_nonexistent_raises(self, tmp_path, monkeypatch):
        """Loading a non-existent session raises FileNotFoundError."""
        monkeypatch.setattr("seclab_taskflow_agent.session._data_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="No session checkpoint found"):
            TaskflowSession.load("nonexistent-id")

from __future__ import annotations

import pytest

from lifemate.data import DatasetError
from tests.support import make_dataset


def test_catalog_is_restricted_to_installed_tools_plus_system_tool(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    sample = dataset.get_sample("public_test_1")
    restricted = dataset.load_catalog().restricted_to(sample.allowed_tool_names)

    assert set(restricted) == {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    assert "sms.send" not in restricted


def test_train_split_is_refused_without_reading_it(tmp_path) -> None:
    with pytest.raises(DatasetError, match="out of scope"):
        type(make_dataset(tmp_path))(tmp_path, "train")

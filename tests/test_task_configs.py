"""Tests for lm-eval YAML task configurations."""

import pathlib

import pytest
import yaml

CUSTOM_TASKS_DIR = pathlib.Path(__file__).resolve().parent.parent / "custom_tasks"

TASK_DIRS = ["esbbq", "cabbq", "veritasQA"]

# Directory name → expected task-name prefix
DIR_PREFIX = {
    "esbbq": "esbbq_",
    "cabbq": "cabbq_",
    "veritasQA": "veritas_",
}

REQUIRED_FIELDS = ["task", "dataset_path", "output_type", "test_split", "metric_list"]

# Columns every esbbq / cabbq row must contain
BBQ_REQUIRED_COLUMNS = {"context", "question", "ans0", "ans1", "ans2", "label"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_function_constructor(loader: type) -> type:
    """Return a YAML loader that treats !function tags as plain strings."""
    loader.add_constructor(
        "!function",
        lambda loader, node: loader.construct_scalar(node),
    )
    return loader


class _SafeLoaderWithFunction(yaml.SafeLoader):
    pass


_add_function_constructor(_SafeLoaderWithFunction)


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_SafeLoaderWithFunction)


def _collect_yaml_files() -> list[pathlib.Path]:
    """Return all YAML config files across all task directories."""
    files = []
    for d in TASK_DIRS:
        task_dir = CUSTOM_TASKS_DIR / d
        if task_dir.is_dir():
            files.extend(sorted(task_dir.glob("*.yaml")))
    return files


ALL_YAML_FILES = _collect_yaml_files()


def _yaml_id(path: pathlib.Path) -> str:
    """Readable test ID: 'esbbq/esbbq_age.yaml'."""
    return f"{path.parent.name}/{path.name}"


# ---------------------------------------------------------------------------
# Offline tests (always run)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaml_path", ALL_YAML_FILES, ids=_yaml_id)
class TestTaskConfigOffline:
    """Structural checks that require no network access."""

    def test_parses_without_error(self, yaml_path: pathlib.Path):
        cfg = _load_yaml(yaml_path)
        assert isinstance(cfg, dict), f"{yaml_path.name} did not parse to a dict"

    def test_has_required_fields(self, yaml_path: pathlib.Path):
        cfg = _load_yaml(yaml_path)
        for field in REQUIRED_FIELDS:
            assert field in cfg, f"{yaml_path.name} missing required field '{field}'"

    def test_filename_matches_task(self, yaml_path: pathlib.Path):
        """Catches copy-paste errors where `task:` doesn't match the filename."""
        cfg = _load_yaml(yaml_path)
        expected_task = yaml_path.stem
        assert cfg["task"] == expected_task, (
            f"{yaml_path.name}: task field is '{cfg['task']}', "
            f"expected '{expected_task}'"
        )

    def test_task_prefix_matches_directory(self, yaml_path: pathlib.Path):
        cfg = _load_yaml(yaml_path)
        dir_name = yaml_path.parent.name
        prefix = DIR_PREFIX[dir_name]
        assert cfg["task"].startswith(prefix), (
            f"{yaml_path.name}: task '{cfg['task']}' does not start with "
            f"expected prefix '{prefix}' for directory '{dir_name}'"
        )

    def test_dataset_name_present_for_bbq(self, yaml_path: pathlib.Path):
        """esbbq and cabbq configs must specify a HuggingFace dataset subset."""
        dir_name = yaml_path.parent.name
        if dir_name not in ("esbbq", "cabbq"):
            pytest.skip("only applies to esbbq/cabbq configs")
        cfg = _load_yaml(yaml_path)
        assert "dataset_name" in cfg and cfg["dataset_name"], (
            f"{yaml_path.name}: missing or empty 'dataset_name'"
        )


# ---------------------------------------------------------------------------
# Network tests (require HuggingFace access)
# ---------------------------------------------------------------------------

def _collect_dataset_combos() -> list[tuple[str, str | None, str]]:
    """Return unique (dataset_path, dataset_name, dir_name) tuples."""
    seen = set()
    combos = []
    for path in ALL_YAML_FILES:
        cfg = _load_yaml(path)
        key = (cfg["dataset_path"], cfg.get("dataset_name"))
        if key not in seen:
            seen.add(key)
            combos.append((cfg["dataset_path"], cfg.get("dataset_name"), path.parent.name))
    return combos


DATASET_COMBOS = _collect_dataset_combos()
DATASET_COMBO_IDS = [
    f"{path}/{name}" if name else path for path, name, _ in DATASET_COMBOS
]


@pytest.mark.network
@pytest.mark.parametrize(
    "dataset_path,dataset_name,dir_name",
    DATASET_COMBOS,
    ids=DATASET_COMBO_IDS,
)
class TestDatasetAccess:
    """Tests that load data from HuggingFace (requires network)."""

    def _load_dataset(self, dataset_path: str, dataset_name: str | None):
        from datasets import load_dataset

        kwargs = {"path": dataset_path, "split": "test"}
        if dataset_name:
            kwargs["name"] = dataset_name
        return load_dataset(**kwargs)

    def test_dataset_loads(self, dataset_path, dataset_name, dir_name):
        ds = self._load_dataset(dataset_path, dataset_name)
        assert len(ds) > 0, "dataset is empty"

    def test_test_split_exists(self, dataset_path, dataset_name, dir_name):
        # load_dataset with split="test" raises if split doesn't exist,
        # so reaching here means the split is valid
        ds = self._load_dataset(dataset_path, dataset_name)
        assert ds is not None

    def test_required_columns_for_bbq(self, dataset_path, dataset_name, dir_name):
        if dir_name not in ("esbbq", "cabbq"):
            pytest.skip("column check only applies to esbbq/cabbq")
        ds = self._load_dataset(dataset_path, dataset_name)
        missing = BBQ_REQUIRED_COLUMNS - set(ds.column_names)
        assert not missing, f"missing columns: {missing}"

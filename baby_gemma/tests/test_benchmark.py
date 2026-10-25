from ..benchmark import BenchmarkHarness


def benchmark_result(model, loss):
    return {
        "Model": model,
        "Positional Emb": "None",
        "Attention Type": "Single-Head",
        "Norm": "RMSNorm",
        "Parameters": "1,000",
        "Train Loss": loss,
        "Best Val Loss": loss,
        "Final Val Loss": loss,
        "Final PPL": 10.0,
        "Steps/sec": 1.0,
        "Completion": "sample text",
    }


def test_summary_appends_new_models_and_replaces_matching_models(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Experiments\n", encoding="utf-8")

    first = BenchmarkHarness(None, [], [])
    first.results = [benchmark_result("Baseline", 5.0)]
    first.summary(readme_path)

    second = BenchmarkHarness(None, [], [])
    second.results = [
        benchmark_result("Baseline", 4.0),
        benchmark_result("Positional", 3.5),
    ]
    merged = second.summary(readme_path)

    assert merged["Model"].tolist() == ["Baseline", "Positional"]
    assert float(merged.loc[merged["Model"] == "Baseline", "Best Val Loss"].iloc[0]) == 4.0

    content = readme_path.read_text(encoding="utf-8")
    assert content.count("<!-- BENCHMARK_START -->") == 1
    assert content.count("Baseline") == 1
    model_rows = [
        line for line in content.splitlines()
        if line.startswith("| Baseline") or line.startswith("| Positional ")
    ]
    assert model_rows[0].startswith("| Baseline")
    assert model_rows[1].startswith("| Positional ")


def test_summary_keeps_multiline_completions_in_one_table_row(tmp_path):
    readme_path = tmp_path / "README.md"
    result = benchmark_result("Multiline", 3.0)
    result["Completion"] = "first line\nsecond line\nthird | line"

    harness = BenchmarkHarness(None, [], [])
    harness.results = [result]
    harness.summary(readme_path)

    content = readme_path.read_text(encoding="utf-8")
    model_rows = [
        line for line in content.splitlines() if line.startswith("| Multiline")
    ]
    assert len(model_rows) == 1
    assert "first line second line third ¦ line" in model_rows[0]

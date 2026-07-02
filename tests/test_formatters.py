from src.data.formatters import format_grpo_example, format_sft_example


def test_format_sft_example_has_gold_answer():
    row = format_sft_example("2+2?", "Work\n#### 4")
    assert row["gold_answer"] == "4"
    assert "Question: 2+2?" in row["prompt"]


def test_format_grpo_example_prompt_only():
    row = format_grpo_example("3+5?", "Steps\n#### 8")
    assert row["gold_answer"] == "8"
    assert "completion" not in row

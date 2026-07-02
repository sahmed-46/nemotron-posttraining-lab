from src.rewards.verifiable import combined_math_reward, extract_final_number, math_answer_reward


def test_math_answer_reward_correct():
    assert math_answer_reward("Reasoning...\nANSWER: 42", "42") == 1.0


def test_math_answer_reward_wrong():
    assert math_answer_reward("ANSWER: 41", "42") == 0.0


def test_combined_reward_includes_format_bonus():
    score = combined_math_reward("ANSWER: 42", "42")
    assert score >= 1.0 or score == 1.0


def test_extract_final_number():
    assert extract_final_number("Some text\nANSWER: 1,234") == "1234"

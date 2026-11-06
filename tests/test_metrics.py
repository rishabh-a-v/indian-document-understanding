from idu.metrics import (anls, bootstrap_ci, exact, is_empty, levenshtein, normalize, numbers,
                         score_text, token_f1)


def test_normalize_digits_punct_and_decimals():
    assert normalize("२७ अक्टूबर, २०२६") == "27 अक्टूबर 2026"
    assert normalize("का.आ. 5830(अ)") == "का आ 5830 अ"
    assert normalize("0.2095") == "0.2095"          # decimal point kept
    assert normalize("  A  b ") == "a b"
    assert normalize(None) == ""


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


def test_exact_with_aliases_and_numerals():
    assert exact("२७/१०/२०२६", ["27 अक्टूबर, 2026", "27/10/2026"]) == 1.0
    assert exact("गुजरात", "जामनगर") == 0.0


def test_anls_threshold():
    assert anls("0.2095", "0.2095") == 1.0
    assert 0.8 < anls("सनाला गांव", "सनाला गाँव") < 1.0   # small char error keeps partial credit
    assert anls("abc", "xyz") == 0.0                        # below threshold -> 0


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert abs(token_f1("a b", "a b c d") - 2 * 1.0 * 0.5 / 1.5) < 1e-9
    assert token_f1("", "a") == 0.0


def test_numbers_and_empty():
    assert numbers("क्षेत्रफल ०.२०९५ हेक्टेयर, 12") == ["0.2095", "12"]
    assert is_empty("NOT FOUND") and is_empty(" null ") and is_empty("")
    assert not is_empty("गुजरात")


def test_score_text_and_bootstrap():
    s = score_text("27 अक्टूबर 2026", ["27 अक्टूबर, 2026"])
    assert s["em"] == 1.0 and s["anls"] == 1.0 and s["f1"] == 1.0
    lo, hi = bootstrap_ci([1, 1, 0, 0, 1, 0, 1, 1])
    assert 0.0 <= lo <= hi <= 1.0
    assert bootstrap_ci([]) == (0.0, 0.0)

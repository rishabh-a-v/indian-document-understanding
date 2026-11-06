from idu.failure import TAGS, tag_failure


def test_correct_has_no_tags():
    assert tag_failure("0.2095", "0.2095") == []


def test_empty():
    assert tag_failure("NOT FOUND", "गुजरात") == ["EMPTY_OR_NOT_FOUND"]
    assert tag_failure("", "गुजरात")[0] == "EMPTY_OR_NOT_FOUND"


def test_retrieval_miss_comes_first():
    tags = tag_failure("NOT FOUND", "0.2095", retrieved_pages=[0, 3], gold_page=1)
    assert tags[0] == "RETRIEVAL_MISS" and "EMPTY_OR_NOT_FOUND" in tags
    assert "RETRIEVAL_MISS" not in tag_failure("NOT FOUND", "0.2095", retrieved_pages=[1], gold_page=1)


def test_wrong_language():
    assert "WRONG_LANGUAGE_OR_SCRIPT" in tag_failure("Gujarat", "गुजरात")
    assert "WRONG_LANGUAGE_OR_SCRIPT" not in tag_failure("गुजरात राज्य", "गुजरात")


def test_numeric_tags():
    assert tag_failure("0.2059", "0.2095")[0] == "NUMERIC_WRONG"
    assert tag_failure("सरकारी", "0.2095")[0] == "NUMERIC_MISSING"
    # number present with extra words is NOT a numeric error
    t = tag_failure("0.2095 हेक्टेयर क्षेत्रफल है", "0.2095")
    assert "NUMERIC_WRONG" not in t and "NUMERIC_MISSING" not in t and "EXTRA_TEXT" in t


def test_near_miss_partial_wrong():
    assert tag_failure("नितिनभाई दामजीभाई सोरठीया", "नितिनभाई दामजीभाई सोरठिया")[0] == "NEAR_MISS_CHAR_ERROR"
    assert tag_failure("नितिनभाई", "नितिनभाई दामजीभाई सोरठिया")[0] == "PARTIAL_ANSWER"
    assert tag_failure("रमेशभाई वशरामभाई पांभर", "नितिनभाई दामजीभाई सोरठिया") == ["WRONG_ANSWER"]


def test_all_tags_documented():
    for t in ["RETRIEVAL_MISS", "EMPTY_OR_NOT_FOUND", "WRONG_LANGUAGE_OR_SCRIPT", "NUMERIC_MISSING",
              "NUMERIC_WRONG", "NEAR_MISS_CHAR_ERROR", "PARTIAL_ANSWER", "EXTRA_TEXT", "WRONG_ANSWER"]:
        assert t in TAGS

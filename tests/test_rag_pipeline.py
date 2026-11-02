import pytest

from modules.rag_pipeline import (
    validate_explanation,
    create_fallback_explanation
)


def test_valid_explanation():

    result = validate_explanation(
        "Toy Story (1995) belongs to Animation Children Comedy.",
        "Toy Story (1995)",
        "Animation Children Comedy"
    )

    assert result is True


def test_reject_unsupported_plot():

    result = validate_explanation(
        "The story follows a young boy and his toys.",
        "Toy Story (1995)",
        "Animation Children Comedy"
    )

    assert result is False


def test_reject_empty_explanation():

    result = validate_explanation(
        "",
        "Toy Story (1995)",
        "Animation Children Comedy"
    )

    assert result is False


def test_reject_wrong_movie():

    result = validate_explanation(
        "Batman belongs to Action Adventure.",
        "Toy Story (1995)",
        "Animation Children Comedy"
    )

    assert result is False


def test_fallback_explanation():

    movie = {
        "title": "Toy Story (1995)",
        "genre_text": "Animation Children Comedy",
        "average_rating": 3.88
    }

    result = create_fallback_explanation(
        "I enjoy comedy and animation movies",
        movie
    )

    assert "Toy Story (1995)" in result
    assert "3.88/5" in result
    assert "Animation Children Comedy" in result


def test_reject_excessively_long_explanation():

    result = validate_explanation(
        "Toy Story " + ("wonderful " * 150),
        "Toy Story (1995)",
        "Animation Children Comedy"
    )

    assert result is False
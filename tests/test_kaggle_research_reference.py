import pytest

from scripts.kaggle_research_job import canonical_kernel_reference


@pytest.mark.parametrize('suffix', ['', '?scriptVersionId=123'])
def test_push_response_title_slug_is_used_for_monitoring(suffix):
    result = canonical_kernel_reference('https://www.kaggle.com/code/nmavros/title-derived-slug'+suffix,
                                        'nmavros/requested-slug', 'nmavros')
    assert result == 'nmavros/title-derived-slug'


def test_missing_response_url_preserves_requested_reference():
    assert canonical_kernel_reference(None, 'nmavros/requested', 'nmavros') == 'nmavros/requested'


@pytest.mark.parametrize('url', ['https://www.kaggle.com/code/someone-else/kernel',
                                'https://untrusted.example/code/nmavros/kernel'])
def test_unexpected_owner_or_host_is_rejected(url):
    with pytest.raises(ValueError):
        canonical_kernel_reference(url, 'nmavros/requested', 'nmavros')

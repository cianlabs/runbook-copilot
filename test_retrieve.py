import pytest

import retrieve


def test_chunking_produces_multiple_pieces():
    chunks = retrieve.build_chunks()
    assert len(chunks) > 1
    assert all(c["source"] == "sample_runbook.txt" for c in chunks)


def test_chunk_sizes_are_reasonable():
    for c in retrieve.build_chunks():
        assert len(c["text"]) <= 500


def test_chunks_overlap():
    text = "word " * 400
    chunks = retrieve.chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert chunks[0][-20:].strip() in chunks[1]


def test_relevant_query_finds_right_section():
    results = retrieve.search("database connection refused too many connections")
    assert 1 <= len(results) <= 3
    assert "connection" in results[0]["text"].lower()
    assert results[0]["score"] > 0.1
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_certificate_query_finds_certificate_section():
    results = retrieve.search("certificate expiring renew")
    assert "certificate" in results[0]["text"].lower()


def test_irrelevant_query_returns_nothing():
    assert retrieve.search("chocolate banana pancake recipe") == []


def test_empty_query_is_a_named_failure():
    with pytest.raises(retrieve.RetrieveError) as e:
        retrieve.search("   ")
    assert e.value.name == "EMPTY_QUERY"


def test_missing_folder_is_a_named_failure(tmp_path):
    with pytest.raises(retrieve.RetrieveError) as e:
        retrieve.load_runbooks(tmp_path / "nope")
    assert e.value.name == "NO_RUNBOOKS"


def test_bad_settings_is_a_named_failure():
    with pytest.raises(retrieve.RetrieveError) as e:
        retrieve.chunk_text("hello", chunk_size=100, overlap=100)
    assert e.value.name == "BAD_SETTINGS"


def test_no_chunk_mixes_two_sections():
    runbook = (retrieve.RUNBOOK_DIR / "sample_runbook.txt").read_text()
    sections = {h: " ".join(b.split()) for h, b in retrieve.split_sections(runbook)}
    headers = list(sections)
    assert len(headers) == 4
    for c in retrieve.build_chunks():
        # the chunk must come entirely from its own section's text...
        assert c["text"] in sections[c["section"]]
        # ...and must not contain any other section's header
        for h in headers:
            if h != c["section"]:
                assert h not in c["text"]
        assert "# " not in c["text"]


def test_every_section_is_covered():
    sections = {c["section"] for c in retrieve.build_chunks()}
    assert sections == {
        "Database connection errors",
        "High CPU on web servers",
        "Disk space almost full",
        "Certificate expiry",
    }


def test_cpu_query_top_result_is_high_cpu_section():
    results = retrieve.search("web server CPU is high")
    assert results[0]["section"] == "High CPU on web servers"

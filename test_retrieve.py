import pytest

import retrieve

# Every runbook file and the sections it must contain.
EXPECTED_SECTIONS = {
    "sample_runbook.txt": {
        "Database connection errors",
        "High CPU on web servers",
        "Disk space almost full",
        "Certificate expiry",
    },
    "network_connectivity.txt": {
        "DNS lookup failures",
        "Packet loss and high latency",
        "Firewall blocks a connection",
        "Network interface is down",
    },
    "memory_leaks.txt": {
        "Out of memory kills",
        "High swap usage",
        "Process memory keeps growing",
        "Container exceeds its memory limit",
    },
    "deployment_failures.txt": {
        "Deploy fails during release",
        "Roll back a bad release",
        "Health check keeps failing",
        "Database change breaks the new release",
    },
}


def test_chunking_produces_multiple_pieces():
    chunks = retrieve.build_chunks()
    assert len(chunks) > 1
    assert {c["source"] for c in chunks} == set(EXPECTED_SECTIONS)


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
    for name, expected in EXPECTED_SECTIONS.items():
        runbook = (retrieve.RUNBOOK_DIR / name).read_text()
        sections = {h: " ".join(b.split()) for h, b in retrieve.split_sections(runbook)}
        headers = list(sections)
        assert set(headers) == expected
        for c in retrieve.build_chunks():
            if c["source"] != name:
                continue
            # the chunk must come entirely from its own section's text...
            assert c["text"] in sections[c["section"]]
            # ...and must not contain any other section's header
            for h in headers:
                if h != c["section"]:
                    assert h not in c["text"]
            assert "# " not in c["text"]


def test_every_section_is_covered():
    found = {}
    for c in retrieve.build_chunks():
        found.setdefault(c["source"], set()).add(c["section"])
    assert found == EXPECTED_SECTIONS


def test_there_are_at_least_four_runbook_files():
    assert len(list(retrieve.RUNBOOK_DIR.glob("*.txt"))) >= 4


@pytest.mark.parametrize("query, source, section", [
    ("DNS name not resolved", "network_connectivity.txt", "DNS lookup failures"),
    ("kernel out of memory killer ended my process", "memory_leaks.txt", "Out of memory kills"),
    ("load balancer marks new servers unhealthy after deploy", "deployment_failures.txt", "Health check keeps failing"),
    ("roll back the last release", "deployment_failures.txt", "Roll back a bad release"),
])
def test_queries_find_the_new_runbooks(query, source, section):
    results = retrieve.search(query)
    assert results, f"no results for {query!r}"
    assert results[0]["source"] == source
    assert results[0]["section"] == section


def test_cpu_query_top_result_is_high_cpu_section():
    results = retrieve.search("web server CPU is high")
    assert results[0]["section"] == "High CPU on web servers"


def test_a_tiny_last_chunk_is_merged_into_the_previous_one():
    # 440 characters: a plain split would leave a short fragment at the end.
    text = " ".join(f"w{i:03d}" for i in range(88))
    plain = retrieve.chunk_text(text, chunk_size=400, overlap=80, min_tail=0)
    merged = retrieve.chunk_text(text, chunk_size=400, overlap=80)
    assert len(plain[-1]) < retrieve.MIN_TAIL  # the problem exists without merging
    assert len(merged) == len(plain) - 1
    assert merged[-1].endswith(text.split()[-1])  # nothing was lost at the end
    assert len(merged[-1]) <= 500


def test_no_chunk_in_the_real_runbooks_is_a_tiny_fragment():
    by_section = {}
    for c in retrieve.build_chunks():
        by_section.setdefault((c["source"], c["section"]), []).append(c["text"])
    for key, pieces in by_section.items():
        if len(pieces) > 1:
            assert all(len(p) >= retrieve.MIN_TAIL for p in pieces), key


def test_dns_query_returns_the_full_section_first():
    results = retrieve.search("DNS name not resolved")
    assert results[0]["section"] == "DNS lookup failures"
    assert results[0]["text"].startswith("Symptom:")

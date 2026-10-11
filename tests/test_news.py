import json

from cfbr.news.collect import Archive, parse_feed
from cfbr.news.policy import ai_opt_out

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>QB out for Saturday</title><link>https://ex.com/a</link><guid>g1</guid>
<description>Starter ruled out.</description><pubDate>Sat, 10 Oct 2026 15:00:00 GMT</pubDate></item>
<item><title>No guid</title><link>https://ex.com/b</link></item></channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>x</title>
<entry><id>tag:1</id><title>Injury update</title><link href="https://ex.com/c"/>
<updated>2026-10-10T12:00:00Z</updated><summary>Questionable.</summary></entry></feed>"""


def test_parse_rss_and_atom():
    r = parse_feed(RSS)
    assert [i["guid"] for i in r] == ["g1", "https://ex.com/b"]  # falls back to link
    assert r[0]["published_at"] == "2026-10-10T15:00:00+00:00"
    a = parse_feed(ATOM)
    assert a[0]["guid"] == "tag:1" and a[0]["url"] == "https://ex.com/c"
    assert a[0]["published_at"] == "2026-10-10T12:00:00+00:00"


def test_ai_opt_out_detection():
    assert ai_opt_out("User-agent: GPTBot\nDisallow: /\nUser-agent: *\nAllow: /") == ["gptbot"]
    assert ai_opt_out("User-agent: anthropic-ai\nDisallow: /") == ["anthropic-ai"]
    assert ai_opt_out("User-agent: Bytespider   # ByteDance\nDisallow: /") == ["bytespider"]
    # Blocking only Common Crawl is not treated as an AI opt-out.
    assert ai_opt_out("User-agent: ccbot\nDisallow: /\nUser-agent: *\nAllow: /") == []


def test_archive_dedupes_items_and_is_append_only(tmp_path):
    from datetime import UTC, datetime

    arc = Archive(tmp_path)
    t = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
    path = arc.store("src", "https://ex.com/feed", t, RSS, "xml")
    assert (tmp_path / path).read_bytes() == RSS
    row = {"source": "src", "guid": "g1", "title": "x", "first_seen_at": t.isoformat()}
    assert arc.add_item(row) is True
    assert Archive(tmp_path).add_item(dict(row, first_seen_at="later")) is False  # survives restart
    lines = (tmp_path / "items.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["first_seen_at"] == t.isoformat()

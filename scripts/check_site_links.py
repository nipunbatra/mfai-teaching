#!/usr/bin/env python3
"""Check that local links and assets in a rendered Quarto site exist."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urldefrag, urlsplit
from urllib.request import Request, urlopen


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.values.append(value)


def target_for(site: Path, page: Path, reference: str) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith(("#", "mailto:", "javascript:", "data:")):
        return None

    path_text = unquote(parsed.path)
    if not path_text:
        return page
    target = site / path_text.lstrip("/") if path_text.startswith("/") else page.parent / path_text

    if path_text.endswith("/"):
        return target / "index.html"
    if target.exists():
        return target
    if not target.suffix:
        html_target = target.with_suffix(".html")
        if html_target.exists():
            return html_target
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", default="_site", type=Path)
    parser.add_argument(
        "--external", action="store_true", help="also probe unique HTTP(S) destinations"
    )
    args = parser.parse_args()

    site = args.site.resolve()
    pages = sorted(site.rglob("*.html"))
    failures: list[tuple[Path, str, Path]] = []
    external: set[str] = set()
    checked = 0

    for page in pages:
        references = References()
        references.feed(page.read_text(encoding="utf-8"))
        for reference in references.values:
            if reference.startswith(("http://", "https://")):
                external.add(urldefrag(reference).url)
            target = target_for(site, page, reference)
            if target is None:
                continue
            checked += 1
            if not target.exists():
                failures.append((page.relative_to(site), reference, target))

    if failures:
        for page, reference, target in failures:
            print(f"FAIL {page}: {reference} -> {target}")
        print(f"{len(failures)} missing local targets across {len(pages)} pages")
        return 1

    print(f"PASS {checked} local references across {len(pages)} HTML pages")

    if args.external:
        def probe(url: str) -> tuple[str, int | None, str | None]:
            headers = {"User-Agent": "mfai-course-link-check/1.0"}
            for method in ("HEAD", "GET"):
                try:
                    with urlopen(Request(url, method=method, headers=headers), timeout=20) as response:
                        return url, response.status, None
                except HTTPError as error:
                    if error.code < 500 and error.code not in {404, 410}:
                        return url, error.code, None
                    if method == "GET" or error.code in {404, 410}:
                        return url, error.code, str(error)
                except URLError as error:
                    if method == "GET":
                        return url, None, str(error.reason)
            return url, None, "no response"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = sorted(pool.map(probe, external))
        broken = [result for result in results if result[2] is not None]
        for url, status, error in broken:
            print(f"FAIL external {status or '-'} {url}: {error}")
        if broken:
            print(f"{len(broken)} failed external destinations out of {len(results)}")
            return 1
        print(f"PASS {len(results)} external destinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

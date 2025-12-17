"""Minimal crawler for Jimdo help center (DE) using LangChain's SitemapLoader.

Fetches up to 10 documents from the German sitemap and writes them to
`jimdo_docs.csv` with URL and page content columns.
"""

from langchain_community.document_loaders.sitemap import SitemapLoader
import json
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

SITEMAP_URL = "https://help.jimdo.com/hc/sitemap.xml"


def _ensure_soup(content: object) -> BeautifulSoup:
    """Coerce loader content into a BeautifulSoup object safely."""
    if isinstance(content, BeautifulSoup):
        return content
    if isinstance(content, (str, bytes)):
        return BeautifulSoup(content, "html.parser")
    if hasattr(content, "text"):
        try:
            return BeautifulSoup(content.text, "html.parser")
        except Exception:
            pass
    # Fallback: stringified content
    return BeautifulSoup(str(content), "html.parser")


def extract_breadcrumbs(content: object) -> list[str]:
    """Try to extract breadcrumbs from the page HTML."""
    soup = _ensure_soup(content)

    # Prefer explicit breadcrumb containers
    candidates = soup.select(
        "nav[aria-label*=bread], nav[class*=bread], ol[class*=bread], ul[class*=bread], "
        "nav[id*=bread], ol[id*=bread], ul[id*=bread]"
    )
    crumbs: list[str] = []
    for container in candidates:
        items = container.find_all(["li", "a"])
        crumbs = [item.get_text(strip=True) for item in items if item.get_text(strip=True)]
        if crumbs:
            break

    # Deduplicate while preserving order
    deduped: list[str] = []
    seen = set()
    for crumb in crumbs:
        if crumb not in seen:
            deduped.append(crumb)
            seen.add(crumb)
    crumbs = deduped

    # Fallback: use header tags if no breadcrumb container found
    if not crumbs:
        header = soup.find("h1")
        if header:
            crumbs = [header.get_text(strip=True)]

    return crumbs


def build_meta(el: dict, html: str) -> dict:
    """Create enriched metadata including breadcrumbs."""
    meta = {
        "source": el.get("loc"),
        "loc": el.get("loc"),
    }
    if "lastmod" in el:
        meta["lastmod"] = el.get("lastmod")

    breadcrumbs = extract_breadcrumbs(html)
    if breadcrumbs:
        meta["breadcrumbs"] = breadcrumbs
        meta["breadcrumbs_path"] = " - ".join(breadcrumbs)

    return meta


def main() -> None:
    max_sites: int | None = None  # set to an int to limit, or None to fetch all

    # Initialize loader for German URLs only
    loader = SitemapLoader(
        web_path=SITEMAP_URL,
        filter_urls=["https://help.jimdo.com/hc/de/"],
        meta_function=build_meta,
    )
    loader.requests_per_second = 1  # Crawl politely

    # Scrape lazily and stop after first 10 documents for a quick test run
    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)
        if max_sites and len(docs) >= max_sites:
            break
    print(f"Geladene Dokumente insgesamt (Testlauf): {len(docs)}")

    rows = []
    for doc in docs:
        rows.append(
            {
                "url": doc.metadata.get("source"),
                "page_content": doc.page_content,
                "metadata": json.dumps(doc.metadata, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("jimdo_docs.csv", index=False)
    print("Gespeichert nach jimdo_docs.csv")

    # Quick preview of first two entries
    for row in rows[:2]:
        preview = row["page_content"].strip().replace("\n", " ")
        preview = (preview[:200] + "...") if len(preview) > 200 else preview
        print("\n--- Sample ---")
        print(f"URL: {row['url']}")
        print(f"Metadata: {row['metadata']}")
        print(f"Content preview: {preview}")


if __name__ == "__main__":
    main()

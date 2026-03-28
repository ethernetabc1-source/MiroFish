"""
Auto News Source Collector
Searches, fetches, and selects diverse news articles for MiroFish simulations.
"""

import uuid
import re
import json
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Dict
from urllib.parse import quote_plus

import httpx

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.news_collector')


class NewsCollectorService:

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    GNEWS_RSS_URL = "https://news.google.com/rss/search"
    FETCH_TIMEOUT = 8
    FETCH_WORKERS = 5

    def __init__(self):
        self.llm = LLMClient()

    # ------------------------------------------------------------------ #
    #  Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def collect(
        self,
        question: str,
        max_sources: int = 20,
        task_callback=None,
    ) -> Dict:
        """
        Full pipeline: query extraction → search → fetch → select.

        Args:
            question: The user's prediction question.
            max_sources: Maximum number of articles to return.
            task_callback: Optional callable(progress, message) for progress updates.

        Returns:
            {"articles": [...], "queries_used": [...]}
        """
        def _cb(p, m):
            if task_callback:
                task_callback(p, m)
            logger.info(f"[{p}%] {m}")

        _cb(10, "Extracting search queries from your question...")
        queries = self._extract_search_queries(question)
        _cb(20, f"Searching news with {len(queries)} queries...")

        raw_articles = []
        for q in queries:
            raw_articles.extend(self._search_google_news_rss(q))
            raw_articles.extend(self._search_gdelt(q))

        raw_articles = self._deduplicate(raw_articles)
        logger.info(f"Found {len(raw_articles)} unique articles before fetching")

        _cb(30, f"Fetching full text for {len(raw_articles)} articles...")
        articles_with_text = self._fetch_all_texts(raw_articles, progress_callback=_cb)

        _cb(85, f"Selecting best {max_sources} diverse sources...")
        selected = self._select_diverse_sources(articles_with_text, n=max_sources)

        if not selected:
            raise RuntimeError(
                "No articles could be fetched. This usually means outbound access to "
                "news sites is blocked by a network firewall or proxy. "
                "Please run MiroFish in an environment with open internet access."
            )

        _cb(100, f"Done. Selected {len(selected)} articles.")
        return {"articles": selected, "queries_used": queries}

    # ------------------------------------------------------------------ #
    #  Query extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_search_queries(self, question: str) -> List[str]:
        """Use Claude to extract 3-5 search queries from the prediction question."""
        prompt = f"""You are a news research assistant. Given a prediction question, extract 3-5 specific search queries to find relevant news articles.

Prediction question: {question}

Requirements:
- Queries should cover the main topic and related subtopics
- Each query should be 2-5 words, suitable for a news search engine
- Cover different angles: the main event, key actors, background context
- Do NOT include question marks or special characters

Respond with a JSON object: {{"queries": ["query1", "query2", "query3"]}}"""

        try:
            result = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.3)
            queries = result.get("queries", [])
            if isinstance(queries, list) and queries:
                return [str(q).strip() for q in queries[:5] if q]
        except Exception as e:
            logger.warning(f"Query extraction failed: {e}")

        # Fallback: use the question itself trimmed
        words = question.split()[:6]
        return [" ".join(words)]

    # ------------------------------------------------------------------ #
    #  News search                                                          #
    # ------------------------------------------------------------------ #

    def _search_google_news_rss(self, query: str) -> List[Dict]:
        """Search Google News RSS feed."""
        articles = []
        try:
            url = f"{self.GNEWS_RSS_URL}?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
            resp = httpx.get(url, timeout=10, follow_redirects=True)
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            ns = {"media": "http://search.yahoo.com/mrss/"}

            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub_date = item.findtext("pubDate", "").strip()
                description = item.findtext("description", "").strip()
                # Source name from <source> tag
                source_el = item.find("source")
                source = source_el.text.strip() if source_el is not None else self._domain_from_url(link)

                # Clean HTML from description
                snippet = re.sub(r"<[^>]+>", "", description).strip()[:300]

                if title and link:
                    articles.append({
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "url": link,
                        "source": source,
                        "published": pub_date,
                        "snippet": snippet,
                        "text": "",
                        "word_count": 0,
                        "category": "",
                        "selection_reason": "",
                    })
        except Exception as e:
            logger.warning(f"Google News RSS search failed for '{query}': {e}")

        return articles

    def _search_gdelt(self, query: str) -> List[Dict]:
        """Search GDELT document API."""
        articles = []
        try:
            params = {
                "query": query,
                "mode": "artlist",
                "maxrecords": "25",
                "format": "json",
                "timespan": "7d",
            }
            resp = httpx.get(self.GDELT_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for art in data.get("articles", []):
                title = (art.get("title") or "").strip()
                url = (art.get("url") or "").strip()
                source = (art.get("domain") or self._domain_from_url(url)).strip()
                published = (art.get("seendate") or "").strip()
                snippet = ""  # GDELT artlist doesn't include snippets

                if title and url:
                    articles.append({
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "url": url,
                        "source": source,
                        "published": published,
                        "snippet": snippet,
                        "text": "",
                        "word_count": 0,
                        "category": "",
                        "selection_reason": "",
                    })
        except Exception as e:
            logger.warning(f"GDELT search failed for '{query}': {e}")

        return articles

    # ------------------------------------------------------------------ #
    #  Deduplication                                                        #
    # ------------------------------------------------------------------ #

    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate URLs, keeping first occurrence."""
        seen_urls = set()
        seen_titles = set()
        unique = []
        for art in articles:
            url_key = art["url"].split("?")[0].rstrip("/")
            title_key = art["title"].lower().strip()[:60]
            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique.append(art)
        return unique

    # ------------------------------------------------------------------ #
    #  Article text fetching                                                #
    # ------------------------------------------------------------------ #

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch and extract clean text from an article URL using trafilatura."""
        try:
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )
                if text and len(text) >= 100:
                    return text.strip()
        except Exception as e:
            logger.debug(f"trafilatura failed for {url}: {e}")

        # Fallback: try plain httpx fetch
        try:
            resp = httpx.get(url, timeout=self.FETCH_TIMEOUT, follow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (compatible; MiroFish/1.0)"})
            if resp.status_code == 200:
                # Very basic text extraction: strip tags
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) >= 100:
                    return text[:5000]
        except Exception:
            pass

        return None

    def _fetch_all_texts(
        self,
        articles: List[Dict],
        progress_callback=None,
    ) -> List[Dict]:
        """Fetch article texts in parallel."""
        total = len(articles)
        completed = 0

        def fetch_one(art):
            text = self._fetch_article_text(art["url"])
            return art["id"], text

        updated = {art["id"]: art.copy() for art in articles}

        with ThreadPoolExecutor(max_workers=self.FETCH_WORKERS) as executor:
            futures = {executor.submit(fetch_one, art): art["id"] for art in articles}
            for future in as_completed(futures, timeout=120):
                completed += 1
                try:
                    art_id, text = future.result(timeout=self.FETCH_TIMEOUT + 2)
                    if text:
                        updated[art_id]["text"] = text
                        updated[art_id]["word_count"] = len(text.split())
                    elif updated[art_id]["snippet"]:
                        updated[art_id]["text"] = updated[art_id]["snippet"]
                        updated[art_id]["word_count"] = len(updated[art_id]["snippet"].split())
                except Exception as e:
                    logger.debug(f"Fetch failed: {e}")

                if progress_callback:
                    pct = 30 + int((completed / total) * 50)
                    progress_callback(pct, f"Fetched {completed}/{total} articles...")

        # Only keep articles that have some text content
        result = [a for a in updated.values() if a["text"]]
        logger.info(f"Articles with text: {len(result)}/{total}")
        return result

    # ------------------------------------------------------------------ #
    #  Diversity selection                                                  #
    # ------------------------------------------------------------------ #

    def _select_diverse_sources(self, articles: List[Dict], n: int = 20) -> List[Dict]:
        """Use Claude to select the best N diverse articles."""
        if len(articles) <= n:
            return articles

        # Build compact metadata list for Claude (no full text to save tokens)
        meta_list = []
        for i, art in enumerate(articles):
            snippet_preview = (art["snippet"] or art["text"])[:200]
            meta_list.append(
                f'{i}. [{art["source"]}] {art["title"]}\n   {snippet_preview}'
            )

        articles_text = "\n\n".join(meta_list)

        prompt = f"""You are a research curator selecting the most valuable and diverse news sources for an AI simulation.

From the articles below, select exactly {n} articles that together provide the best coverage. Prioritize:
1. Diversity of perspectives (pro/con/neutral/analysis)
2. Diversity of source types (major outlets, regional, opinion, official)
3. Diversity of geography/angle (US, international, local)
4. Relevance and informativeness
5. Avoid near-duplicates

Articles:
{articles_text}

Respond with a JSON object:
{{
  "selected": [
    {{"index": 0, "category": "mainstream/analysis/opinion/regional/official", "reason": "brief reason"}},
    ...
  ]
}}

Select exactly {n} articles (or fewer if fewer are available)."""

        selected_articles = []
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            for item in result.get("selected", []):
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(articles):
                    art = articles[idx].copy()
                    art["category"] = item.get("category", "")
                    art["selection_reason"] = item.get("reason", "")
                    selected_articles.append(art)
        except Exception as e:
            logger.warning(f"Diversity selection failed: {e}, falling back to first {n}")
            selected_articles = articles[:n]

        return selected_articles[:n]

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _domain_from_url(url: str) -> str:
        """Extract domain name from URL as fallback source name."""
        try:
            match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if match:
                return match.group(1)
        except Exception:
            pass
        return "Unknown"

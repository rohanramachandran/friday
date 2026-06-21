"""Web search via DuckDuckGo HTML scrape (no API key)."""
import asyncio, urllib.parse, urllib.request, re, html

async def search_tool(query: str, n: int = 5) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search, query, n)

def _search(query, n):
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            html_text = r.read().decode("utf-8", errors="ignore")
        # crude extraction of result snippets
        results = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>', html_text, re.DOTALL)
        if not results:
            return "No results."
        out = []
        for title, snippet in results[:n]:
            t = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            s = html.unescape(re.sub(r"<[^>]+>", "", snippet)).strip()
            out.append(f"- {t}: {s}")
        return "\n".join(out)
    except Exception as e:
        return f"Search error: {e}"

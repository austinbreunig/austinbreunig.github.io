"""Check site pages against docs/contracts/site-contract.md. Plain stdlib, no deps."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

CONTRACTS = Path(__file__).resolve().parent
ROOT = CONTRACTS.parent.parent
SKIP_DIRS = {"docs", "visuals", ".git", ".lifecycle", "node_modules"}
WIDGET_PAGES = {"projects/plumber.html", "projects/skills.html"}
EMBED_SRC = "../embed.js"
VOID = {"meta", "link", "img", "br", "hr", "input"}
FORBIDDEN_JS = re.compile(r"\b(fetch|XMLHttpRequest|WebSocket|import|eval)\b|document\.write")
CLASSES = {
    line.strip()
    for line in (CONTRACTS / "classes.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.classes, self.links, self.bar = [], set(), [], []
        self.lang = self.desc = self.viewport = self.title = False
        self.inline_style = self.in_bar = 0
        self.imgs_no_alt = 0
        self.h1 = 0
        self.icon = self.fonts = False
        self.todo_no_ph = 0
        self.stack = []  # open (tag, classes) pairs, to see where an iframe sits
        self.scripts = []  # (src or None, type)
        self.script_text = ""
        self.in_script = False
        self.iframe_problems = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang") == "en"
        if tag == "meta" and a.get("name") == "description":
            self.desc = True
        if tag == "meta" and a.get("name") == "viewport":
            self.viewport = True
        if tag == "title":
            self.title = True
        if tag == "h1":
            self.h1 += 1
        if tag == "style" or "style" in a:
            self.inline_style += 1
        if tag == "script":
            self.scripts.append((a.get("src"), a.get("type")))
            self.in_script = True
        if tag == "iframe":
            self.check_iframe(a)
        if tag == "img" and "alt" not in a:
            self.imgs_no_alt += 1
        if tag == "link" and "icon" in (a.get("rel") or ""):
            self.icon = True
        if tag == "link" and "fonts.googleapis.com" in a.get("href", ""):
            self.fonts = True
        if a.get("href") == "#todo" and "data-placeholder" not in a:
            self.todo_no_ph += 1
        self.classes.update((a.get("class") or "").split())
        if tag not in VOID:
            self.stack.append((tag, (a.get("class") or "").split()))
        if tag == "link" and a.get("rel") == "stylesheet" and a.get("href", "").startswith("http"):
            return
        for key in ("href", "src"):
            if a.get(key):
                self.links.append((tag, a[key]))
        if tag == "header" and "topbar" in (a.get("class") or ""):
            self.in_bar = 1
        if self.in_bar and tag == "a":
            self.bar.append(a.get("href", ""))

    def check_iframe(self, a):
        parents = self.stack[-2:]
        in_frame = (
            len(parents) == 2
            and parents[0][0] == "figure" and "visual" in parents[0][1]
            and parents[1][0] == "div" and "visual-frame" in parents[1][1]
        )
        if not in_frame:
            self.iframe_problems.append("iframe must sit in figure.visual > div.visual-frame")
        if a.get("sandbox") != "allow-scripts":
            self.iframe_problems.append('iframe needs sandbox="allow-scripts"')
        if not a.get("title"):
            self.iframe_problems.append("iframe needs a title")
        if not (a.get("src") or "").startswith("../visuals/"):
            self.iframe_problems.append("iframe src must be relative, starting ../visuals/")

    def handle_data(self, data):
        if self.in_script:
            self.script_text += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag == "header":
            self.in_bar = 0


def script_problems(p, virtual):
    """Rule 7: see docs/contracts/site-contract.md."""
    bad = []
    embeds = [src for src, _ in p.scripts if src == EMBED_SRC]
    inline = [src for src, _ in p.scripts if src is None]
    other = [src for src, _ in p.scripts if src is not None and src != EMBED_SRC]
    if other:
        bad.append(f"script src must be {EMBED_SRC}: {other}")
    if any(kind == "module" for _, kind in p.scripts):
        bad.append('script type="module" not allowed')
    if len(embeds) > 1:
        bad.append("only one embed script allowed")
    if embeds and "visual" not in p.classes:
        bad.append("embed script on a page with no .visual")
    if "visual" in p.classes and not embeds:
        bad.append(f"page with .visual needs <script src=\"{EMBED_SRC}\">")
    if inline and virtual not in WIDGET_PAGES:
        bad.append("inline <script> only on plumber.html and skills.html")
    if len(inline) > 1:
        bad.append("only one inline script allowed")
    if FORBIDDEN_JS.search(p.script_text):
        bad.append("script uses a forbidden call (fetch, XHR, WebSocket, import, eval, document.write)")
    return bad


def check(path, virtual, files):
    """Return a list of problems. `virtual` is the page's site path, e.g. projects/bap.html."""
    p = Page()
    p.feed(Path(path).read_text())
    prefix = "../" * virtual.count("/")
    bad = []
    if not p.lang: bad.append('missing <html lang="en">')
    if not p.title: bad.append("missing <title>")
    if not p.desc: bad.append("missing meta description")
    if not p.viewport: bad.append("missing viewport meta")
    if p.h1 != 1: bad.append(f"needs exactly one h1, found {p.h1}")
    if p.inline_style: bad.append("inline <style> or style= attribute")
    bad += script_problems(p, virtual)
    bad += p.iframe_problems
    if p.imgs_no_alt: bad.append("img without alt")
    if not p.icon: bad.append('missing <link rel="icon">')
    if not p.fonts: bad.append("missing Google Fonts link")
    if p.todo_no_ph: bad.append("#todo link without data-placeholder")
    for c in sorted(p.classes - CLASSES):
        bad.append(f"class not in classes.txt: {c}")
    if (prefix + "style.css") not in [h for _, h in p.links]:
        bad.append(f"must link {prefix}style.css")
    for tag, ref in p.links:
        if ref.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            continue
        if ref.startswith("/"):
            bad.append(f"root-absolute link: {ref}")
            continue
        if not ref.startswith(prefix):
            bad.append(f"link needs prefix '{prefix}': {ref}")
            continue
        if files and not (ROOT / ref[len(prefix):].split("#")[0]).exists():
            bad.append(f"broken link: {ref}")
    if p.bar:
        # the middle three may be real URLs later
        got = p.bar
        if len(got) != 5 or got[0] != prefix + "index.html" or got[4] != prefix + "resume.html":
            bad.append(f"top bar links must be 5 in order (name, Email, LinkedIn, GitHub, Resume): {got}")
    else:
        bad.append("missing header.topbar")
    return bad


def site_pages():
    for f in sorted(ROOT.rglob("*.html")):
        rel = f.relative_to(ROOT)
        if rel.parts[0] not in SKIP_DIRS:
            yield f, rel.as_posix()


def main():
    if "--selftest" in sys.argv:
        ok = True
        for f in sorted((CONTRACTS / "fixtures").glob("*.html")):
            text = f.read_text()
            expect = re.search(r"expect: (pass|fail)", text)
            if not expect:
                continue  # not a page fixture (e.g. visual-stub)
            virtual = re.search(r"check-as: (\S+)", text).group(1)
            bad = check(f, virtual, files=False)
            if bool(bad) == (expect.group(1) == "pass"):
                ok = False
                print(f"SELFTEST FAIL {f.stem}: {bad}")
        print("selftest ok" if ok else "selftest FAILED")
        return 0 if ok else 1
    bad_total = 0
    if not (ROOT / "style.css").exists():
        print("style.css: missing")
        bad_total += 1
    for f, rel in site_pages():
        for msg in check(f, rel, files=True):
            print(f"{rel}: {msg}")
            bad_total += 1
    print("OK" if not bad_total else f"{bad_total} problem(s)")
    return 1 if bad_total else 0


if __name__ == "__main__":
    sys.exit(main())

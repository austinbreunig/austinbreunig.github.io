"""Check site pages against docs/contracts/site-contract.md. Plain stdlib, no deps."""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

CONTRACTS = Path(__file__).resolve().parent
ROOT = CONTRACTS.parent.parent
SKIP_DIRS = {"docs", "visuals", ".git", ".lifecycle", "node_modules"}
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
        self.inline_style = self.script = self.in_bar = 0
        self.imgs_no_alt = 0
        self.h1 = 0
        self.icon = self.fonts = False
        self.todo_no_ph = 0

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
            self.script += 1
        if tag == "img" and "alt" not in a:
            self.imgs_no_alt += 1
        if tag == "link" and "icon" in (a.get("rel") or ""):
            self.icon = True
        if tag == "link" and "fonts.googleapis.com" in a.get("href", ""):
            self.fonts = True
        if a.get("href") == "#todo" and "data-placeholder" not in a:
            self.todo_no_ph += 1
        self.classes.update((a.get("class") or "").split())
        if tag == "link" and a.get("rel") == "stylesheet" and a.get("href", "").startswith("http"):
            return
        for key in ("href", "src"):
            if a.get(key):
                self.links.append((tag, a[key]))
        if tag == "header" and "topbar" in (a.get("class") or ""):
            self.in_bar = 1
        if self.in_bar and tag == "a":
            self.bar.append(a.get("href", ""))

    def handle_endtag(self, tag):
        if tag == "header":
            self.in_bar = 0


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
    if p.script: bad.append("<script> not allowed at launch")
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
        want = [prefix + "index.html", "#todo", "#todo", "#todo", prefix + "resume.html"]
        got = [h for h in p.bar]
        # real URLs may replace the three placeholders
        if len(got) != 5 or got[0] != want[0] or got[4] != want[4]:
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
        for name, should_pass in (("good-index", True), ("good-project", True), ("bad-page", False)):
            f = CONTRACTS / "fixtures" / f"{name}.html"
            virtual = re.search(r"check-as: (\S+)", f.read_text()).group(1)
            bad = check(f, virtual, files=False)
            if bool(bad) == should_pass:
                ok = False
                print(f"SELFTEST FAIL {name}: {bad}")
        print("selftest ok" if ok else "selftest FAILED")
        return 0 if ok else 1
    # style.css must define every contract class
    bad_total = 0
    css = ROOT / "style.css"
    if css.exists():
        text = css.read_text()
        for c in sorted(CLASSES):
            if not re.search(r"\." + re.escape(c) + r"(?![\w-])", text):
                print(f"style.css: no rule for .{c}")
                bad_total += 1
        for tok in ("bg", "surface", "ink", "muted", "accent", "accent2", "line"):
            if f"--{tok}:" not in text:
                print(f"style.css: token --{tok} not declared")
                bad_total += 1
    else:
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

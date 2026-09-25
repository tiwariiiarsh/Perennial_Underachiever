"""Record normalisation for names and addresses.

Everything is language-agnostic: non-Latin scripts are transliterated with
unidecode and tokens are additionally reduced to a *consonant skeleton*
(vowels dropped, aspirates/digraphs folded, repeats collapsed) so that
"maarkettiNg" (Devanagari transliteration), "Makreitrdng" (typo) and
"marketing" land on similar keys.  No country-specific branching is done, so
unseen countries (France in test) go through the same code path.
"""
import re
from multiprocessing import Pool

import pandas as pd
from unidecode import unidecode

# ----------------------------------------------------------------- vocabularies
LEGAL_SUFFIX = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "llc": "llc", "l l c": "llc", "ltd": "ltd",
    "limited": "ltd", "pvt": "pvt", "private": "pvt", "pc": "pc", "p c": "pc",
    "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "lllp": "llp",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl", "gmbh": "gmbh",
    "opc": "opc", "pa": "pa",
}
HONORIFIC = {"mr", "mrs", "ms", "m s", "messrs", "shri", "sri", "smt", "mme", "mlle", "m"}
ALIAS_MARKERS = re.compile(
    r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|dba|formerly known as|f\s*/\s*k\s*/\s*a|fka|"
    r"also known as|a\s*/\s*k\s*/\s*a|aka|trading as|t\s*/\s*a|anciennement|dite?)\b")

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard", "bd": "boulevard",
    "bld": "boulevard", "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "ter": "terrace", "trl": "trail", "sq": "square", "mkt": "market", "ngr": "nagar", "r": "rue",
    "ch": "chemin", "rte": "route", "imp": "impasse", "all": "allee", "apt": "apartment",
    "ste": "suite", "fl": "floor", "flr": "floor", "no": "number", "nr": "near", "opp": "opposite",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "mt": "mount", "ft": "fort", "pt": "point",
    "fbg": "faubourg", "qt": "quartier", "pvt": "private", "ltd": "limited",
}
ADDR_GENERIC = {
    "street", "road", "avenue", "drive", "lane", "court", "boulevard", "place", "highway",
    "parkway", "circle", "terrace", "trail", "square", "way", "rue", "chemin", "route",
    "impasse", "allee", "apartment", "suite", "unit", "floor", "ground", "first", "second",
    "number", "box", "po", "p", "o", "near", "opposite", "behind", "the", "de", "du", "des",
    "la", "le", "les", "d", "l", "et", "and", "of", "plot", "flat", "shop", "office", "building",
    "bldg", "house", "h", "kh", "sector", "block", "bis", "ter", "b", "a", "c",
    "cedex", "bp", "cs",
}


LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}


def skel(tok):
    """Consonant skeleton of an ASCII token."""
    t = tok.replace("ph", "f").replace("ck", "k").replace("x", "ks").replace("q", "k")
    t = t.replace("c", "k").replace("v", "b").replace("w", "b").replace("z", "s").replace("j", "g")
    t = re.sub(r"(?<=[bcdfgjklmnpqrstvxz])h", "", t)          # aspirates: bh->b, th->t, ...
    t = re.sub(r"[aeiouy]", "", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


SUFFIX_SKELS = {skel(w) for w in ("private", "limited", "incorporated", "corporation", "company",
                                  "privet", "limitad")} | {"lp", "lk"}


def _ascii(s):
    return unidecode(s).lower() if s else ""


def _name_tokens(text):
    text = text.replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"\b(www\.)?([a-z0-9-]+)\.(com|net|org|in|co\.in|fr|biz|info|us|io)\b", r" \2 ", text)
    text = re.sub(r"\b([a-z])\.(?=[a-z]\b)", r"\1", text)       # p.c. / s.a.s -> pc / sas
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    # leetspeak / OCR digits inside words: tayl0r -> taylor, c1inic -> clinic
    text = re.sub(r"(?<=[a-z])[0-9]|[0-9](?=[a-z])", lambda m: LEET.get(m.group(), m.group()), text)
    return text.split()


def _core(tokens):
    core, suffixes = [], []
    for i, t in enumerate(tokens):
        if t in LEGAL_SUFFIX:
            suffixes.append(LEGAL_SUFFIX[t])
        elif len(t) >= 4 and skel(t) in SUFFIX_SKELS:
            suffixes.append("pvt" if skel(t) == "prbt" else "ltd" if skel(t) == "lmtd" else skel(t))
        elif i == 0 and t in HONORIFIC and len(tokens) > 1:
            continue
        else:
            core.append(t)
    # de-duplicate stutters ("Renaissancere Renaissancere", "Bailey, Bailey")
    out = [t for i, t in enumerate(core) if i == 0 or t != core[i - 1]]
    return out, suffixes


def normalize_name(raw):
    s = _ascii(raw)
    parts = [p for p in ALIAS_MARKERS.split(s) if p.strip()]
    main, suffixes = _core(_name_tokens(ALIAS_MARKERS.sub(" ", s)))
    # alias: text after the last marker (DBA / FKA); main tokens then already contain it
    alt = _core(_name_tokens(parts[-1]))[0] if len(parts) > 1 else []
    return main, alt, suffixes


def _addr_parse(raw):
    s = _ascii(raw)
    segs = [seg for seg in s.split(",")]
    nums, words, seg_words = [], [], []
    for seg in segs:
        for m in re.finditer(r"\d+(?:\s*[-/]\s*\d+)*", seg):
            nums.append(re.sub(r"\D", "", m.group()).lstrip("0") or "0")
        toks = re.sub(r"[^a-z ]+", " ", seg).split()
        toks = [ADDR_ABBR.get(t, t) for t in toks]
        words.extend(toks)
        seg_words.append([t for t in toks if t not in ADDR_GENERIC])
    return nums, words, seg_words


def normalize_record(name, addr):
    main, alt, suffixes = normalize_name(name)
    nums, words, seg_words = _addr_parse(addr)
    return (
        " ".join(main), " ".join(alt), " ".join(sorted(set(suffixes))),
        " ".join(skel(t) for t in main), " ".join(nums), " ".join(words),
        "|".join(" ".join(skel(t) for t in sw) for sw in seg_words if sw),
        int(bool(name) and bool(re.search(r"[^\x00-\x7f]", name)) and not re.search(r"[A-Za-z]", name)),
    )


NORM_COLS = ["name", "alt", "suffix", "name_skel", "nums", "addr", "addr_skel", "nonlatin_name"]


def _norm_chunk(args):
    names, addrs = args
    return [normalize_record(n, a) for n, a in zip(names, addrs)]


def normalize_df(df, n_workers=8, chunk=50_000):
    """Return DataFrame with entity_id, country + NORM_COLS."""
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(n_workers) as p:
        rows = [r for part in p.imap(_norm_chunk, jobs) for r in part]
    out = pd.DataFrame(rows, columns=NORM_COLS)
    out.insert(0, "country", df.country.str.lower().to_numpy())
    out.insert(0, "entity_id", df.entity_id.to_numpy())
    out["nonlatin_name"] = out.nonlatin_name.astype("int8")
    return out

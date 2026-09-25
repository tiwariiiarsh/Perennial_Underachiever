"""Build <team>_submission.zip in the structure required by the challenge.

    python src/make_submission.py --team Perennial_Underachiever
"""
import argparse
import zipfile

from config import OUTPUT_DIR, ROOT

CODE_ITEMS = ["src", "api", "app", "README.md", "requirements.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", default=str(ROOT.parent / "Documentation_template.md"))
    a = ap.parse_args()
    out = ROOT.parent / f"{a.team}_submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(OUTPUT_DIR / f, f"output/{f}")
        for item in CODE_ITEMS:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f"code/business_entity_resolution/{f.relative_to(ROOT)}")
        z.write(a.doc, "Documentation_template.md")
    print("wrote", out)


if __name__ == "__main__":
    main()

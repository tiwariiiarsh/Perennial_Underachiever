"""Streamlit frontend (talks to the FastAPI backend).  Run from business_entity_resolution/:

    streamlit run app/streamlit_app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("ER_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Business Entity Resolution", layout="wide")
st.title("Business Entity Resolution")


def call(method, path, **kw):
    try:
        r = requests.request(method, API + path, timeout=300, **kw)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        st.error(f"API error ({API}{path}): {e}")
        st.stop()


def record_form(key, defaults=("", "", "")):
    name = st.text_input("Business name", defaults[0], key=key + "n")
    addr = st.text_input("Address", defaults[1], key=key + "a")
    country = st.text_input("Country", defaults[2], key=key + "c")
    return {"business_name": name, "business_address": addr, "country": country}


tab1, tab2, tab3, tab4 = st.tabs(["Search / Match", "Compare two records", "Batch upload", "Model dashboard"])

with tab1:
    rec = record_form("m", ("Laxmi Marketing Private Limited", "12-11-1178, Boudha Nagar, Secunderabad, Telangana", "India"))
    k = st.slider("Candidates to show", 5, 40, 10)
    if st.button("Find matches", type="primary"):
        res = call("POST", f"/match?top_k={k}", json=rec).json()
        st.subheader(f"{len(res['matches'])} match(es)")
        for c in res["candidates"]:
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.markdown(f"**{c['name']}**  \n{c['address']}  \n`{c['entity_id']}` · {c['country']}")
                b.metric("Match probability", f"{c['probability']:.1%}", "MATCH" if c["matched"] else "no match",
                         delta_color="normal" if c["matched"] else "off")
                b.progress(c["probability"])
                with st.expander("Evidence"):
                    st.json(c["evidence"])

with tab2:
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Record A**")
        a = record_form("a", ("Blue Software Private Limited", "108, P B Road Karur Industrial Area, Davanagere, Karnataka", "India"))
    with cb:
        st.markdown("**Record B**")
        b = record_form("b", ("Blue Sóftware Pvt Ltd", "Davanagere, 108, KA", "India"))
    if st.button("Compare", type="primary"):
        res = call("POST", "/compare", json={"a": a, "b": b}).json()
        st.metric("Same business?", "YES" if res["match"] else "NO", f"p = {res['probability']:.1%}")
        st.progress(res["probability"])
        f = pd.Series(res["features"], name="value")
        st.bar_chart(f[[c for c in f.index if c.startswith(("n_", "k", "a_", "aw", "num"))]])
        n = pd.DataFrame({"A": res["normalized_a"], "B": res["normalized_b"]})
        st.dataframe(n, use_container_width=True)

with tab3:
    st.write("Upload a TSV with columns `entity_id, business_name, business_address, country` (max 5000 rows).")
    up = st.file_uploader("TSV file", type=["tsv", "txt"])
    if up and st.button("Run batch matching", type="primary"):
        r = call("POST", "/batch", files={"file": (up.name, up.getvalue())})
        st.download_button("Download matching_results.tsv", r.content, "matching_results.tsv")
        st.dataframe(pd.read_csv(pd.io.common.StringIO(r.text), sep="\t", dtype=str, keep_default_na=False).head(200))

with tab4:
    m = call("GET", "/stats").json()
    c = st.columns(4)
    c[0].metric("OOF macro F0.5", m["oof_macro_f05"])
    c[1].metric("Blocking pair recall", f"{m['blocking_pair_recall']:.1%}")
    c[2].metric("Candidates / S1", m["candidates_per_s1"])
    c[3].metric("Pair PR-AUC", m["pair_ap"])
    st.subheader("F0.5 by country")
    st.bar_chart(pd.Series(m["per_country_f05"]))
    st.subheader("Decision-logic ablation (macro F0.5)")
    st.dataframe(pd.Series(m["ablation"], name="macro F0.5"))
    st.subheader("Feature importance (gain)")
    st.bar_chart(pd.Series(m["feature_importance"]).head(20))
    st.caption(f"Decision parameters: {m['decision_params']}")

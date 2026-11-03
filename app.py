"""
The whole website (backend + frontend) in one file.
Run with: python app.py
Needs model/model.joblib, created by running train.py first.
"""
import re

import joblib
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("model/model.joblib")


def clean(text: str) -> str:
    """Must match the version in train.py."""
    text = str(text)
    text = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fake News Checker</title>
<style>
  :root { --bg:#eef1f4; --panel:#fff; --ink:#1b2430; --muted:#5b6776; --line:#d3d9e0; --real:#0f7b6c; --fake:#c2334d; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141a22; --panel:#1c2430; --ink:#e8edf3; --muted:#9aa6b5; --line:#2e3947; --real:#3fbfa8; --fake:#f0708a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,sans-serif; }
  main { max-width:640px; margin:0 auto; padding:40px 20px 64px; }
  h1 { font-size:2rem; margin:0 0 10px; }
  p.lede { color:var(--muted); margin:0 0 24px; }
  textarea { width:100%; min-height:200px; padding:12px; background:var(--panel); color:var(--ink);
    border:1px solid var(--line); border-radius:6px; font:inherit; resize:vertical; }
  .row { display:flex; gap:12px; align-items:center; margin-top:12px; }
  button { background:var(--ink); color:var(--bg); border:0; border-radius:6px; padding:12px 20px;
    font:600 1rem inherit; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  #count { margin-left:auto; color:var(--muted); font-size:.875rem; }
  #error { color:var(--fake); margin-top:10px; min-height:1.4em; }
  #result { margin-top:28px; display:none; }
  #verdict { font-size:1.6rem; font-weight:700; margin:0 0 14px; }
  .meter { height:12px; border-radius:6px; background:linear-gradient(90deg,var(--fake),var(--line) 50%,var(--real)); position:relative; }
  .marker { position:absolute; top:-5px; width:4px; height:22px; background:var(--ink); left:50%; transition:left .4s; }
  .ends { display:flex; justify-content:space-between; color:var(--muted); font-size:.85rem; margin-top:6px; }
  .note { color:var(--muted); font-size:.85rem; margin-top:28px; }
</style>
</head>
<body>
<main>
  <h1>Does this article read like real news?</h1>
  <p class="lede">Paste the headline and text of a news article and check it.</p>

  <textarea id="text" placeholder="Paste the headline and article here"></textarea>
  <div class="row">
    <button id="check">Check article</button>
    <span id="count">0 words</span>
  </div>
  <div id="error"></div>

  <section id="result">
    <p id="verdict"></p>
    <div class="meter"><div class="marker" id="marker"></div></div>
    <div class="ends"><span>Fake</span><span>Real</span></div>
    <p id="detail"></p>
  </section>

  <p class="note">This tool judges writing style, not facts. A high score does not prove a story is true.</p>
</main>
<script>
  const $ = (id) => document.getElementById(id);
  const words = (s) => s.trim().split(/\\s+/).filter(Boolean).length;

  $("text").addEventListener("input", () => { $("count").textContent = words($("text").value) + " words"; });

  $("check").addEventListener("click", async () => {
    $("error").textContent = "";
    $("check").disabled = true; $("check").textContent = "Checking…";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: $("text").value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");

      const real = data.real_percent;
      $("result").style.display = "block";
      $("verdict").textContent = data.label;
      $("verdict").style.color = real >= 50 ? "var(--real)" : "var(--fake)";
      $("marker").style.left = `calc(${real}% - 2px)`;
      $("detail").textContent = `${real}% real, ${(100 - real).toFixed(1)}% fake.`;
    } catch (e) {
      $("error").textContent = e.message;
    } finally {
      $("check").disabled = false; $("check").textContent = "Check article";
    }
  });
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if len(text.split()) < 20:
        return jsonify(error="Paste at least 20 words of the article."), 400

    real_prob = float(model.predict_proba([clean(text)])[0][1])
    return jsonify(
        label="Likely real" if real_prob >= 0.5 else "Likely fake",
        real_percent=round(real_prob * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)

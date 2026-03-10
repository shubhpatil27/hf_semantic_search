const API = "http://127.0.0.1:8000";

const noteText = document.getElementById("noteText");
const addBtn = document.getElementById("addBtn");
const addMsg = document.getElementById("addMsg");

const query = document.getElementById("query");
const searchBtn = document.getElementById("searchBtn");
const results = document.getElementById("results");

addBtn.onclick = async () => {
  const text = noteText.value.trim();
  if (!text) { addMsg.textContent = "Please write a note first."; return; }

  addMsg.textContent = "Adding...";
  const res = await fetch(`${API}/notes`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });
  const data = await res.json();
  if (data.ok) {
    addMsg.textContent = `✅ Added note #${data.note.id} (total: ${data.count})`;
    noteText.value = "";
  } else {
    addMsg.textContent = `❌ ${data.error || "Error"}`;
  }
};

searchBtn.onclick = async () => {
  const q = query.value.trim();
  results.innerHTML = "";
  if (!q) return;

  results.innerHTML = "<div class='msg'>Searching...</div>";

  const res = await fetch(`${API}/search`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({query: q, top_k: 5})
  });
  const data = await res.json();

  if (!data.results.length) {
    results.innerHTML = "<div class='msg'>No results.</div>";
    return;
  }

  results.innerHTML = "";
  data.results.forEach(r => {
    const div = document.createElement("div");
    div.className = "result";
    div.innerHTML = `
      <div>${r.text}</div>
      <div class="score">score: ${r.score.toFixed(4)}</div>
    `;
    results.appendChild(div);
  });
};
const $ = (sel) => document.querySelector(sel);
const docList = $("#doc-list");
const answers = $("#answers");
const editor = $("#editor");

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  return body;
}

let toastTimer;
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("err", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), isError ? 6000 : 3000);
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function describeSync(report) {
  const parts = [];
  if (report.added.length) parts.push(`added ${report.added.join(", ")}`);
  if (report.updated.length) parts.push(`re-indexed ${report.updated.join(", ")}`);
  if (report.removed.length) parts.push(`removed ${report.removed.join(", ")}`);
  return parts.length ? `Synced: ${parts.join("; ")}` : "No changes — nothing to re-index";
}

function formatSize(bytes) {
  return bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`;
}

async function loadDocs(flash = []) {
  try {
    const docs = await api("/documents");
    if (!docs.length) {
      docList.innerHTML = `<li class="muted small">No documents yet. Upload one or write a new one.</li>`;
      return;
    }
    docList.innerHTML = docs.map((d) => `
      <li class="doc${flash.includes(d.name) ? " flash" : ""}" data-name="${escapeHtml(d.name)}">
        <div class="info">
          <span class="name" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
          <span class="muted small">${formatSize(d.size)} · ${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span>
        </div>
        ${d.editable ? `<button class="icon-btn edit" title="Edit">✎</button>` : ""}
        <button class="icon-btn del" title="Delete">✕</button>
      </li>`).join("");
  } catch (err) {
    docList.innerHTML = `<li class="small" style="color:var(--danger)">${escapeHtml(err.message)}</li>`;
  }
}

docList.addEventListener("click", async (e) => {
  const item = e.target.closest(".doc");
  if (!item) return;
  const name = item.dataset.name;
  if (e.target.closest(".edit")) {
    try {
      const doc = await api(`/documents/${encodeURIComponent(name)}`);
      openEditor(doc.name, doc.content);
    } catch (err) { toast(err.message, true); }
  } else if (e.target.closest(".del")) {
    if (!confirm(`Delete ${name}? It will be removed from the index too.`)) return;
    try {
      toast(describeSync(await api(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" })));
      loadDocs();
    } catch (err) { toast(err.message, true); }
  }
});

// Uploads
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
async function upload(files) {
  if (!files.length) return;
  const form = new FormData();
  [...files].forEach((f) => form.append("files", f));
  toast(`Uploading and indexing ${files.length} file${files.length > 1 ? "s" : ""}…`);
  try {
    const report = await api("/documents", { method: "POST", body: form });
    toast(describeSync(report));
    loadDocs([...report.added, ...report.updated]);
  } catch (err) { toast(err.message, true); }
  fileInput.value = "";
}
fileInput.addEventListener("change", () => upload(fileInput.files));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => upload(e.dataTransfer.files));

// Editor
let editingOriginal = null;
function openEditor(name = "", content = "") {
  editingOriginal = name || null;
  $("#editor-name").value = name;
  $("#editor-name").readOnly = Boolean(name);
  $("#editor-content").value = content;
  editor.showModal();
  (name ? $("#editor-content") : $("#editor-name")).focus();
}
$("#new-doc").addEventListener("click", () => openEditor());
$("#editor-form").addEventListener("submit", async (e) => {
  if (e.submitter?.value !== "save") return;
  e.preventDefault();
  let name = $("#editor-name").value.trim();
  if (!/\.(txt|md|markdown|rst)$/i.test(name)) name += ".md";
  const btn = $("#save-btn");
  btn.disabled = true;
  btn.textContent = "Syncing…";
  try {
    const report = await api(`/documents/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: $("#editor-content").value }),
    });
    editor.close();
    toast(describeSync(report));
    loadDocs([...report.added, ...report.updated]);
  } catch (err) {
    toast(err.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Save & sync";
  }
});

// Asking
function renderAnswer(text, id) {
  const html = escapeHtml(text.trim())
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) =>
      nums.split(/\s*,\s*/).map((n) => `<button class="cite" data-target="${id}-src-${n}">${n}</button>`).join(""));
  return html.split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function renderTrace(trace) {
  if (!trace.length) return "";
  return `<details class="trace-wrap"><summary class="muted small">How the agents answered (${trace.length} steps)</summary><ul class="trace">${trace.map((s) => {
    const items = [...(s.detail.queries || []), ...(s.detail.issues || [])];
    const warn = s.agent === "Verifier" && s.summary !== "approved";
    return `<li class="${warn ? "warn" : ""}"><span class="agent">${escapeHtml(s.agent)}</span> ${escapeHtml(s.summary)} <span class="muted small">· ${(s.ms / 1000).toFixed(1)}s</span>
      ${items.length ? `<ul>${items.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>` : ""}</li>`;
  }).join("")}</ul></details>`;
}

async function ask(question) {
  const id = `qa${Date.now()}`;
  const block = document.createElement("div");
  block.className = "qa";
  const agents = document.querySelector('input[name="mode"]:checked').value === "agents";
  block.innerHTML = `<div class="q">${escapeHtml(question)}</div><div class="a"><span class="thinking"><i></i><i></i><i></i></span>
    ${agents ? `<span class="muted small"> the agents are working — planning, searching, writing, then fact-checking…</span>` : ""}</div>`;
  answers.prepend(block);
  const btn = $("#ask-btn");
  btn.disabled = true;
  try {
    const res = await api("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: document.querySelector('input[name="mode"]:checked').value }),
    });
    if (res.verified !== null) {
      block.querySelector(".q").insertAdjacentHTML("beforeend",
        res.verified ? `<span class="badge ok">✓ verified</span>` : `<span class="badge warn">unverified</span>`);
    }
    block.querySelector(".a").innerHTML = renderAnswer(res.answer, id) + res.notes.map((n) => `<p class="note">${escapeHtml(n)}</p>`).join("");
    block.insertAdjacentHTML("beforeend", renderTrace(res.trace));
    if (res.sources.length) {
      block.insertAdjacentHTML("beforeend", `<div class="sources">${res.sources.map((s, i) => `
        <details id="${id}-src-${i + 1}">
          <summary><strong>[${i + 1}]</strong> ${escapeHtml(s.doc)} <span class="muted small">· chunk ${s.chunk} · score ${s.score}</span></summary>
          <pre>${escapeHtml(s.text)}</pre>
        </details>`).join("")}</div>`);
    }
  } catch (err) {
    block.querySelector(".a").innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

answers.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite) return;
  const src = document.getElementById(cite.dataset.target);
  if (!src) return;
  src.open = true;
  src.classList.add("hl");
  src.scrollIntoView({ behavior: "smooth", block: "nearest" });
  setTimeout(() => src.classList.remove("hl"), 1500);
});

$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q) ask(q);
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});
$("#suggestions").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  $("#question").value = e.target.textContent;
  $("#ask-form").requestSubmit();
});

document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
  $("#mode-hint").textContent = r.value === "agents" && r.checked
    ? "Planner → Researcher → Writer → Verifier" : "One search, one Gemini call — faster, uses less quota";
}));

api("/health").then((h) => { $("#models").textContent = `${h.chat_model} · ${h.embed_model}`; }).catch(() => {});
loadDocs();

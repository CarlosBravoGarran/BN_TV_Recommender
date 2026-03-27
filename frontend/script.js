/* ================================================================
   🔌  INTEGRACIÓN CON EL BACKEND
   ================================================================

   Tu endpoint debe aceptar:
     POST /api/chat
     Body: { "message": "texto del usuario" }

   Y responder con este JSON:
     {
       "message":  "Texto conversacional del asistente",
       "action":   "RECOMMEND" | "ALTERNATIVE" | "FEEDBACK" | "SMALLTALK" | "ASK",
       "item":     "Título recomendado o null",
       "content":  {                         ← objeto TMDB, puede ser null
         "title":          "...",
         "overview":       "...",
         "vote_average":   7.5,
         "release_date":   "2023-05-12",     ← o "first_air_date" para series
         "media_type":     "movie" | "tv",
         "poster_path":    "/abc123.jpg"     ← ruta relativa TMDB, puede ser null
       }
     }

   Ejemplo mínimo con Flask (api.py junto a tu main.py):

     from flask import Flask, request, jsonify
     from flask_cors import CORS
     from main import run_pipeline   # o importa tu lógica directamente

     app = Flask(__name__)
     CORS(app)   # imprescindible para que el navegador pueda hacer fetch

     @app.route("/api/chat", methods=["POST"])
     def chat():
         mensaje = request.json.get("message", "")
         respuesta = run_pipeline(mensaje)   # devuelve el dict de arriba
         return jsonify(respuesta)

     if __name__ == "__main__":
         app.run(port=5000)

   ================================================================ */

const API_URL     = "http://localhost:5000/api/chat";
const PROFILE_URL = "http://localhost:5000/api/profile";

// ── Perfil de usuario ───────────────────────────────────────────
let currentUserId  = null;
let currentProfile = null;

async function initProfile() {
  currentUserId = localStorage.getItem("tv_user_id");

  if (!currentUserId) {
    showProfileModal();
    return;
  }

  try {
    const res = await fetch(`${PROFILE_URL}/${currentUserId}`);
    if (res.ok) {
      currentProfile = await res.json();
      _applyProfileToUI();
    } else {
      showProfileModal();
    }
  } catch {
    // Backend aún no disponible — continuar sin perfil
  }
}

function showProfileModal() {
  const modal    = document.getElementById("profileModal");
  const closeBtn = document.getElementById("profileClose");

  // Mostrar X solo si ya hay perfil (edición)
  closeBtn.style.display = currentProfile ? "flex" : "none";

  // Pre-rellenar si estamos editando
  if (currentProfile) {
    document.getElementById("profileName").value = currentProfile.name || "";
    _setActive("ageGroup",       currentProfile.age);
    _setActive("genderGroup",    currentProfile.gender);
    _setActive("householdGroup", currentProfile.household);
  }

  modal.classList.add("show");
}

function hideProfileModal() {
  document.getElementById("profileModal").classList.remove("show");
}

function skipProfile() {
  // Generar un user_id anónimo para que no vuelva a salir el modal
  if (!currentUserId) {
    currentUserId = crypto.randomUUID();
    localStorage.setItem("tv_user_id", currentUserId);
  }
  hideProfileModal();
}

function onModalOverlayClick(e) {
  if (e.target === document.getElementById("profileModal") && currentProfile) {
    hideProfileModal();
  }
}

async function saveProfile() {
  const name      = document.getElementById("profileName").value.trim();
  const age       = document.querySelector("#ageGroup .sel-btn.active")?.dataset.val || null;
  const gender    = document.querySelector("#genderGroup .sel-btn.active")?.dataset.val || null;
  const household = document.querySelector("#householdGroup .sel-btn.active")?.dataset.val || null;

  if (!name) {
    const inp = document.getElementById("profileName");
    inp.classList.add("shake");
    setTimeout(() => inp.classList.remove("shake"), 500);
    inp.focus();
    return;
  }

  if (!currentUserId) {
    currentUserId = crypto.randomUUID();
    localStorage.setItem("tv_user_id", currentUserId);
  }

  const profile = { user_id: currentUserId, name, age, gender, household };

  try {
    await fetch(PROFILE_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(profile),
    });
  } catch (e) {
    console.warn("No se pudo guardar el perfil en el servidor:", e);
  }

  currentProfile = profile;
  _applyProfileToUI();
  hideProfileModal();
}

function _applyProfileToUI() {
  const badge = document.getElementById("profileBadge");
  const name  = document.getElementById("profileBadgeName");

  if (currentProfile?.name) {
    name.textContent  = currentProfile.name;
    badge.style.display = "flex";

    const h2 = document.querySelector("#welcomeScreen h2");
    if (h2) h2.textContent = `Hola ${currentProfile.name}, ¿qué quieres ver hoy?`;
  }
}

function _setActive(groupId, value) {
  document.querySelectorAll(`#${groupId} .sel-btn`).forEach(b => {
    b.classList.toggle("active", b.dataset.val === value);
  });
}

// Delegación de eventos para los grupos de selección
document.querySelectorAll(".select-group").forEach(group => {
  group.addEventListener("click", e => {
    const btn = e.target.closest(".sel-btn");
    if (!btn) return;
    group.querySelectorAll(".sel-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  });
});

// ── Fin gestión de perfil ───────────────────────────────────────

async function callBackend(userMessage) {
  const body = { message: userMessage };
  if (currentUserId) body.user_id = currentUserId;

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json(); 
}

/* ================================================================
   FIN DE LA ZONA DE INTEGRACIÓN — no necesitas tocar nada más
   ================================================================ */


// ── Estado global ──────────────────────────────────────────────
let isLoading    = false;
let welcomeShown = true;

// URL base para imágenes de TMDB
const TMDB_IMG = "https://image.tmdb.org/t/p/w200";


// ── Enviar mensaje ──────────────────────────────────────────────
async function sendMessage() {
  try {
  const input = document.getElementById("userInput");
  const text  = input.value.trim();
  if (!text || isLoading) return;

  hideWelcome();
  input.value = "";
  input.style.height = "auto";

  appendUserMsg(text);
  setLoading(true);

  try {
    const data = await callBackend(text);
    setLoading(false);
    renderBotMsg(data);
  } catch (err) {
    setLoading(false);
    renderError("No se pudo conectar con el servidor. ¿Está corriendo el backend?");
    console.error(err);
  }
  } catch(e) { console.error("ERROR EN sendMessage:", e); }

}


// ── Renderizar respuesta del bot ────────────────────────────────
//
// Campos esperados en `data`:
//   data.message  → texto a mostrar en la burbuja
//   data.action   → RECOMMEND | ALTERNATIVE | FEEDBACK | SMALLTALK | ASK
//   data.item     → título de la recomendación (string o null)
//   data.content  → objeto TMDB completo (o null)
//
function renderBotMsg(data) {
  const wrap  = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "msg bot";

  const message = data.message || "";
  const action  = data.action  || "SMALLTALK";
  const content = data.content || null;

  // ── Tarjeta de película/serie ──
  let cardHTML = "";
  if (content && (action === "RECOMMEND" || action === "ALTERNATIVE")) {
    const title    = esc(content.title || data.item || "");
    const overview = esc(content.overview || "");
    const rating   = content.vote_average
      ? parseFloat(content.vote_average).toFixed(1)
      : null;
    const year     = (content.release_date || content.first_air_date || "").slice(0, 4);
    const typeIcon = content.media_type === "movie" ? "🎬 Película" : "📺 Serie";

    const posterHTML = content.poster_path
      ? `<img src="${TMDB_IMG}${content.poster_path}" alt="${title}" loading="lazy">`
      : `<span>🎬</span>`;

    const starsHTML = rating
      ? `<div class="stars">
           ${"⭐".repeat(Math.round(rating / 2))}
           <span style="color:var(--muted);font-size:.8rem"> ${rating}/10</span>
         </div>`
      : "";

    cardHTML = `
      <div class="movie-card">
        <div class="movie-poster">${posterHTML}</div>
        <div class="movie-info">
          <div class="movie-title">${title}</div>
          <div class="movie-meta">${typeIcon}${year ? " · " + year : ""}</div>
          ${overview ? `<div class="movie-overview">${overview}</div>` : ""}
          ${starsHTML}
        </div>
      </div>`;
  }

  // ── Botones de respuesta rápida ──
  let qrHTML = "";
  if (action === "RECOMMEND" || action === "ALTERNATIVE") {
    qrHTML = `
      <div class="quick-replies">
        <button class="qr-btn" onclick="sendQuick('Me parece bien, la veo')">👍 Me apetece</button>
        <button class="qr-btn" onclick="sendQuick('Dame otra opción')">🔀 Otra opción</button>
        <button class="qr-btn" onclick="sendQuick('No me apetece ese género')">❌ No me gusta</button>
      </div>`;
  }

  msgEl.innerHTML = `
    <div class="avatar">🤖</div>
    <div>
      <div class="bubble">${esc(message)}${cardHTML}</div>
      ${qrHTML}
    </div>`;

  wrap.appendChild(msgEl);
  scrollBottom();
}


// ── Helpers de UI ───────────────────────────────────────────────
function appendUserMsg(text) {
  const wrap  = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "msg user";
  msgEl.innerHTML = `
    <div class="avatar">🧑</div>
    <div class="bubble">${esc(text)}</div>`;
  wrap.appendChild(msgEl);
  scrollBottom();
}

function renderError(text) {
  const wrap  = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "msg bot error-bubble";
  msgEl.innerHTML = `
    <div class="avatar">⚠️</div>
    <div class="bubble">${esc(text)}</div>`;
  wrap.appendChild(msgEl);
  scrollBottom();
}

function setLoading(on) {
  isLoading = on;
  document.getElementById("sendBtn").disabled = on;
  document.getElementById("typingIndicator").classList.toggle("show", on);
  if (on) scrollBottom();
}

function hideWelcome() {
  if (!welcomeShown) return;
  const w = document.getElementById("welcomeScreen");
  if (w) w.style.display = "none";
  welcomeShown = false;
}

function scrollBottom() {
  const el = document.getElementById("messages");
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}

function sendQuick(text) {
  document.getElementById("userInput").value = text;
  sendMessage();
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

function esc(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Inicializar perfil y dar foco al input
initProfile();
document.getElementById("userInput").focus();
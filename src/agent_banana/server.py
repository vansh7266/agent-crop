from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

from .pipeline import AgentBananaApp
from .vision import decode_image_payload

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Banana Studio</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0c0f13;
      --surface: rgba(22, 27, 34, 0.85);
      --surface-2: rgba(30, 37, 48, 0.7);
      --border: rgba(99, 120, 150, 0.15);
      --border-hi: rgba(99, 120, 150, 0.3);
      --text: #e6edf3;
      --text-dim: rgba(230, 237, 243, 0.6);
      --text-muted: rgba(230, 237, 243, 0.4);
      --accent: #58a6ff;
      --green: #3fb950;
      --amber: #d29922;
      --purple: #bc8cff;
      --red: #f85149;
      --orange: #f0883e;
      --radius: 16px;
      --radius-sm: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 20px;
      background-image:
        radial-gradient(ellipse at 20% 0%, rgba(88,166,255,0.06), transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(188,140,255,0.04), transparent 50%);
    }
    .app { max-width: 1100px; margin: 0 auto; }

    .header {
      display: flex; align-items: center; gap: 14px;
      padding: 20px 0 24px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .header-icon {
      width: 42px; height: 42px; border-radius: 12px;
      background: linear-gradient(135deg, var(--amber), var(--orange));
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; flex-shrink: 0;
    }
    .header h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
    .header p { color: var(--text-dim); font-size: 0.85rem; margin-top: 2px; }

    .input-area {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      backdrop-filter: blur(16px);
      margin-bottom: 24px;
    }
    .input-row { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: end; }
    .input-group { display: grid; gap: 10px; }
    .input-group label {
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: var(--text-dim); font-weight: 600;
    }
    textarea, input[type="text"] {
      width: 100%; border: 1px solid var(--border); border-radius: var(--radius-sm);
      background: rgba(0,0,0,0.3); padding: 12px 14px; color: var(--text);
      font: 0.9rem/1.5 'Inter', sans-serif; outline: none; transition: border 0.2s;
    }
    input[type="file"] { font: 0.85rem 'Inter', sans-serif; color: var(--text-dim); }
    textarea { min-height: 70px; resize: vertical; }
    textarea:focus, input[type="text"]:focus { border-color: var(--accent); }
    .file-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button, .btn {
      border: 0; border-radius: var(--radius-sm); padding: 12px 24px; cursor: pointer;
      font: 600 0.9rem 'Inter', sans-serif; transition: all 0.2s;
      background: linear-gradient(135deg, var(--accent), #388bfd); color: #fff;
      box-shadow: 0 4px 16px rgba(88,166,255,0.2); display: inline-block; text-align: center;
    }
    button:hover, .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(88,166,255,0.3); }
    button:disabled, .btn:disabled { opacity: 0.5; cursor: wait; transform: none; }
    .btn-sm { padding: 8px 16px; font-size: 0.82rem; }
    .btn-green { background: linear-gradient(135deg, var(--green), #2ea043); box-shadow: 0 4px 16px rgba(63,185,80,0.2); }
    .btn-amber { background: linear-gradient(135deg, var(--amber), var(--orange)); box-shadow: 0 4px 16px rgba(210,153,34,0.2); }

    .timeline { display: grid; gap: 0; }
    .agent-step {
      position: relative; padding-left: 40px; padding-bottom: 4px;
      animation: fadeSlideIn 0.4s ease-out both;
    }
    .agent-step::before {
      content: ''; position: absolute; left: 15px; top: 32px; bottom: 0;
      width: 2px; background: var(--border);
    }
    .agent-step:last-child::before { display: none; }
    .step-dot {
      position: absolute; left: 8px; top: 12px; width: 16px; height: 16px;
      border-radius: 50%; border: 2px solid var(--border-hi);
      background: var(--bg); z-index: 1;
      display: flex; align-items: center; justify-content: center;
    }
    .step-dot.active { border-color: var(--accent); background: rgba(88,166,255,0.15); }
    .step-dot.done { border-color: var(--green); background: rgba(63,185,80,0.15); }
    .step-dot.done::after { content: '\2713'; font-size: 9px; color: var(--green); }
    .step-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 16px; margin-bottom: 12px;
      backdrop-filter: blur(12px); transition: border 0.3s;
    }
    .step-card:hover { border-color: var(--border-hi); }
    .step-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .step-label {
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em;
      font-weight: 700; padding: 3px 8px; border-radius: 6px;
    }
    .step-label.planning { background: rgba(188,140,255,0.15); color: var(--purple); }
    .step-label.preview { background: rgba(240,136,62,0.15); color: var(--orange); }
    .step-label.llm { background: rgba(88,166,255,0.15); color: var(--accent); }
    .step-label.grounding { background: rgba(63,185,80,0.15); color: var(--green); }
    .step-label.compose { background: rgba(210,153,34,0.15); color: var(--amber); }
    .step-title { font-weight: 600; font-size: 0.95rem; }
    .step-body { color: var(--text-dim); font-size: 0.88rem; line-height: 1.6; }

    .llm-reasoning {
      background: rgba(88,166,255,0.05); border: 1px solid rgba(88,166,255,0.15);
      border-radius: var(--radius-sm); padding: 14px; margin-top: 10px;
    }
    .thinking-header {
      display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
      font-weight: 600; font-size: 0.85rem; color: var(--accent);
    }
    .llm-text {
      font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
      color: var(--text); line-height: 1.7; white-space: pre-wrap;
    }
    .phrase-tag {
      display: inline-block; padding: 3px 10px; border-radius: 20px;
      background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.2);
      font-size: 0.78rem; color: var(--accent); margin: 3px 4px 3px 0;
      font-family: 'JetBrains Mono', monospace;
    }
    .confidence-bar {
      height: 6px; border-radius: 3px; background: rgba(255,255,255,0.06);
      margin-top: 8px; overflow: hidden;
    }
    .confidence-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }

    .image-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px; margin-top: 10px;
    }
    .img-card {
      background: rgba(0,0,0,0.2); border: 1px solid var(--border);
      border-radius: var(--radius-sm); overflow: hidden;
    }
    .img-card img { width: 100%; display: block; min-height: 160px; object-fit: contain; background: #111; }
    .img-label {
      padding: 8px 12px; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.1em; color: var(--text-muted); font-weight: 600;
    }
    .meta-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px; margin-top: 10px;
    }
    .meta-item { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 10px 12px; }
    .meta-key {
      font-size: 0.68rem; text-transform: uppercase; color: var(--text-muted);
      letter-spacing: 0.08em;
    }
    .meta-val { font-size: 0.9rem; font-weight: 600; margin-top: 2px; }
    .quality-note {
      font-size: 0.82rem; color: var(--text-dim); padding: 4px 0;
      border-left: 2px solid var(--border-hi); padding-left: 10px; margin: 4px 0;
    }

    /* BBox editor */
    .bbox-editor-wrap {
      position: relative; display: inline-block; margin-top: 10px;
      border: 1px solid var(--border); border-radius: var(--radius-sm);
      overflow: hidden; background: #111;
    }
    .bbox-editor-wrap img { display: block; max-width: 100%; }
    .bbox-editor-wrap canvas {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      cursor: crosshair;
    }
    .bbox-controls {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
      margin-top: 10px;
    }
    .bbox-coords {
      font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
      color: var(--text-dim); padding: 6px 10px;
      background: rgba(0,0,0,0.3); border-radius: 6px;
    }

    @keyframes fadeSlideIn {
      from { opacity: 0; transform: translateY(12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 700px) {
      body { padding: 10px; }
      .input-row { grid-template-columns: 1fr; }
      .file-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="header">
      <div class="header-icon">&#127820;</div>
      <div>
        <h1>Agent Banana Studio</h1>
        <p>AI image editing agent &mdash; watch the reasoning steps unfold</p>
      </div>
    </header>

    <section class="input-area">
      <div class="input-group">
        <div class="file-row">
          <div>
            <label>Image</label>
            <input id="imageInput" type="file" accept="image/*">
          </div>
          <div>
            <label>Session ID (optional)</label>
            <input id="sessionId" type="text" placeholder="For multi-turn edits">
          </div>
        </div>
        <label>Edit Instruction</label>
        <div class="input-row">
          <textarea id="instruction" placeholder="Describe what you want to change...">Remove the glasses from the table.</textarea>
          <button id="runButton">Run Agent</button>
        </div>
      </div>
    </section>

    <div id="timeline" class="timeline"></div>
  </div>

  <script>
    var imageInput = document.getElementById("imageInput");
    var instructionInput = document.getElementById("instruction");
    var sessionInput = document.getElementById("sessionId");
    var runButton = document.getElementById("runButton");
    var timeline = document.getElementById("timeline");

    /* Stores for recompose */
    var _lastSourcePayload = null;
    var _iterCount = 0;
    var _lastStepPreviews = {};

    function dataUrlFromFile(file) {
      return new Promise(function(resolve, reject) {
        var reader = new FileReader();
        reader.onload = function() { resolve(reader.result); };
        reader.onerror = function() { reject(new Error("Failed to read image.")); };
        reader.readAsDataURL(file);
      });
    }

    function esc(s) {
      var d = document.createElement("div");
      d.textContent = s;
      return d.innerHTML;
    }

    function addStep(type, title, bodyHtml, dotClass) {
      dotClass = dotClass || "done";
      var step = document.createElement("div");
      step.className = "agent-step";
      step.style.animationDelay = (timeline.children.length * 0.08) + "s";
      step.innerHTML =
        '<div class="step-dot ' + dotClass + '"></div>' +
        '<div class="step-card">' +
          '<div class="step-header">' +
            '<span class="step-label ' + type + '">' + type + '</span>' +
            '<span class="step-title">' + title + '</span>' +
          '</div>' +
          '<div class="step-body">' + bodyHtml + '</div>' +
        '</div>';
      timeline.appendChild(step);
      step.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return step;
    }

    function confColor(c) {
      if (c >= 0.8) return "var(--green)";
      if (c >= 0.5) return "var(--amber)";
      return "var(--red)";
    }

    /* ---- Interactive BBox editor ---- */
    function createBboxEditor(previewSrc, initBbox, imgW, imgH, stepIdx, stepData) {
      var editorId = "bbox-editor-" + stepIdx;
      var html =
        '<div style="font-size:0.78rem;color:var(--text-muted);font-weight:600;margin-bottom:6px">ADJUST BOUNDING BOX</div>' +
        '<div style="font-size:0.78rem;color:var(--text-dim);margin-bottom:8px">Click and drag to draw a new bounding box on the preview image, then click Re-compose.</div>' +
        '<div class="bbox-editor-wrap" id="wrap-' + editorId + '">' +
          '<img id="img-' + editorId + '" src="' + previewSrc + '" alt="Preview">' +
          '<canvas id="cvs-' + editorId + '"></canvas>' +
        '</div>' +
        '<div class="bbox-controls">' +
          '<span class="bbox-coords" id="coords-' + editorId + '">' +
            '[' + initBbox.left + ', ' + initBbox.top + '] \u2192 [' + initBbox.right + ', ' + initBbox.bottom + ']' +
          '</span>' +
          '<div style="margin-top:8px;display:flex;gap:8px;align-items:center">' +
          '<input type="text" id="instr-' + editorId + '" placeholder="Custom instruction (e.g. fill with table texture)" ' +
            'style="flex:1;padding:6px 10px;border-radius:6px;border:1px solid var(--border);background:var(--bg-card);color:var(--text);font-size:0.82rem" />' +
          '<button class="btn btn-sm btn-green" id="recomp-' + editorId + '">Re-compose</button>' +
        '</div>' +
        '</div>';

      var wrapStep = addStep("grounding", "Edit Bounding Box (Step " + (stepIdx + 1) + ")", html);

      /* Wait for image to load, then setup canvas */
      setTimeout(function() {
        var img = document.getElementById("img-" + editorId);
        var cvs = document.getElementById("cvs-" + editorId);
        var coordsEl = document.getElementById("coords-" + editorId);
        var recompBtn = document.getElementById("recomp-" + editorId);
        if (!img || !cvs) return;

        function setup() {
          var dispW = img.clientWidth, dispH = img.clientHeight;
          cvs.width = dispW; cvs.height = dispH;
          var scaleX = dispW / imgW, scaleY = dispH / imgH;
          var ctx = cvs.getContext("2d");

          var box = {
            x1: initBbox.left * scaleX, y1: initBbox.top * scaleY,
            x2: initBbox.right * scaleX, y2: initBbox.bottom * scaleY
          };
          var drawing = false;

          function drawBox() {
            ctx.clearRect(0, 0, dispW, dispH);
            ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            /* semi-transparent fill */
            ctx.fillStyle = "rgba(88,166,255,0.12)";
            ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            /* corner handles */
            ctx.fillStyle = "#58a6ff";
            [[box.x1,box.y1],[box.x2,box.y1],[box.x1,box.y2],[box.x2,box.y2]].forEach(function(pt) {
              ctx.beginPath(); ctx.arc(pt[0], pt[1], 4, 0, Math.PI*2); ctx.fill();
            });
            /* update coords display in real image pixels */
            var realBox = getRealBox();
            coordsEl.textContent = '[' + realBox.left + ', ' + realBox.top + '] \u2192 [' + realBox.right + ', ' + realBox.bottom + '] (' + (realBox.right-realBox.left) + '\u00d7' + (realBox.bottom-realBox.top) + 'px)';
          }

          function getRealBox() {
            var l = Math.round(Math.min(box.x1, box.x2) / scaleX);
            var t = Math.round(Math.min(box.y1, box.y2) / scaleY);
            var r = Math.round(Math.max(box.x1, box.x2) / scaleX);
            var b = Math.round(Math.max(box.y1, box.y2) / scaleY);
            return { left: Math.max(0,l), top: Math.max(0,t), right: Math.min(imgW,r), bottom: Math.min(imgH,b) };
          }

          cvs.addEventListener("mousedown", function(e) {
            var rect = cvs.getBoundingClientRect();
            box.x1 = e.clientX - rect.left; box.y1 = e.clientY - rect.top;
            box.x2 = box.x1; box.y2 = box.y1;
            drawing = true;
          });
          cvs.addEventListener("mousemove", function(e) {
            if (!drawing) return;
            var rect = cvs.getBoundingClientRect();
            box.x2 = e.clientX - rect.left; box.y2 = e.clientY - rect.top;
            drawBox();
          });
          cvs.addEventListener("mouseup", function() { drawing = false; });

          recompBtn.addEventListener("click", function() {
            var realBox = getRealBox();
            if (realBox.right - realBox.left < 5 || realBox.bottom - realBox.top < 5) return;
            recompBtn.disabled = true;
            recompBtn.textContent = "Re-composing\u2026";
            fetch("/api/recompose", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                source_image: _lastSourcePayload,
                preview_image: _lastSourcePayload,
                bbox: realBox,
                target: stepData.step.target,
                verb: stepData.step.verb,
                custom_instruction: (document.getElementById("instr-" + editorId) || {}).value || "",
              })
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
              if (data.error) throw new Error(data.error);
              var q = data.quality;
              var statusColor = q.accepted ? "var(--green)" : "var(--red)";
              var statusText = q.accepted ? "\u2713 Accepted" : "\u2717 Rejected";
              var notesHtml = (q.notes || []).map(function(n) { return '<div class="quality-note">' + esc(n) + '</div>'; }).join("");
              addStep("compose", "Re-composed with manual bbox",
                '<div class="image-grid">' +
                  '<div class="img-card"><img src="' + data.overlay_image + '" alt="Overlay"><div class="img-label">Adjusted Region</div></div>' +
                  '<div class="img-card"><img src="' + data.final_image + '" alt="Result"><div class="img-label">New Output</div></div>' +
                '</div>' +
                '<div class="meta-grid">' +
                  '<div class="meta-item"><div class="meta-key">Quality</div><div class="meta-val" style="color:' + statusColor + '">' + statusText + '</div></div>' +
                  '<div class="meta-item"><div class="meta-key">Score</div><div class="meta-val">' + q.score.toFixed(3) + '</div></div>' +
                  '<div class="meta-item"><div class="meta-key">Inside \u0394</div><div class="meta-val">' + q.inside_change.toFixed(3) + '</div></div>' +
                  '<div class="meta-item"><div class="meta-key">Outside \u0394</div><div class="meta-val">' + q.outside_change.toFixed(3) + '</div></div>' +
                  '<div class="meta-item"><div class="meta-key">BBox</div><div class="meta-val" style="font-size:0.72rem">' + data.bbox.left + ',' + data.bbox.top + ' \u2192 ' + data.bbox.right + ',' + data.bbox.bottom + '</div></div>' +
                '</div>' +
                (notesHtml ? '<div style="margin-top:8px">' + notesHtml + '</div>' : ''));
              recompBtn.disabled = false;
              recompBtn.textContent = "Re-compose";
              /* Iterative loop: update source to new output and show a new bbox editor */
              _lastSourcePayload = data.final_image;
              _iterCount = (_iterCount || 0) + 1;
              var newBbox = data.bbox;
              createBboxEditor(data.final_image, newBbox, stepData.image_width || imgW, stepData.image_height || imgH, stepIdx + "_iter" + _iterCount, stepData);
            })
            .catch(function(err) {
              addStep("compose", "Recompose Error", '<div style="color:var(--red)">' + esc(err.message) + '</div>');
              recompBtn.disabled = false;
              recompBtn.textContent = "Re-compose with this box";
            });
          });

          drawBox();
        }

        if (img.complete) { setup(); }
        else { img.onload = setup; }
      }, 100);
    }

    /* ---- Main pipeline run ---- */
    async function runPipeline() {
      var file = imageInput.files[0];
      var instruction = instructionInput.value.trim();
      if (!file || !instruction) return;
      runButton.disabled = true;
      timeline.innerHTML = "";
      _lastStepPreviews = {};

      var imagePayload = await dataUrlFromFile(file);
      _lastSourcePayload = imagePayload;

      addStep("planning", "Received image & instruction",
        '<div class="image-grid"><div class="img-card">' +
        '<img src="' + imagePayload + '" alt="Source"><div class="img-label">Source Image</div></div></div>' +
        '<div style="margin-top:10px"><strong>Instruction:</strong> <span style="color:var(--accent)">&quot;' + esc(instruction) + '&quot;</span></div>');

      addStep("planning", "Parsing instruction & generating plan\u2026",
        '<div style="color:var(--text-muted)">Decomposing edits, ranking candidate plans, selecting optimal path\u2026</div>', "active");

      try {
        var response = await fetch("/api/edit", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ image: imagePayload, instruction: instruction, session_id: sessionInput.value.trim() || null })
        });
        var payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Request failed");
        sessionInput.value = payload.session_id;

        timeline.removeChild(timeline.lastChild);

        var plan = payload.selected_plan;
        var stepsHtml = plan.steps.map(function(s) {
          return '<div style="padding:6px 0;border-bottom:1px solid var(--border)">' +
            '<strong style="color:var(--purple)">' + s.order + '.</strong> ' + esc(s.verb) + ' <strong>' + esc(s.target) + '</strong>' +
            '<span style="color:var(--text-muted);font-size:0.8rem;margin-left:8px">[' + s.mode + ']</span></div>';
        }).join("");
        addStep("planning", "Selected plan: " + plan.plan_id + " (score " + plan.score.toFixed(3) + ")",
          stepsHtml +
          '<div class="meta-grid">' +
            '<div class="meta-item"><div class="meta-key">Candidates</div><div class="meta-val">' + payload.candidate_plans.length + '</div></div>' +
            '<div class="meta-item"><div class="meta-key">Edits parsed</div><div class="meta-val">' + payload.parsed_edits.length + '</div></div>' +
            '<div class="meta-item"><div class="meta-key">Session</div><div class="meta-val" style="font-size:0.72rem;word-break:break-all">' + payload.session_id.slice(0,12) + '\u2026</div></div>' +
          '</div>');

        payload.step_results.forEach(function(sr, idx) {
          var s = sr.step;
          _lastStepPreviews[idx] = sr.preview_image;

          /* Preview */
          addStep("preview", "Preview: " + esc(s.verb) + " " + esc(s.target),
            '<div class="image-grid"><div class="img-card">' +
            '<img src="' + sr.preview_image + '" alt="Preview"><div class="img-label">Gemini Preview (before bbox)</div></div></div>' +
            '<div style="margin-top:8px;font-size:0.82rem;color:var(--text-muted)">Mode: <strong style="color:var(--text)">' + esc(payload.mode) + '</strong></div>');

          /* LLM Advisor */
          if (sr.llm_object_description || (sr.llm_refined_phrases && sr.llm_refined_phrases.length > 0)) {
            var phraseTags = (sr.llm_refined_phrases || []).map(function(p) {
              return '<span class="phrase-tag">' + esc(p) + '</span>';
            }).join("");
            var bboxHtml = "";
            if (sr.llm_bbox_hint) {
              bboxHtml = '<div style="margin-top:8px;font-size:0.82rem">&#128205; <strong>Expected region:</strong> [' + sr.llm_bbox_hint.left + ', ' + sr.llm_bbox_hint.top + '] &rarr; [' + sr.llm_bbox_hint.right + ', ' + sr.llm_bbox_hint.bottom + '] (' + sr.llm_bbox_hint.width + '&times;' + sr.llm_bbox_hint.height + 'px)</div>';
            }
            var conf = sr.llm_confidence || 0;
            addStep("llm", "LLM Grounding Advisor",
              '<div class="llm-reasoning">' +
                '<div class="thinking-header">&#129504; Spatial Reasoning</div>' +
                '<div class="llm-text">' + esc(sr.llm_object_description || "No description returned.") + '</div>' +
              '</div>' +
              '<div style="margin-top:12px">' +
                '<div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:6px;font-weight:600">REFINED GROUNDING PHRASES</div>' +
                (phraseTags || '<span style="color:var(--text-muted);font-size:0.82rem">None generated</span>') +
              '</div>' +
              bboxHtml +
              '<div style="margin-top:10px">' +
                '<div style="font-size:0.78rem;color:var(--text-muted);display:flex;justify-content:space-between">' +
                  '<span>CONFIDENCE</span>' +
                  '<span style="color:' + confColor(conf) + ';font-weight:700">' + (conf * 100).toFixed(0) + '%</span>' +
                '</div>' +
                '<div class="confidence-bar"><div class="confidence-fill" style="width:' + (conf * 100) + '%;background:' + confColor(conf) + '"></div></div>' +
              '</div>');
          }

          /* Grounding */
          var candidates = sr.grounding_candidates || [];
          var candHtml = candidates.slice(0, 3).map(function(c) {
            return '<div style="padding:4px 0;font-size:0.82rem">' +
              '<span class="phrase-tag" style="background:rgba(63,185,80,0.1);border-color:rgba(63,185,80,0.2);color:var(--green)">' + esc(c.phrase) + '</span> ' +
              'score ' + c.score.toFixed(2) + ' &middot; [' + c.bbox.left + ',' + c.bbox.top + ',' + c.bbox.right + ',' + c.bbox.bottom + ']</div>';
          }).join("");
          if (!candHtml) candHtml = '<span style="color:var(--text-muted);font-size:0.82rem">No candidates found</span>';
          var phrasesSent = (sr.grounding_phrases || []).map(function(p) {
            return '<span class="phrase-tag" style="background:rgba(63,185,80,0.06);border-color:rgba(63,185,80,0.15);color:var(--green)">' + esc(p) + '</span>';
          }).join("");
          addStep("grounding", "Florence-2 Grounding",
            '<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:8px">Mode: <strong style="color:var(--text)">' + esc(sr.localizer_mode) + '</strong></div>' +
            '<div style="margin-bottom:8px;font-size:0.78rem;color:var(--text-muted);font-weight:600">PHRASES SENT</div>' +
            '<div>' + phrasesSent + '</div>' +
            '<div style="margin-top:10px;font-size:0.78rem;color:var(--text-muted);font-weight:600">TOP CANDIDATES</div>' +
            candHtml +
            '<div class="image-grid" style="margin-top:10px"><div class="img-card">' +
            '<img src="' + sr.overlay_image + '" alt="Overlay"><div class="img-label">Detected Region</div></div></div>');
          /* Interactive BBox editor — inline between grounding and composition */
          createBboxEditor(_lastSourcePayload, sr.bbox, sr.image_width || 512, sr.image_height || 512, idx, sr);

          /* Composition */
          var q = sr.quality;
          var statusColor = q.accepted ? "var(--green)" : "var(--red)";
          var statusText = q.accepted ? "\u2713 Accepted" : "\u2717 Rejected";
          var notesHtml = (q.notes || []).map(function(n) { return '<div class="quality-note">' + esc(n) + '</div>'; }).join("");
          addStep("compose", "Composed: " + esc(s.verb) + " " + esc(s.target),
            '<div class="image-grid"><div class="img-card">' +
            '<img src="' + sr.edited_image + '" alt="Result"><div class="img-label">Final Composition</div></div></div>' +
            '<div class="meta-grid">' +
              '<div class="meta-item"><div class="meta-key">Quality</div><div class="meta-val" style="color:' + statusColor + '">' + statusText + '</div></div>' +
              '<div class="meta-item"><div class="meta-key">Score</div><div class="meta-val">' + q.score.toFixed(3) + '</div></div>' +
              '<div class="meta-item"><div class="meta-key">Inside \u0394</div><div class="meta-val">' + q.inside_change.toFixed(3) + '</div></div>' +
              '<div class="meta-item"><div class="meta-key">Outside \u0394</div><div class="meta-val">' + q.outside_change.toFixed(3) + '</div></div>' +
              '<div class="meta-item"><div class="meta-key">Preview Align</div><div class="meta-val">' + q.preview_alignment.toFixed(3) + '</div></div>' +
              '<div class="meta-item"><div class="meta-key">BBox</div><div class="meta-val" style="font-size:0.72rem">' + sr.bbox.left + ',' + sr.bbox.top + ' \u2192 ' + sr.bbox.right + ',' + sr.bbox.bottom + '</div></div>' +
            '</div>' +
            (notesHtml ? '<div style="margin-top:8px">' + notesHtml + '</div>' : ''));
        });

        /* Final summary */
        addStep("compose", "Done \u2014 Reward " + payload.reward.toFixed(3),
          '<div class="image-grid">' +
            '<div class="img-card"><img src="' + payload.source_image + '" alt="Source"><div class="img-label">Original</div></div>' +
            '<div class="img-card"><img src="' + payload.final_image + '" alt="Final"><div class="img-label">Final Output</div></div>' +
          '</div>' +
          '<div class="meta-grid" style="margin-top:12px">' +
            '<div class="meta-item"><div class="meta-key">Image Mode</div><div class="meta-val">' + esc(payload.mode) + '</div></div>' +
            '<div class="meta-item"><div class="meta-key">Grounding</div><div class="meta-val">' + esc(payload.grounding_mode) + '</div></div>' +
            '<div class="meta-item"><div class="meta-key">Reward</div><div class="meta-val" style="color:var(--green)">' + payload.reward.toFixed(3) + '</div></div>' +
            '<div class="meta-item"><div class="meta-key">Steps</div><div class="meta-val">' + payload.step_results.length + '</div></div>' +
          '</div>');

      } catch (err) {
        addStep("compose", "Error", '<div style="color:var(--red)">' + esc(err.message) + '</div>');
      } finally {
        runButton.disabled = false;
      }
    }

    runButton.addEventListener("click", runPipeline);
  </script>
</body>
</html>
"""


def make_handler(app: AgentBananaApp) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(HTML_PAGE)
                return
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON payload"})
                return

            if self.path == "/api/edit":
                instruction = str(payload.get("instruction", "")).strip()
                image_payload = str(payload.get("image", "")).strip()
                session_id = payload.get("session_id") or None
                if not instruction:
                    self._send_json(400, {"error": "Instruction is required"})
                    return
                if not image_payload:
                    self._send_json(400, {"error": "Image payload is required"})
                    return
                try:
                    image = decode_image_payload(image_payload)
                    result = app.run(image, instruction, session_id=session_id)
                except Exception as exc:  # pragma: no cover
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, result.to_dict())

            elif self.path == "/api/recompose":
                source_payload = str(payload.get("source_image", "")).strip()
                preview_payload = str(payload.get("preview_image", "")).strip()
                print(f"[agent-banana] recompose: source_len={len(source_payload)}, preview_len={len(preview_payload)}, bbox={payload.get('bbox')}")
                bbox_data = payload.get("bbox")
                target = str(payload.get("target", "object"))
                verb = str(payload.get("verb", "edit"))
                if not source_payload or not bbox_data:
                    self._send_json(400, {"error": "source_image and bbox are required"})
                if not preview_payload:
                    preview_payload = source_payload
                    return
                try:
                    source_image = decode_image_payload(source_payload)
                    preview_image = decode_image_payload(preview_payload)
                    custom_instruction = str(payload.get("custom_instruction", "")).strip()
                    result = app.recompose(source_image, preview_image, bbox_data, target, verb, custom_instruction=custom_instruction)
                except Exception as exc:  # pragma: no cover
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, result)

            else:
                self._send_json(404, {"error": "Not found"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"[agent-banana] {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Serve the Agent Banana image editing demo over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    app = AgentBananaApp.from_env(default_root)
    server = HTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving Agent Banana Studio on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

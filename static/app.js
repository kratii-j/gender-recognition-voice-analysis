const recordBtn = document.getElementById("recordBtn");
const fileInput = document.getElementById("fileInput");
const statusText = document.getElementById("statusText");
const resultBox = document.getElementById("resultBox");
const historyBody = document.getElementById("historyBody");
const refreshBtn = document.getElementById("refreshBtn");
const totalPredictions = document.getElementById("totalPredictions");
const avgConfidence = document.getElementById("avgConfidence");
const maleCount = document.getElementById("maleCount");
const femaleCount = document.getElementById("femaleCount");
const insightsBox = document.getElementById("insightsBox");

recordBtn.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Microphone access is not supported in this browser.");
    return;
  }

  try {
    setStatus("Recording for 3 seconds...");
    recordBtn.disabled = true;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const chunks = [];

    processor.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    const sampleRate = audioContext.sampleRate;

    await new Promise((resolve) => setTimeout(resolve, 3000));

    stream.getTracks().forEach((track) => track.stop());
    processor.disconnect();
    source.disconnect();
    await audioContext.close();

    const wavBlob = encodeWav(chunks, sampleRate);
    await submitFile(new File([wavBlob], "recorded.wav", { type: "audio/wav" }));
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Recording failed.");
  } finally {
    recordBtn.disabled = false;
  }
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  await submitFile(file);
  fileInput.value = "";
});

refreshBtn.addEventListener("click", loadHistory);

async function submitFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  setStatus("Analyzing audio...");

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    setStatus(payload.detail || "Prediction failed.");
    return;
  }

  renderResult(payload);
  setStatus("Prediction complete.");
  await loadHistory();
}

function renderResult(payload) {
  resultBox.classList.remove("empty");
  resultBox.innerHTML = `
    <div><strong>Label:</strong> ${payload.label}</div>
    <div><strong>Confidence:</strong> ${Math.round(payload.confidence * 100)}%</div>
    <div><strong>Model:</strong> ${payload.model_type || "n/a"}</div>
    <div><strong>Dataset:</strong> ${payload.dataset_name || "n/a"}</div>
    <div><strong>Saved test accuracy:</strong> ${
      payload.test_accuracy ? `${Math.round(payload.test_accuracy * 100)}%` : "n/a"
    }</div>
    <div><strong>Mix:</strong> ${payload.male_percent}% male and ${payload.female_percent}% female</div>
    <div><strong>Decision:</strong> taking the majority as ${payload.label}</div>
    <div><strong>Estimated pitch:</strong> ${payload.pitch_hz} Hz</div>
    <div><strong>RMS energy:</strong> ${payload.rms_energy}</div>
    <div><strong>Zero-crossing rate:</strong> ${payload.zero_crossing_rate}</div>
    <div><strong>Duration:</strong> ${payload.duration_seconds}s</div>
  `;
}

async function loadHistory() {
  const response = await fetch("/api/history");
  const rows = await response.json();
  renderOverview(rows);
  renderInsights(rows);
  if (!rows.length) {
    historyBody.innerHTML = `<tr><td colspan="6">No saved predictions yet.</td></tr>`;
    return;
  }

  historyBody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${new Date(row.created_at).toLocaleString()}</td>
        <td>${row.filename}</td>
        <td>${row.label}</td>
        <td>${Math.round(row.confidence * 100)}%</td>
        <td>${row.pitch_hz} Hz</td>
        <td>${row.duration_seconds}s</td>
      </tr>
    `,
    )
    .join("");
}

function renderOverview(rows) {
  const total = rows.length;
  const average = total ? rows.reduce((sum, row) => sum + row.confidence, 0) / total : 0;
  const male = rows.filter((row) => row.label === "male").length;
  const female = rows.filter((row) => row.label === "female").length;

  totalPredictions.textContent = String(total);
  avgConfidence.textContent = `${Math.round(average * 100)}%`;
  maleCount.textContent = String(male);
  femaleCount.textContent = String(female);
}

function renderInsights(rows) {
  if (!rows.length) {
    insightsBox.innerHTML = `<p class="subtle">Insights will appear after predictions are saved.</p>`;
    return;
  }

  const latest = rows[0];
  const averagePitch = rows.reduce((sum, row) => sum + row.pitch_hz, 0) / rows.length;
  const dominantLabel =
    rows.filter((row) => row.label === "male").length >= rows.filter((row) => row.label === "female").length
      ? "male"
      : "female";

  insightsBox.innerHTML = `
    <div class="insight-item">Latest prediction favors <strong>${latest.label}</strong> with ${Math.round(latest.confidence * 100)}% confidence.</div>
    <div class="insight-item">Average estimated pitch across saved samples is <strong>${averagePitch.toFixed(1)} Hz</strong>.</div>
    <div class="insight-item">Saved history currently trends toward <strong>${dominantLabel}</strong> as the majority class.</div>
  `;
}

function setStatus(message) {
  statusText.textContent = message;
}

function encodeWav(chunks, sampleRate) {
  const merged = mergeBuffers(chunks);
  const buffer = new ArrayBuffer(44 + merged.length * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + merged.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, merged.length * 2, true);

  let offset = 44;
  for (let index = 0; index < merged.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, merged[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function mergeBuffers(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

loadHistory().catch(() => {
  historyBody.innerHTML = `<tr><td colspan="6">Could not load history.</td></tr>`;
});

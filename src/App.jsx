import React, { useEffect, useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs"; // pulls in TFJS + WebGL backend

// ---- Shared (singleton) detector ----
// React 18 StrictMode mounts/unmounts components twice in development.
// MediaPipe WASM solutions can crash if initialized twice.
// We keep a single detector instance for the whole module.
let sharedDetector = null;
let sharedDetectorPromise = null;

// A couple of tiny default overlays (data URLs) so you can run instantly.
// You can replace these with your own PNGs, or upload via the UI.
const DEFAULT_HAT_PNG = "data:image/svg+xml;base64," + btoa(`
<svg xmlns="http://www.w3.org/2000/svg" width="600" height="360" viewBox="0 0 600 360">
  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="6" stdDeviation="8" flood-opacity="0.25"/>
    </filter>
  </defs>
  <g filter="url(#s)">
    <path d="M80 290c70 30 370 30 440 0 0 35-120 70-220 70S80 325 80 290z" fill="#111"/>
    <path d="M170 290c0-70 55-190 130-190s130 120 130 190" fill="#111"/>
    <rect x="150" y="235" width="300" height="45" rx="18" fill="#2b6cb0"/>
  </g>
</svg>
`);

const DEFAULT_GLASSES_PNG = "data:image/svg+xml;base64," + btoa(`
<svg xmlns="http://www.w3.org/2000/svg" width="700" height="240" viewBox="0 0 700 240">
  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="6" stdDeviation="6" flood-opacity="0.25"/>
    </filter>
  </defs>
  <g filter="url(#s)" fill="none" stroke="#111" stroke-width="18" stroke-linecap="round" stroke-linejoin="round">
    <rect x="40" y="60" width="250" height="140" rx="60"/>
    <rect x="410" y="60" width="250" height="140" rx="60"/>
    <path d="M290 130h120"/>
  </g>
  <g opacity="0.12">
    <rect x="55" y="75" width="220" height="110" rx="50" fill="#111"/>
    <rect x="425" y="75" width="220" height="110" rx="50" fill="#111"/>
  </g>
</svg>
`);

const DEFAULT_MUSTACHE_PNG = "data:image/svg+xml;base64," + btoa(`
<svg xmlns="http://www.w3.org/2000/svg" width="500" height="220" viewBox="0 0 500 220">
  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="6" stdDeviation="7" flood-opacity="0.25"/>
    </filter>
  </defs>
  <g filter="url(#s)" fill="#111">
    <path d="M250 130c-35 0-55-25-75-25-25 0-40 20-70 20-35 0-70-25-70-55 0-22 15-40 40-45 25-5 55 10 80 25 25 15 40 25 95 25 55 0 70-10 95-25 25-15 55-30 80-25 25 5 40 23 40 45 0 30-35 55-70 55-30 0-45-20-70-20-20 0-40 25-75 25z"/>
  </g>
</svg>
`);

const DEFAULT_MASK_PNG = "data:image/svg+xml;base64," + btoa(`
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="360" viewBox="0 0 800 360">
  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-opacity="0.25"/>
    </filter>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#7c3aed"/>
      <stop offset="0.5" stop-color="#06b6d4"/>
      <stop offset="1" stop-color="#f59e0b"/>
    </linearGradient>
  </defs>

  <!-- playful costume mask -->
  <g filter="url(#s)">
    <!-- mask body -->
    <path d="M90 190c0-70 75-130 170-145 60-10 140-10 200 0 95 15 170 75 170 145 0 60-45 105-115 125-75 22-165 32-255 32s-180-10-255-32C135 295 90 250 90 190z" fill="url(#g)"/>

    <!-- eye holes -->
    <g fill="#000" opacity="0.22">
      <path d="M220 205c0-35 28-62 64-62 40 0 72 28 72 62 0 34-32 58-72 58-36 0-64-23-64-58z"/>
      <path d="M444 205c0-35 28-62 64-62 40 0 72 28 72 62 0 34-32 58-72 58-36 0-64-23-64-58z"/>
    </g>

    <!-- outline + bridge -->
    <path d="M90 190c0-70 75-130 170-145 60-10 140-10 200 0 95 15 170 75 170 145 0 60-45 105-115 125-75 22-165 32-255 32s-180-10-255-32C135 295 90 250 90 190z" fill="none" stroke="#111" stroke-width="14" stroke-linejoin="round"/>
    <path d="M378 196c22-18 48-18 70 0" fill="none" stroke="#111" stroke-width="14" stroke-linecap="round"/>

    <!-- little decorative dots -->
    <g fill="#fff" opacity="0.85">
      <circle cx="160" cy="210" r="10"/>
      <circle cx="620" cy="210" r="10"/>
      <circle cx="190" cy="255" r="8"/>
      <circle cx="590" cy="255" r="8"/>
      <circle cx="400" cy="285" r="6"/>
    </g>

    <!-- straps -->
    <path d="M120 210c-60 20-95 45-110 78" fill="none" stroke="#111" stroke-width="14" stroke-linecap="round"/>
    <path d="M680 210c60 20 95 45 110 78" fill="none" stroke="#111" stroke-width="14" stroke-linecap="round"/>
  </g>
</svg>
`);

const LOGO_DATA_PATH = "/c13f1e65bbb212a8855d.svg";

function useImage(src) {
  const [img, setImg] = useState(null);

  useEffect(() => {
    if (!src) return setImg(null);
    const i = new Image();
    i.onload = () => setImg(i);
    i.onerror = () => setImg(null);
    i.src = src;
  }, [src]);

  return img;
}

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraContainerRef = useRef(null);

  const [isFullscreen, setIsFullscreen] = useState(false);

  const [status, setStatus] = useState("loading model…");
  const [filter, setFilter] = useState("hat"); // hat | glasses | mustache | custom | none
  const [customPng, setCustomPng] = useState(null);

  // tuning knobs (kept minimal)
  const [scale, setScale] = useState(1.25);
  const [xOffset, setXOffset] = useState(0);
  const [yOffset, setYOffset] = useState(-20);

  const hatImg = useImage(DEFAULT_HAT_PNG);
  const glassesImg = useImage(DEFAULT_GLASSES_PNG);
  const mustacheImg = useImage(DEFAULT_MUSTACHE_PNG);
  const customImg = useImage(customPng);
  const maskImg = useImage(DEFAULT_MASK_PNG);
  const logoImg = useImage(LOGO_DATA_PATH);

  const activeOverlayImg = useMemo(() => {
    if (filter === "hat") return hatImg;
    if (filter === "glasses") return glassesImg;
    if (filter === "mustache") return mustacheImg;
    if (filter === "custom") return customImg;
    if (filter === "mask") return maskImg;
    if (filter === "logo") return logoImg;
    return null;
  }, [filter, hatImg, glassesImg, mustacheImg, customImg, maskImg, logoImg]);

  const detectorRef = useRef(null);
  const rafRef = useRef(null);
  const busyRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setStatus("loading face model…");

        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;

        // Reuse a single detector across StrictMode remounts.
        if (sharedDetector) {
          if (!cancelled) {
            detectorRef.current = sharedDetector;
            setStatus("ready ✅");
          }
          return;
        }

        if (!sharedDetectorPromise) {
          sharedDetectorPromise = faceLandmarksDetection.createDetector(model, {
            runtime: "mediapipe",
            // Pin to a stable package path; jsDelivr serves files from the npm package.
            // (You can also pin a version like `@mediapipe/face_mesh@0.4.1646424915` if desired.)
            solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
            refineLandmarks: true,
            // Allow multiple faces (keep small for performance)
            maxFaces: 5,
          }).then((d) => {
            sharedDetector = d;
            return d;
          });
        }

        const detector = await sharedDetectorPromise;
        if (cancelled) return;
        detectorRef.current = detector;
        setStatus("ready ✅");
      } catch (e) {
        console.error(e);
        setStatus("failed to load model (check console)");
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      setIsFullscreen(Boolean(fsEl));
    };

    document.addEventListener("fullscreenchange", onFsChange);
    document.addEventListener("webkitfullscreenchange", onFsChange);

    // initialize once
    onFsChange();

    return () => {
      document.removeEventListener("fullscreenchange", onFsChange);
      document.removeEventListener("webkitfullscreenchange", onFsChange);
    };
  }, []);

  function toggleFullscreen() {
    const el = cameraContainerRef.current;
    if (!el) return;

    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (!fsEl) {
      const req = el.requestFullscreen || el.webkitRequestFullscreen;
      if (req) req.call(el);
      return;
    }

    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    if (exit) exit.call(document);
  }

  // MediaPipe FaceMesh landmark indices (commonly used for eye outer corners)
  // 33 = left eye outer corner, 263 = right eye outer corner
  const LEFT_EYE_IDX = 33;
  const RIGHT_EYE_IDX = 263;

  function getHeadTiltRadians(face) {
    const kps = face?.keypoints;
    if (!kps || kps.length <= RIGHT_EYE_IDX) return 0;

    const left = kps[LEFT_EYE_IDX];
    const right = kps[RIGHT_EYE_IDX];
    if (!left || !right) return 0;

    const dx = right.x - left.x;
    const dy = right.y - left.y;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return 0;

    return Math.atan2(dy, dx);
  }

  function drawImageRotated(ctx, img, x, y, w, h, angleRad) {
    const cx = x + w / 2;
    const cy = y + h / 2;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angleRad);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();
  }

  useEffect(() => {
    function draw() {
      const detector = detectorRef.current;
      const webcam = webcamRef.current;
      const canvas = canvasRef.current;

      if (!detector || !webcam || !canvas) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const video = webcam.video;
      if (!video || video.readyState !== 4) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const w = video.videoWidth;
      const h = video.videoHeight;

      // match canvas to video size
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      // Prevent overlapping calls (can happen if estimateFaces takes >1 frame).
      if (busyRef.current) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }
      busyRef.current = true;

      (async () => {
        try {
          const faces = await detector.estimateFaces(video, { flipHorizontal: true });

          if (!faces?.length || filter === "none" || !activeOverlayImg) {
            return;
          }

          for (const face of faces) {
            const box = face.box;
            if (!box) continue;

            const tilt = getHeadTiltRadians(face);

            // Base placement: compute an “anchor” per filter using the face bounding box.
            let targetW = box.width * scale;
            let targetH = targetW * (activeOverlayImg.height / activeOverlayImg.width);

            let x = box.xMin + (box.width - targetW) / 2;
            let y = box.yMin;

            if (filter === "hat") {
              // place above the face
              y = box.yMin - targetH * 0.85;
            } else if (filter === "glasses") {
              // upper third of face
              y = box.yMin + box.height * 0.20;
              targetW = box.width * (scale * 1.15);
              targetH = targetW * (activeOverlayImg.height / activeOverlayImg.width);
              x = box.xMin + (box.width - targetW) / 2;
            } else if (filter === "mustache") {
              // lower-middle of face
              y = box.yMin + box.height * 0.55;
              targetW = box.width * (scale * 0.85);
              targetH = targetW * (activeOverlayImg.height / activeOverlayImg.width);
              x = box.xMin + (box.width - targetW) / 2;
            } else if (filter === "custom") {
              // default to center of face
              y = box.yMin + (box.height - targetH) / 2;
            }

            // apply user offsets
            x += xOffset;
            y += yOffset;

            // draw overlay with head tilt
            drawImageRotated(ctx, activeOverlayImg, x, y, targetW, targetH, tilt);
          }
        } catch (err) {
          // If estimateFaces throws intermittently, just keep going.
          // (Often happens during camera start/stop transitions.)
        } finally {
          busyRef.current = false;
          rafRef.current = requestAnimationFrame(draw);
        }
      })(); 
    } 

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [filter, activeOverlayImg, scale, xOffset, yOffset]);

  function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setCustomPng(String(reader.result));
    reader.readAsDataURL(file);
  }

  return (
    <div style={{ padding: 16, maxWidth: 980, margin: "0 auto" }}>
      <h1 style={{ margin: "8px 0 4px" }}>Funny Filters (sup Rosie 👋)</h1>
      <div style={{ color: "#555", marginBottom: 12 }}>
        Status: <b>{status}</b>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 320px",
          gap: 16,
          alignItems: "start",
        }}
      >
        <div
          ref={cameraContainerRef}
          style={{
            position: "relative",
            width: "100%",
            aspectRatio: "16 / 9",
            background: "#000",
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <Webcam
            ref={webcamRef}
            audio={false}
            mirrored={true}
            screenshotFormat="image/png"
            videoConstraints={{ facingMode: "user" }}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />

          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
            }}
          />

          <img
            src={LOGO_DATA_PATH}
            alt="logo"
            style={{
              position: "absolute",
              right: 12,
              bottom: 12,
              width: 164,
              height: "auto",
              borderRadius: 10,
              opacity: 0.92,
              pointerEvents: "none",
              boxShadow: "0 8px 22px rgba(0,0,0,0.25)",
            }}
          />
        </div>

        <div
          style={{
            background: "#fff",
            border: "1px solid #eee",
            borderRadius: 12,
            padding: 12,
          }}
        >
          <div style={{ display: "grid", gap: 10 }}>
            <button
              type="button"
              onClick={toggleFullscreen}
              style={{
                padding: "10px 12px",
                borderRadius: 10,
                border: "1px solid #e6e6e6",
                background: "#f7f7f7",
                cursor: "pointer",
                fontWeight: 600, 
              }}
            >
              {isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            </button>
            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontWeight: 600 }}>Filter</div>
              <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="hat">Hat</option>
                <option value="glasses">Glasses</option>
                <option value="mustache">Mustache</option>
                <option value="custom">Custom PNG</option>
                <option value="mask">Mask</option>
                <option value="logo">Logo</option>
                <option value="none">None</option>
              </select>
            </label>

            {filter === "custom" && (
              <label style={{ display: "grid", gap: 6 }}>
                <div style={{ fontWeight: 600 }}>Upload transparent PNG</div>
                <input type="file" accept="image/png,image/webp,image/jpeg" onChange={onUpload} />
                <div style={{ fontSize: 12, color: "#666" }}>
                  Tip: transparent PNGs look best.
                </div>
              </label>
            )}

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontWeight: 600 }}>Scale</span>
                <span style={{ fontVariantNumeric: "tabular-nums" }}>{scale.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.6"
                max="2.2"
                step="0.01"
                value={scale}
                onChange={(e) => setScale(Number(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontWeight: 600 }}>X offset</span>
                <span style={{ fontVariantNumeric: "tabular-nums" }}>{xOffset}px</span>
              </div>
              <input
                type="range"
                min="-200"
                max="200"
                step="1"
                value={xOffset}
                onChange={(e) => setXOffset(Number(e.target.value))}
              />
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontWeight: 600 }}>Y offset</span>
                <span style={{ fontVariantNumeric: "tabular-nums" }}>{yOffset}px</span>
              </div>
              <input
                type="range"
                min="-200"
                max="200"
                step="1"
                value={yOffset}
                onChange={(e) => setYOffset(Number(e.target.value))}
              />
            </label>

            <div style={{ fontSize: 12, color: "#666", lineHeight: 1.4 }}>
              Under the hood: TFJS FaceMesh gives you a face <code>box</code> + <code>keypoints</code>.  [oai_citation:6‡GitHub](https://raw.githubusercontent.com/tensorflow/tfjs-models/master/face-landmarks-detection/README.md)  
              This demo uses the bounding box (simplest). You can upgrade to keypoint-anchoring later.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
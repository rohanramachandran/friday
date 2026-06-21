"""Screen understanding via Apple's built-in OCR and Accessibility, no VLM needed."""
import asyncio, base64, subprocess, tempfile, json
from pathlib import Path

# Lazy-loaded VLM for true visual queries (charts, photos)
_vlm_model = None
_vlm_processor = None
_vlm_config = None
VL_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"  # smaller fallback

async def screenshot_tool(image_b64: str = None, query: str = "", mode: str = "auto", analyze: bool = True) -> str:
    """
    Modes:
      auto   - OCR + window context (default, ~0 memory cost)
      visual - load VLM for true image understanding (charts, photos)
      capture - just save screenshot, no analysis
    """
    loop = asyncio.get_event_loop()

    # Always capture a fresh screenshot unless one was provided
    if not image_b64:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        subprocess.run(["screencapture", "-x", path], check=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_b64))
            path = f.name

    try:
        if mode == "capture" or not analyze:
            return f"Screenshot captured at {path}"

        # Decide mode automatically based on the query
        if mode == "auto":
            visual_kw = ["chart", "graph", "photo", "picture", "image", "diagram",
                         "drawing", "color", "shape", "icon looks", "visual"]
            mode = "visual" if any(k in query.lower() for k in visual_kw) else "text"

        if mode == "text" or mode == "auto":
            return await loop.run_in_executor(None, _ocr_pipeline, path, query)
        else:
            return await loop.run_in_executor(None, _vlm_pipeline, path, query)
    finally:
        Path(path).unlink(missing_ok=True)


def _ocr_pipeline(img_path: str, query: str) -> str:
    """Free, fast, zero-memory: macOS Vision OCR + active window context."""
    # 1. Get active app and window title
    app_script = '''
    tell application "System Events"
        set frontApp to first application process whose frontmost is true
        set appName to name of frontApp
        try
            set winName to name of window 1 of frontApp
        on error
            set winName to "(no window)"
        end try
        return appName & "|||" & winName
    end tell
    '''
    try:
        r = subprocess.run(["osascript", "-e", app_script],
                          capture_output=True, text=True, timeout=3)
        parts = r.stdout.strip().split("|||")
        app, window = (parts[0], parts[1]) if len(parts) == 2 else ("Unknown", "")
    except Exception:
        app, window = "Unknown", ""

    # 2. Run pre-compiled Apple Vision OCR binary (fast, ~100ms)
    ocr_bin = Path(__file__).parent.parent / "bin" / "ocr"
    try:
        r = subprocess.run([str(ocr_bin), img_path],
                          capture_output=True, text=True, timeout=10)
        ocr_text = r.stdout.strip()
    except Exception as e:
        ocr_text = f"(OCR failed: {e})"

    # 3. Compose structured context
    result = f"Active app: {app}\nWindow: {window}\n\n--- Visible text on screen ---\n{ocr_text}"

    if query:
        result += f"\n\n--- User question ---\n{query}"
    return result


def _vlm_pipeline(img_path: str, query: str) -> str:
    """Heavy path: load small VLM only when text isn't enough."""
    global _vlm_model, _vlm_processor, _vlm_config
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template
    import mlx.core as mx
    import gc

    if _vlm_model is None:
        _vlm_model, _vlm_processor = load(VL_MODEL)
        _vlm_config = load_config(VL_MODEL)

    prompt = query or "Describe what you see in detail."
    formatted = apply_chat_template(_vlm_processor, _vlm_config, prompt, num_images=1)
    out = generate(_vlm_model, _vlm_processor, formatted, [img_path],
                   max_tokens=300, verbose=False)
    text = out.text if hasattr(out, "text") else str(out)

    # free VLM memory immediately after use
    _vlm_model = None
    _vlm_processor = None
    _vlm_config = None
    gc.collect()
    mx.clear_cache()
    return text

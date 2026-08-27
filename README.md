# On-Device Product Detection & Recognition (YOLO + PaddleOCR + MiniLM)

Android application for **on-device product understanding** from images using a three-stage pipeline:

1. **YOLO (NCNN, C++/JNI)** for object detection  
2. **PaddleOCR (ONNX Runtime + OpenCV, Kotlin)** for text extraction  
3. **Hybrid matcher (MiniLM embeddings + fuzzy matching, Kotlin)** for mapping OCR text to product names

The app runs locally on device (no server inference in this repository).

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Setup](#setup)
- [Build & Run](#build--run)
- [Benchmarking](#benchmarking)
- [Configuration](#configuration)
- [Assets & Models](#assets--models)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Security & Privacy](#security--privacy)
- [License](#license)

---

## Overview

This project detects product regions in an image, extracts text once from the full image, maps OCR lines to detected objects, and resolves each object to the most likely catalog product.

**Main characteristics:**

- Native YOLO inference through NCNN (`app/src/main/cpp/native-lib.cpp`)
- Kotlin OCR SDK module (`ppocr-sdk`) built on ONNX Runtime + OpenCV
- Hybrid product resolver combining:
  - semantic similarity from MiniLM embeddings
  - fuzzy lexical similarity from normalized string/token matching
- Supports **`arm64-v8a`** and **`x86_64`** ABIs

---

## Repository Structure

```text
.
├─ app/                         # Android app module (UI + native YOLO + matcher orchestration)
│  ├─ src/main/java/com/example/yolo_paddle_poc/
│  │  ├─ MainActivity.kt        # End-to-end pipeline orchestration
│  │  ├─ ProductMatcher.kt      # Fuzzy product matcher
│  │  └─ TextEmbeddingMatcher.kt# MiniLM tokenizer + ONNX embedding matcher
│  ├─ src/main/cpp/
│  │  ├─ CMakeLists.txt
│  │  ├─ native-lib.cpp         # YOLO NCNN JNI inference
│  │  └─ ncnn/                  # Prebuilt NCNN headers/libs per ABI
│  ├─ src/main/assets/
│  │  ├─ yolo/model.ncnn.{param,bin}
│  │  ├─ products.txt
│  │  └─ text_matcher/          # MiniLM ONNX + vocab + product embeddings
│  └─ src/main/res/layout/activity_main.xml
│
├─ ppocr-sdk/                   # OCR library module
│  ├─ src/main/java/com/paddle/ocr/
│  │  ├─ PaddleOCR.kt
│  │  ├─ engine/                # OCREngine, DetectionEngine, RecognitionEngine, ORTSessionManager
│  │  ├─ preprocess/            # DetPreprocessor, RecPreprocessor
│  │  ├─ postprocess/           # DBPostProcessor, CTCDecoder, BoxSorter, QuadTextCrop...
│  │  ├─ model/                 # OCR models, config parsing, errors
│  │  └─ util/                  # OpenCV + image/bitmap helpers
│  ├─ src/main/assets/models/
│  │  ├─ det/inference.onnx
│  │  └─ rec/inference.onnx + inference.yml
│  └─ src/androidTest/          # OCR benchmark tests + fixture image
│
├─ gradle/
├─ build.gradle.kts
└─ settings.gradle.kts
```

---

## How the Pipeline Works

### 1) Model initialization

At startup (`MainActivity`):

- Loads native library `yolo_native`
- Initializes OpenCV
- Creates `PaddleOCR` with custom model asset paths
- Loads `TextEmbeddingMatcher` (MiniLM ONNX + vocab + precomputed product embeddings)
- Loads YOLO NCNN model from assets (`yolo/model.ncnn.param` + `.bin`)

The **Select Image** button is enabled only after all three are ready.

### 2) Image processing flow

When a user picks an image:

1. Decode and convert to `ARGB_8888`
2. Resize to max side `1280` for inference stability/perf
3. Run **YOLO** once on full image
4. Run **PaddleOCR** once on full image
5. Convert OCR quads to axis-aligned `RectF`
6. Map OCR boxes to YOLO boxes using:
   - center-inside rule, or
   - overlap threshold (`OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50`)
7. Build one recognized text string per object (reading order sort)
8. Resolve product via hybrid strategy:
   - embedding top-K from MiniLM
   - fuzzy top-K from lexical matcher
   - weighted score fusion + agreement bonus + acceptance thresholds
9. Draw YOLO boxes on output image and render object-wise results

### 3) Hybrid product resolution

`resolveProductHybrid()` combines:

- Embedding score weight: **0.55**
- Fuzzy score weight: **0.45**
- Agreement bonus: **+0.10** when both systems propose same canonical product

Acceptance rules:

- agreement + score ≥ `0.50`, or
- strong score ≥ `0.75`, or
- strong fuzzy override ≥ `0.88`

Otherwise object is labeled **Unknown**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Android app | Kotlin, AppCompat, ConstraintLayout |
| Native detection | C++, JNI, NCNN |
| OCR runtime | ONNX Runtime (Android), OpenCV |
| Text semantic matching | MiniLM ONNX (`all-MiniLM-L6-v2` style embeddings) |
| Build system | Gradle Kotlin DSL, Android Gradle Plugin |

---

## Requirements

- **Android Studio** (recent stable with AGP 9.x support)
- **JDK 17+**
- Android SDK with:
  - compile/target SDK used by module build files
  - NDK + CMake (for native YOLO build)
- A connected Android device or emulator (x86_64 supported)

Project defaults include:

- App module: `minSdk 26`, `targetSdk 37`
- OCR module: `minSdk 26`, `compileSdk 35`
- Gradle wrapper: `9.5.0`

---

## Setup

1. Clone repository.
2. Open in Android Studio.
3. Let Gradle sync complete.
4. Ensure `local.properties` points to valid Android SDK.
5. Confirm NDK/CMake are installed from SDK Manager.

No additional remote model download is required because required model assets are committed in-module.

---

## Build & Run

### Android Studio

- Select the `app` run configuration.
- Run on emulator/device.
- After models load, tap **Select Image** and choose an image from gallery/files provider.

### CLI (Windows PowerShell)

```powershell
.\gradlew.bat assembleDebug
.\gradlew.bat :app:installDebug
```

---

## Benchmarking

`ppocr-sdk` contains instrumentation benchmarks:

- Accuracy export (`accuracy_export.json`)
- Latency benchmark (`latency_benchmark.json`)
- Memory benchmark (`memory_benchmark.json`)

Output directory on device:

```text
<app_external_files>/ocr_benchmark/
```

Run:

```powershell
.\gradlew.bat :ppocr-sdk:connectedDebugAndroidTest
```

Optional instrumentation args (passed as Gradle properties):

- `warmup` (default `3`)
- `iterations` (default `10`)
- `rec_batch_size` (default `1`)

Example:

```powershell
.\gradlew.bat :ppocr-sdk:connectedDebugAndroidTest `
  -Pandroid.testInstrumentationRunnerArguments.warmup=5 `
  -Pandroid.testInstrumentationRunnerArguments.iterations=20 `
  -Pandroid.testInstrumentationRunnerArguments.rec_batch_size=1
```

---

## Configuration

### App-level constants (`MainActivity.kt`)

- `MAX_IMAGE_SIZE = 1280`
- `OCR_CONFIDENCE_THRESHOLD = 0.25`
- `OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50`
- Hybrid matching thresholds/weights (see [How the Pipeline Works](#how-the-pipeline-works))

### PaddleOCR config

Constructed in `loadPaddleOCR()` with:

- `detThresh = 0.3`
- `detBoxThresh = 0.6`
- `recScoreThresh = OCR_CONFIDENCE_THRESHOLD`
- `recBatchSize = 1`
- `EngineConfig(numThreads = 2)`

### Native YOLO config (`native-lib.cpp`)

- Input size: `640x640` letterboxed
- `NUM_CLASSES = 17`
- confidence threshold: `0.25`
- NMS threshold: `0.45`
- `num_threads = 2`
- Vulkan compute disabled in current setup

---

## Assets & Models

### `app/src/main/assets`

- `yolo/model.ncnn.param`
- `yolo/model.ncnn.bin`
- `products.txt` (**353** non-empty aliases)
- `text_matcher/`
  - `model_qint8_arm64.onnx`
  - `vocab.txt`, tokenizer/config JSON files
  - `product_names.json` (**331** canonical product names)
  - `product_embeddings.bin`
  - `product_embeddings_metadata.json` (`count=331`, `dimension=384`, normalized float32)

### `ppocr-sdk/src/main/assets/models`

- `det/inference.onnx`
- `det/inference.yml`
- `rec/inference.onnx`
- `rec/inference.yml` (character dictionary parsed by `ModelConfig`)

---

## Output Format

Per detected object, the app displays:

- Object index
- Matched product (or `Unknown`)
- Match score (%)
- Match method (`HYBRID_AGREEMENT`, `HYBRID_STRONG`, `FUZZY_STRONG`, `UNKNOWN`, etc.)
- Mapped OCR text and OCR line count
- YOLO confidence
- Shared full-image OCR time

Bounding boxes are drawn on the selected image in the app UI.

---

## Troubleshooting

- **Models never become ready**  
  Check Logcat tags: `YOLO_NCNN`, `PADDLE_OCR`, `TEXT_EMBEDDING_MATCHER`.

- **OpenCV init failure (`opencv_java4`)**  
  Verify OpenCV dependency packaging in `ppocr-sdk` and ABI compatibility.

- **Native build/link errors (NCNN)**  
  Ensure ABI-specific NCNN files exist under `app/src/main/cpp/ncnn/<ABI>/...`.

- **No detections or weak matches**  
  Tune thresholds in `MainActivity.kt`, detector thresholds in `native-lib.cpp`, or product catalog assets.

---

## Known Limitations

- YOLO output currently uses **best class confidence only**; class label mapping is not surfaced in UI.
- Matching quality depends on OCR quality and product catalog coverage.
- Large model assets increase APK size.
- Current demo UI is functional and diagnostic-focused, not production-polished.

---

## Security & Privacy

- Inference runs on-device with local assets in this repository.
- No network calls are implemented in the main detection/OCR/matching flow.

---

## License

This repository currently does **not** include a top-level `LICENSE` file.

Some source files (notably in `ppocr-sdk`) include Apache 2.0 headers from PaddleOCR-related components. Add an explicit repository license and third-party attribution policy before external distribution.


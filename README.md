# On_Edge_YOLO_PaddleOCR

---

![Platform](https://img.shields.io/badge/Platform-Android-3DDC84?style=for-the-badge)
![UI](https://img.shields.io/badge/UI-Android%20Views-1E88E5?style=for-the-badge)
![Language](https://img.shields.io/badge/Language-Kotlin-7F52FF?style=for-the-badge)
![Native](https://img.shields.io/badge/Native-C%2B%2B%20(JNI)-00599C?style=for-the-badge)
![Detector](https://img.shields.io/badge/Detector-YOLO%20(NCNN)-FF8F00?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-PaddleOCR%20(ONNX)-00A86B?style=for-the-badge)
![Matcher](https://img.shields.io/badge/Matcher-MiniLM%20%2B%20Fuzzy-8E24AA?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)

<p align="center">
  <b>On-device object detection, OCR, and product-name resolution pipeline for Android</b>
</p>

<p align="center">
  <i>Detect objects with YOLO (NCNN), extract text with PaddleOCR (ONNX Runtime + OpenCV), and map OCR strings to product catalog entities via hybrid semantic + fuzzy matching.</i>
</p>

---

## 1) Project Overview

This repository contains a multi-module Android project that performs full on-device visual understanding for product images:

1. **Object Detection** using a native YOLO model executed via **NCNN** (`C++`, `JNI`).
2. **Text Detection + Recognition** using a local **PaddleOCR** runtime implemented in Kotlin on top of **ONNX Runtime** and **OpenCV**.
3. **Product Resolution** using a hybrid strategy:
   - semantic embedding retrieval (`MiniLM` ONNX + precomputed product embeddings),
   - fuzzy lexical matching (`Levenshtein`, token coverage, containment/partial logic),
   - score fusion and acceptance thresholds.

All critical inference runs are local to device in this codebase.

---

## 2) Repository Structure

```text
.
├─ app/
│  ├─ build.gradle.kts
│  └─ src/
│     ├─ main/
│     │  ├─ AndroidManifest.xml
│     │  ├─ assets/
│     │  │  ├─ yolo/
│     │  │  │  ├─ model.ncnn.param
│     │  │  │  └─ model.ncnn.bin
│     │  │  ├─ text_matcher/
│     │  │  │  ├─ model_qint8_arm64.onnx
│     │  │  │  ├─ vocab.txt
│     │  │  │  ├─ tokenizer.json
│     │  │  │  ├─ tokenizer_config.json
│     │  │  │  ├─ special_tokens_map.json
│     │  │  │  ├─ config.json
│     │  │  │  ├─ product_names.json
│     │  │  │  ├─ product_embeddings.bin
│     │  │  │  └─ product_embeddings_metadata.json
│     │  │  └─ products.txt
│     │  ├─ cpp/
│     │  │  ├─ CMakeLists.txt
│     │  │  ├─ native-lib.cpp
│     │  │  └─ ncnn/ (prebuilt headers/libs for arm64-v8a and x86_64)
│     │  ├─ java/com/example/yolo_paddle_poc/
│     │  │  ├─ MainActivity.kt
│     │  │  ├─ ProductMatcher.kt
│     │  │  └─ TextEmbeddingMatcher.kt
│     │  ├─ keepRules/rules.keep
│     │  └─ res/
│     │     ├─ layout/activity_main.xml
│     │     ├─ values/*.xml
│     │     └─ xml/*.xml
│     ├─ androidTest/java/.../ExampleInstrumentedTest.kt
│     └─ test/java/.../ExampleUnitTest.kt
│
├─ ppocr-sdk/
│  ├─ build.gradle.kts
│  └─ src/
│     ├─ main/
│     │  ├─ AndroidManifest.xml
│     │  ├─ assets/models/
│     │  │  ├─ det/inference.onnx
│     │  │  ├─ det/inference.yml
│     │  │  ├─ rec/inference.onnx
│     │  │  └─ rec/inference.yml
│     │  └─ java/com/paddle/ocr/
│     │     ├─ PaddleOCR.kt
│     │     ├─ PaddleOCRConfig.kt
│     │     ├─ EngineConfig.kt
│     │     ├─ engine/
│     │     │  ├─ OCREngine.kt
│     │     │  ├─ DetectionEngine.kt
│     │     │  ├─ RecognitionEngine.kt
│     │     │  ├─ ORTSessionManager.kt
│     │     │  └─ OCREngineResult.kt
│     │     ├─ preprocess/
│     │     │  ├─ DetPreprocessor.kt
│     │     │  └─ RecPreprocessor.kt
│     │     ├─ postprocess/
│     │     │  ├─ DBPostProcessor.kt
│     │     │  ├─ CTCDecoder.kt
│     │     │  ├─ BoxSorter.kt
│     │     │  ├─ PolygonUnclip.kt
│     │     │  ├─ QuadGeometry.kt
│     │     │  └─ QuadTextCrop.kt
│     │     ├─ model/
│     │     │  ├─ OCRBox.kt
│     │     │  ├─ OCRResult.kt
│     │     │  ├─ OCRRunResult.kt
│     │     │  ├─ OCRError.kt
│     │     │  └─ ModelConfig.kt
│     │     └─ util/
│     │        ├─ OpenCVUtils.kt
│     │        ├─ BitmapUtils.kt
│     │        ├─ ImageUtils.kt
│     │        ├─ MathUtils.kt
│     │        └─ YamlUtils.kt
│     └─ androidTest/
│        ├─ java/com/paddle/ocr/benchmark/
│        │  ├─ OCRBenchmarkTest.kt
│        │  └─ BenchmarkFixtures.kt
│        └─ res/raw/android_ocr_benchmark_reference.png
│
├─ build.gradle.kts
├─ settings.gradle.kts
├─ gradle/
│  ├─ libs.versions.toml
│  └─ wrapper/gradle-wrapper.properties
├─ gradle.properties
└─ .gitignore
```

---

## 3) Build System and Versions

### Project Modules

- `:app` (Android application)
- `:ppocr-sdk` (Android library module)

### Root Gradle

- Android Gradle Plugin version: **9.3.2**
- Gradle wrapper: **9.5.0**
- Repository resolution mode: `FAIL_ON_PROJECT_REPOS`

### SDK / Java Configuration

| Module | compileSdk | minSdk | targetSdk | Java |
|---|---:|---:|---:|---|
| `app` | 37 | 26 | 37 | 11 |
| `ppocr-sdk` | 35 | 26 | n/a (library) | 17 |

### ABI Filters (`app`)

- `arm64-v8a`
- `x86_64`

### App dependencies (not exhaustive of transitive)

- `androidx.activity:activity-ktx`
- `androidx.appcompat:appcompat`
- `androidx.constraintlayout:constraintlayout`
- `androidx.core:core-ktx`
- `com.google.android.material:material`
- `com.microsoft.onnxruntime:onnxruntime-android:1.21.1`
- module dependency: `implementation(project(":ppocr-sdk"))`

### OCR module dependencies

- `com.microsoft.onnxruntime:onnxruntime-android:1.21.1`
- `com.quickbirdstudios:opencv:4.5.3.0`
- `org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0`
- `androidx.core:core-ktx:1.15.0`

---

## 4) End-to-End Runtime Flow

`MainActivity.kt` drives the full user-facing pipeline.

### Model boot sequence

At startup, app initializes:

1. **TextEmbeddingMatcher**
   - loads vocabulary, tokenizer metadata, product names, embeddings, ONNX session.
2. **YOLO model**
   - native JNI call `loadYoloModel(assets)`.
3. **PaddleOCR**
   - initializes OpenCV,
   - creates OCR engine with detection + recognition model assets.

Select button is enabled only when all three states are ready:

- `yoloReady`
- `ocrReady`
- `embeddingReady`

### Inference sequence per selected image

1. Decode input to `Bitmap`.
2. Convert to `ARGB_8888`.
3. Resize (max side `1280`) for inference.
4. Run YOLO once on full image (`detectObjects` JNI).
5. Run PaddleOCR once on full image (`ocr.recognize(bitmap)`).
6. Convert OCR quads to axis-aligned rectangles.
7. Assign OCR boxes to object boxes (center-inside or overlap threshold).
8. Merge OCR lines per object in reading order.
9. Resolve product by hybrid matcher.
10. Draw detection boxes and display structured text output.

---

## 5) Native YOLO (NCNN) Implementation Details

File: `app/src/main/cpp/native-lib.cpp`

### Exposed JNI methods

- `loadYoloModel(AssetManager): Boolean`
- `detectObjects(Bitmap): FloatArray`

### Core settings

- `INPUT_SIZE = 640`
- `NUM_CLASSES = 17`
- `CONF_THRESHOLD = 0.25f`
- `NMS_THRESHOLD = 0.45f`
- `g_yolo.opt.use_vulkan_compute = false`
- `g_yolo.opt.num_threads = 2`

### Processing steps

1. Validate model loaded.
2. Validate bitmap format `RGBA_8888`.
3. Copy bitmap with stride-safe memory handling.
4. Resize to letterboxed input preserving aspect ratio.
5. Manual normalization (`0..255 -> 0..1`).
6. Execute NCNN extractor:
   - input blob `"in0"`
   - output blob `"out0"`
7. Decode YOLO output:
   - supports orientation of `[features, predictions]` or `[predictions, features]`
   - feature count = `4 + NUM_CLASSES`.
8. Select best class score per prediction.
9. Convert `xywh` to `xyxy`, undo letterbox, map to original image coordinates.
10. Apply class-agnostic NMS.
11. Return flattened `FloatArray`:
    - `[x1, y1, x2, y2, confidence]` per detection.

### Build integration

`CMakeLists.txt` resolves per-ABI NCNN package from:

`app/src/main/cpp/ncnn/${ANDROID_ABI}/lib/cmake/ncnn`

Linked libs:

- `ncnn`
- `log`
- `android`
- `jnigraphics`

---

## 6) OCR SDK (`ppocr-sdk`) Architecture

### Public API

`PaddleOCR.kt`

- `create(context)`
- `create(context, config, engineConfig)`
- `create(context, config, engineConfig, detModelAssetPath, recModelAssetPath, recConfigAssetPath)`
- `recognize(bitmap)`
- `recognize(imageBytes)`
- `release()`

Returns `OCRRunResult` with both prediction data and detailed timing metrics.

### Engine graph

`OCREngine` composes:

1. `ORTSessionManager` (loads ONNX models, owns ORT sessions)
2. `DetectionEngine`
3. `RecognitionEngine`

### Detection pipeline

`DetPreprocessor`:

- optional BGR->RGB conversion depending on config,
- resize to multiple of 32 with side-length constraints,
- normalize by ImageNet-like mean/std,
- output NCHW tensor.

`DetectionEngine`:

- ORT detection inference,
- DB postprocess via `DBPostProcessor`,
- outputs quadrilateral `OCRBox` list.

### Recognition pipeline

`QuadTextCrop`:

- perspective crop from each detection quad,
- auto-rotate highly vertical crops.

`RecPreprocessor`:

- resize each crop to fixed height (`48`) while preserving width ratio,
- normalize with `x/127.5 - 1`,
- pad to batch max width,
- assemble NCHW batch.

`RecognitionEngine`:

- ORT recognition inference,
- CTC decode (`CTCDecoder`),
- per-line confidence.

### Ordering and filtering

- Boxes sorted by reading order (`BoxSorter`).
- Final OCR lines filtered by `recScoreThresh`.

### Error model

Typed OCR exceptions in `OCRError`:

- `ModelNotFound`
- `ModelLoadFailed`
- `ConfigParseFailed`
- `InvalidImage`
- `InferenceFailed`
- `DecodeError`

---

## 7) Product Matching System

Two independent matchers run and are fused.

### A) Semantic matcher (`TextEmbeddingMatcher.kt`)

Uses:

- ONNX MiniLM model (`model_qint8_arm64.onnx`)
- custom tokenizer implementation:
  - basic tokenization (lowercase/accent strip/punctuation split),
  - WordPiece tokenization against vocab.

Pipeline:

1. Tokenize query text.
2. Run ONNX transformer session.
3. Mean-pool token embeddings using attention mask.
4. L2-normalize vector.
5. Dot-product against precomputed product embeddings.
6. Return top-k by score.

### B) Fuzzy matcher (`ProductMatcher.kt`)

Reads aliases from `products.txt`, then computes score from:

- exact/compact exact matches,
- containment signals,
- Levenshtein-based similarity,
- token coverage weighted by token length,
- partial sliding-window similarity,
- multi-token agreement bonus,
- penalty for generic one-word candidates when OCR has multiple words.

Final fuzzy score range: `0..100`.

### C) Hybrid fusion (`MainActivity.resolveProductHybrid`)

Constants:

- `TOP_K = 5`
- `EMBEDDING_WEIGHT = 0.55`
- `FUZZY_WEIGHT = 0.45`
- `AGREEMENT_BONUS = 0.10`
- `FINAL_STRONG_THRESHOLD = 0.75`
- `FINAL_AGREEMENT_THRESHOLD = 0.50`
- `FUZZY_STRONG_OVERRIDE = 0.88`

Decision methods used in result output:

- `HYBRID_AGREEMENT`
- `HYBRID_STRONG`
- `FUZZY_STRONG`
- `UNKNOWN`
- `NO_MAPPED_TEXT` (when object has no mapped OCR lines)

---

## 8) OCR-to-Object Mapping Logic

For each OCR text box:

1. Compute OCR rectangle center.
2. For each detected object box:
   - candidate if center lies inside object box, **or**
   - intersection-over-OCR-area >= `0.50`.
3. Association score:
   - center-inside: `1 + overlap`
   - overlap-only: `overlap`
4. Attach OCR line to best scoring object.

Then per object:

- OCR lines sorted roughly top-to-bottom then left-to-right.
- Merged into single string for matching.

---

## 9) Assets, Data, and Model Inventory

### App asset inventory

- `products.txt`: **353** non-empty aliases.
- `text_matcher/product_names.json`: **331** product names.
- `text_matcher/product_embeddings_metadata.json`:
  - `count = 331`
  - `dimension = 384`
  - `dtype = float32`
  - `normalized = true`
  - `model = sentence-transformers/all-MiniLM-L6-v2`

### OCR model assets

- Detection:
  - `models/det/inference.onnx`
  - `models/det/inference.yml`
- Recognition:
  - `models/rec/inference.onnx`
  - `models/rec/inference.yml`

`ModelConfig` parses recognition `character_dict` from YAML and ensures trailing space token exists.

### Native detector assets

- `assets/yolo/model.ncnn.param`
- `assets/yolo/model.ncnn.bin`

---

## 10) UI and User Experience

Layout: `app/src/main/res/layout/activity_main.xml`

Main components:

- `statusText`: model/loading/progress state
- `selectImageButton`: image picker trigger (disabled until ready)
- `imageView`: renders detected image with bounding boxes
- `resultsText` in `ScrollView`: per-object output details

Displayed per object:

- object index
- matched product or unknown
- match score (if available)
- match method
- mapped OCR text + line count
- YOLO confidence
- full-image OCR elapsed time

---

## 11) Testing and Benchmarking

### App module tests

- `ExampleUnitTest.kt` (placeholder local unit test)
- `ExampleInstrumentedTest.kt` (package-name check)

### OCR benchmark suite (`ppocr-sdk` androidTest)

`OCRBenchmarkTest.kt` includes:

1. `testOCRExportJSON` (accuracy-style export payload)
2. `testLatencyBenchmark`
3. `testMemoryBenchmark`

Benchmark output path:

- `<externalFilesDir>/ocr_benchmark/`

JSON outputs:

- `accuracy_export.json`
- `latency_benchmark.json`
- `memory_benchmark.json`

Supported instrumentation args:

- `warmup` (default `3`)
- `iterations` (default `10`)
- `rec_batch_size` (default `1`)

Run:

```powershell
.\gradlew.bat :ppocr-sdk:connectedDebugAndroidTest
```

With args:

```powershell
.\gradlew.bat :ppocr-sdk:connectedDebugAndroidTest `
  -Pandroid.testInstrumentationRunnerArguments.warmup=5 `
  -Pandroid.testInstrumentationRunnerArguments.iterations=20 `
  -Pandroid.testInstrumentationRunnerArguments.rec_batch_size=1
```

---

## 12) Build and Run Instructions

### Android Studio

1. Open project root.
2. Sync Gradle.
3. Ensure SDK/NDK/CMake components are installed.
4. Run `app` on supported emulator/device.
5. Wait for model readiness message.
6. Select image and inspect results.

### CLI

```powershell
.\gradlew.bat assembleDebug
.\gradlew.bat :app:installDebug
```

---

## 13) Configuration Reference

### `app/build.gradle.kts`

- noCompress includes:
  - `bin`
  - `param`
  - `onnx`
- external native build through `src/main/cpp/CMakeLists.txt`

### `MainActivity` critical constants

- `MAX_IMAGE_SIZE = 1280`
- `OCR_CONFIDENCE_THRESHOLD = 0.25f`
- `OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50f`
- hybrid thresholds/weights as listed above

### `PaddleOCRConfig` defaults (SDK)

- `detImgMode = "BGR"`
- `detLimitSideLen = 64`
- `detLimitType = "min"`
- `detMaxSideLimit = 4000`
- `detThresh = 0.3f`
- `detBoxThresh = 0.6f`
- `detUnclipRatio = 1.5f`
- `detMaxCandidates = 3000`
- `detUseDilation = false`
- `detScoreMode = "fast"`
- `detBoxType = "quad"`
- `recScoreThresh = 0.0f`
- `recBatchSize = 1`

---

## 14) Logging Tags

Useful logcat tags used in app:

- `YOLO_NCNN`
- `PADDLE_OCR`
- `PRODUCT_MATCHER`
- `TEXT_EMBEDDING_MATCHER`
- `HYBRID_PRODUCT_RESOLVER`
- `OCR_OBJECT_MAPPING`

Useful logcat tag in OCR util:

- `OpenCVUtils`

---

## 15) Error/Troubleshooting Guide

### Models not ready / button disabled

Check whether one of these failed:

- YOLO model load (`loadYoloModel`)
- OpenCV init (`System.loadLibrary("opencv_java4")`)
- PaddleOCR engine creation
- MiniLM ONNX session creation

### Native inference returns no detections

- validate input bitmap format is RGBA_8888,
- verify NCNN assets exist and are loadable,
- inspect output shape compatibility (`w/h` against features count).

### OCR model load errors

- verify model files under `ppocr-sdk/src/main/assets/models`,
- check `inference.yml` character dict parse validity.

### Matcher anomalies

- verify consistency of:
  - `product_names.json`
  - `product_embeddings.bin`
  - `product_embeddings_metadata.json` (`count * dimension` float count check).

---

## 16) Security and Privacy

- Inference and matching are implemented locally on device in this repo.
- No network service is required for core pipeline execution.
- Product/model assets are bundled with the app modules.

---

## 17) Known Limitations

- YOLO class label IDs are not surfaced in UI output; confidence only is used.
- Matching accuracy depends heavily on OCR quality and product alias coverage.
- Current app UI is functional and debug-oriented, not a finalized production UX.
- Model assets significantly increase APK/app size.

---

## 18) License Status

A top-level repository `LICENSE` file is currently not present.

Several files in `ppocr-sdk` carry Apache 2.0 headers (Paddle-derived components). Add an explicit repository license and third-party notices before distribution.

---

## 19) Quick Start Checklist

1. Sync Gradle successfully.
2. Ensure SDK + NDK + CMake availability.
3. Confirm app launches and all three model states become ready.
4. Select representative image.
5. Validate:
   - boxes drawn,
   - OCR text mapped,
   - product match method and score emitted.


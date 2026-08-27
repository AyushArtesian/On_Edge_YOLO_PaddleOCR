# On-Edge YOLO + PaddleOCR

---

![Platform](https://img.shields.io/badge/Platform-Android-3DDC84?style=for-the-badge)
![Frontend](https://img.shields.io/badge/UI-Android%20Views-4A90E2?style=for-the-badge)
![Language](https://img.shields.io/badge/Language-Kotlin-7F52FF?style=for-the-badge)
![Native](https://img.shields.io/badge/Native-C%2B%2B-00599C?style=for-the-badge)
![Detection](https://img.shields.io/badge/Detection-NCNN-orange?style=for-the-badge)
![OCR](https://img.shields.io/badge/OCR-PaddleOCR-00A86B?style=for-the-badge)
![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime-005CED?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=for-the-badge)

<p align="center">
  <b>On-device product detection, OCR, and hybrid product matching pipeline</b>
</p>

<p align="center">
  <i>Detect objects with YOLO, extract text with PaddleOCR, and map OCR text to product catalog entries using MiniLM embeddings + fuzzy matching in one Android app.</i>
</p>

---

## Project Summary

This project is an Android implementation of a fully on-device computer vision + OCR + text-matching system for retail/product understanding workflows.

It combines:

- **YOLO (NCNN, JNI/C++)** for object detection  
- **PaddleOCR (ONNX Runtime + OpenCV)** for text detection/recognition  
- **Hybrid matcher (MiniLM embeddings + fuzzy similarity)** for product name resolution

---

## Tech Stack

| Layer | Technology |
|---|---|
| App | Android (Kotlin, AppCompat, ConstraintLayout) |
| Native Inference | C++ + JNI + NCNN |
| OCR Runtime | ONNX Runtime Android + OpenCV |
| Semantic Matching | MiniLM (`all-MiniLM-L6-v2` style embeddings) |
| Build System | Gradle Kotlin DSL |

---

## Repository Structure

```text
.
├─ app/
│  ├─ src/main/java/com/example/yolo_paddle_poc/
│  │  ├─ MainActivity.kt
│  │  ├─ ProductMatcher.kt
│  │  └─ TextEmbeddingMatcher.kt
│  ├─ src/main/cpp/
│  │  ├─ CMakeLists.txt
│  │  ├─ native-lib.cpp
│  │  └─ ncnn/
│  └─ src/main/assets/
│     ├─ yolo/
│     ├─ text_matcher/
│     └─ products.txt
│
├─ ppocr-sdk/
│  ├─ src/main/java/com/paddle/ocr/
│  ├─ src/main/assets/models/
│  └─ src/androidTest/
│
├─ build.gradle.kts
├─ settings.gradle.kts
└─ gradle/
```

---

## Pipeline Architecture

1. Load YOLO NCNN model from app assets.  
2. Load PaddleOCR detection + recognition ONNX models.  
3. Load MiniLM ONNX model, tokenizer vocab, and precomputed product embeddings.  
4. User selects image.  
5. Run full-image YOLO detection.  
6. Run full-image OCR once.  
7. Map OCR boxes to object boxes by center/overlap rules.  
8. Resolve each object’s text to product name via hybrid scoring.  
9. Render bounding boxes + per-object product result.

---

## Requirements

- Android Studio (recent stable)
- JDK 17+
- Android SDK + NDK + CMake
- Device/emulator (`arm64-v8a` or `x86_64`)

Project-level defaults:

- `app`: minSdk 26, targetSdk 37
- `ppocr-sdk`: minSdk 26, compileSdk 35

---

## Setup

1. Clone the repository.
2. Open project in Android Studio.
3. Let Gradle sync complete.
4. Ensure `local.properties` points to a valid SDK path.
5. Install required SDK components (NDK/CMake).

---

## Build & Run

```powershell
.\gradlew.bat assembleDebug
.\gradlew.bat :app:installDebug
```

Launch the app, wait for all models to load, then select an image.

---

## Benchmarking (OCR Module)

`ppocr-sdk` includes instrumentation benchmarks for:

- Accuracy export
- Latency profiling
- Memory profiling

Run:

```powershell
.\gradlew.bat :ppocr-sdk:connectedDebugAndroidTest
```

Optional instrumentation args:

- `warmup`
- `iterations`
- `rec_batch_size`

---

## Key Configuration Points

- `MainActivity.kt`: image sizing, OCR confidence, OCR-object overlap threshold, hybrid scoring thresholds
- `native-lib.cpp`: YOLO input size, confidence/NMS thresholds, NCNN threads
- `PaddleOCRConfig`: detection/recognition thresholds and batching

---

## Assets Included

### App Assets

- `assets/yolo/model.ncnn.param`
- `assets/yolo/model.ncnn.bin`
- `assets/products.txt`
- `assets/text_matcher/` (MiniLM ONNX, tokenizer files, embeddings)

### OCR SDK Assets

- `models/det/inference.onnx`
- `models/det/inference.yml`
- `models/rec/inference.onnx`
- `models/rec/inference.yml`

---

## Output

For each detected object:

- Bounding box + YOLO confidence
- Mapped OCR text
- Product match result (`name`, `score`, `method`)
- OCR timing metadata

---

## Troubleshooting

- Check Logcat tags:
  - `YOLO_NCNN`
  - `PADDLE_OCR`
  - `TEXT_EMBEDDING_MATCHER`
  - `HYBRID_PRODUCT_RESOLVER`

- If OpenCV fails to initialize, verify ABI/dependency packaging.
- If native build fails, verify `app/src/main/cpp/ncnn/<ABI>` libraries are present.

---

## License

No top-level `LICENSE` file is currently present in this repository.

Some files in `ppocr-sdk` carry Apache 2.0 headers from Paddle-related code; add an explicit repository license before external distribution.

---

## README Template Notes (for reuse)

To reuse this style in other projects:

1. Keep a clean **title + badge strip + centered one-liner**.
2. Add a short **business/technical summary**.
3. Use clear sections: stack, architecture, setup, run, config, troubleshooting.
4. Keep formatting consistent with horizontal dividers and tables where useful.


# On-Device YOLO + PaddleOCR Product Intelligence (Android)

An Android multi-module project that performs **end-to-end, on-device product understanding**:

1. **Object detection** using YOLO in native C++ via NCNN.
2. **OCR detection + recognition** using a Kotlin PaddleOCR SDK backed by ONNX Runtime + OpenCV.
3. **Product resolution** using a hybrid semantic-embedding + fuzzy-text matcher.

---

## 1) What this repository contains

This repository is a full Android app + OCR SDK implementation with bundled model assets.

- App module: `:app`
- OCR SDK module: `:ppocr-sdk`
- Native detector: `app/src/main/cpp/native-lib.cpp`
- Embedded model/data assets in both modules
- Instrumented OCR benchmark suite

All core inference and matching in this codebase run locally on device.

---

## 2) Tech stack

- **Language**: Kotlin (app + SDK), C++ (native detector)
- **Android build**: AGP `9.3.2`, Gradle `9.5.0`
- **App SDK levels**: `compileSdk 37`, `targetSdk 37`, `minSdk 26`
- **OCR SDK**: `compileSdk 35`, `minSdk 26`
- **Native inference**: NCNN (prebuilt static libs)
- **OCR inference**: ONNX Runtime Android `1.21.1`
- **Image processing**: OpenCV Android `4.5.3.0`
- **Concurrency**: Kotlin coroutines

---

## 3) Repository structure

```text
.
├─ app/                          # Android application module
│  ├─ src/main/java/com/example/yolo_paddle_poc/
│  │  ├─ MainActivity.kt
│  │  ├─ ProductMatcher.kt
│  │  ├─ TextEmbeddingMatcher.kt
│  │  ├─ matching/HybridProductResolver.kt
│  │  ├─ model/Models.kt
│  │  ├─ ocr/OcrMapper.kt
│  │  ├─ ocr/ProminentTextSelector.kt
│  │  ├─ pipeline/ProductDetectionPipeline.kt
│  │  └─ util/{BitmapUtils,DrawingUtils}.kt
│  ├─ src/main/cpp/
│  │  ├─ CMakeLists.txt
│  │  ├─ native-lib.cpp
│  │  └─ ncnn/                   # prebuilt include/lib trees for arm64-v8a, x86_64
│  ├─ src/main/assets/
│  │  ├─ yolo/model.ncnn.{param,bin}
│  │  ├─ text_matcher/*          # MiniLM model, vocab, tokenizer files, embeddings
│  │  └─ products.txt            # legacy alias file (currently not used in matcher logic)
│  ├─ src/main/res/layout/activity_main.xml
│  ├─ src/main/AndroidManifest.xml
│  └─ src/{test,androidTest}/...
│
├─ ppocr-sdk/                    # OCR library module
│  ├─ src/main/java/com/paddle/ocr/
│  │  ├─ PaddleOCR.kt
│  │  ├─ PaddleOCRConfig.kt
│  │  ├─ EngineConfig.kt
│  │  ├─ engine/*
│  │  ├─ preprocess/*
│  │  ├─ postprocess/*
│  │  ├─ model/*
│  │  └─ util/*
│  ├─ src/main/assets/models/{det,rec}/*
│  └─ src/androidTest/java/com/paddle/ocr/benchmark/*
│
├─ gradle/libs.versions.toml
├─ settings.gradle.kts
├─ build.gradle.kts
├─ product_embeddings.bin                     # root-level legacy artifact
└─ product_embeddings_metadata.json           # root-level legacy metadata summary
```

---

## 4) App module (`:app`) in detail

### 4.1 MainActivity responsibilities

`MainActivity.kt` coordinates the full workflow:

- Loads native `yolo_native` library
- Declares JNI methods:
  - `loadYoloModel(assetManager)`
  - `detectObjects(bitmap)`
- Initializes:
  - `ProductMatcher`
  - `TextEmbeddingMatcher`
  - `PaddleOCR`
  - `HybridProductResolver`
  - `ProductDetectionPipeline`
- Handles two image inputs:
  - gallery (`GetContent`)
  - camera (`TakePicture` + `FileProvider` cache URI)
- Runs pipeline and renders:
  - boxed image overlay
  - grouped recognized/unknown product output
  - timing/status summary

### 4.2 Product pipeline orchestration

`pipeline/ProductDetectionPipeline.kt` performs two OCR passes per object candidate:

1. **Full-image OCR pass**
   - OCR once on full image
   - map OCR lines to object boxes
   - prominent-text selection
   - hybrid product resolution
2. **Crop fallback pass** (if full-image match fails)
   - crop object region with padding
   - upscale crop
   - OCR on crop
   - prominent-text selection
   - hybrid product resolution

Result is a list of `DetectedItem` objects with box coordinates, OCR text, match score, method, and timings.

### 4.3 Matching subsystem

- `TextEmbeddingMatcher.kt`
  - Loads MiniLM ONNX model + vocab + embedding metadata + binary vectors
  - Implements lightweight BERT-style tokenization (basic + WordPiece)
  - Generates normalized query embedding
  - Computes dot-product similarity against stored embeddings
  - Collapses multiple variants to best score per canonical product ID

- `ProductMatcher.kt`
  - Loads canonical products and aliases from `text_matcher/product_embeddings_metadata.json`
  - Excludes `search_text` variants from fuzzy alias scoring
  - Scores aliases with exact/compact exact, containment, token coverage, Levenshtein, partial similarity, bonuses/penalties
  - Returns best canonical fuzzy matches

- `matching/HybridProductResolver.kt`
  - Merges top embedding and fuzzy candidates by canonical product ID
  - Weighted fusion:
    - embedding: `0.55`
    - fuzzy: `0.45`
    - agreement bonus: `+0.10`
  - Decision thresholds:
    - strong: `0.75`
    - agreement: `0.50`
    - fuzzy override: `0.88`

### 4.4 OCR mapping and text selection

- `ocr/OcrMapper.kt`
  - Parses flattened YOLO output into `ObjectBox` list
  - Converts OCR quadrilateral to axis-aligned rect
  - Assigns OCR line to best object using center-inside and overlap ratio

- `ocr/ProminentTextSelector.kt`
  - Scores OCR lines using relative height/area, uppercase/title cues, confidence
  - Drops packaging noise (MRP, net weight, batch/expiry, etc.)
  - Selects up to 4 high-value lines in reading order

### 4.5 UI and utilities

- `res/layout/activity_main.xml`
  - status text
  - select/capture buttons
  - result image preview
  - scrollable detected-items output
- `util/BitmapUtils.kt`
  - EXIF-correct decode
  - resize for inference
  - crop with padding
  - upscale for OCR fallback
- `util/DrawingUtils.kt`
  - draws red detection boxes and numeric object labels
- `res/xml/file_paths.xml`
  - scoped cache path for camera file sharing via `FileProvider`

---

## 5) Native YOLO detector (`app/src/main/cpp/native-lib.cpp`)

### Exposed JNI API

- `Java_com_example_yolo_1paddle_1poc_MainActivity_loadYoloModel`
- `Java_com_example_yolo_1paddle_1poc_MainActivity_detectObjects`

### Runtime behavior

- Loads NCNN model assets from `assets/yolo/`
- Uses letterbox resize to `640x640`
- Converts RGBA bitmap safely with stride-aware copy
- Normalizes to `[0,1]`
- Runs extractor on blobs:
  - input: `in0`
  - output: `out0`
- Decodes output for `NUM_CLASSES = 17`
- Applies class-agnostic NMS (`NMS_THRESHOLD = 0.45`, `CONF_THRESHOLD = 0.25`)
- Returns flattened array as `[x1, y1, x2, y2, confidence] * N`

### Native build

`CMakeLists.txt` links:

- `ncnn`
- `log`
- `android`
- `jnigraphics`

and resolves package path by ABI:

- `app/src/main/cpp/ncnn/arm64-v8a/...`
- `app/src/main/cpp/ncnn/x86_64/...`

> The additional `arm64-v8a-openmp-backup` tree is present as a backup variant.

---

## 6) OCR SDK module (`:ppocr-sdk`) architecture

### Public API

`PaddleOCR.kt`

- `create(context)`
- `create(context, config, engineConfig)`
- `create(context, config, engineConfig, detModelAssetPath, recModelAssetPath, recConfigAssetPath)`
- `recognize(bitmap)`
- `recognize(imageBytes)`
- `release()`

### Config types

- `PaddleOCRConfig`
  - detection pre/post-process parameters and `recBatchSize`
- `EngineConfig`
  - ONNX Runtime thread count (`numThreads`)

### Engine pipeline

- `engine/ORTSessionManager.kt`
  - loads ONNX sessions
  - runs det/rec inference
  - tracks cold load time
- `engine/DetectionEngine.kt`
  - det preprocess -> ONNX det inference -> DB postprocess
- `engine/RecognitionEngine.kt`
  - rec preprocess batch -> ONNX rec inference -> CTC decode
- `engine/OCREngine.kt`
  - orchestrates full run and detailed timing aggregation

### Preprocess and postprocess packages

- `preprocess/DetPreprocessor.kt`
  - resize to multiple-of-32, normalize, NCHW tensor build
- `preprocess/RecPreprocessor.kt`
  - BGR->RGB, fixed height=48, width padding, batch tensor build
- `postprocess/DBPostProcessor.kt`
  - contour extraction, score filtering, polygon unclip, quad scaling
- `postprocess/QuadTextCrop.kt`
  - perspective crop and vertical auto-rotation
- `postprocess/CTCDecoder.kt`
  - CTC greedy decode with blank/repeat collapse
- `postprocess/BoxSorter.kt`
  - reading-order sorting with row-threshold refinement
- `postprocess/PolygonUnclip.kt`, `QuadGeometry.kt`
  - geometric helpers

### Models and error contracts

- `model/OCRBox.kt`, `OCRResult.kt`, `OCRRunResult.kt`
- `model/ModelConfig.kt`
  - parses recognition YAML `character_dict`
- `model/OCRError.kt`
  - typed failures: model not found/load failed, config parse, invalid image, inference failure, decode error

### Utility package

- `util/OpenCVUtils.kt`: OpenCV native init
- `util/BitmapUtils.kt`: Bitmap/Mat conversion + decode
- `util/ImageUtils.kt`: detector resize policy
- `util/MathUtils.kt`: half-even rounding
- `util/YamlUtils.kt`: YAML indentation helper

---

## 7) Assets and data inventory

### App-side matcher assets (`app/src/main/assets/text_matcher`)

- `model_qint8_arm64.onnx`
- `vocab.txt`
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `config.json`
- `product_embeddings.bin`
- `product_embeddings_metadata.json`
- `product_names.json` (legacy list file)

Current metadata (`product_embeddings_metadata.json`):

- `embedding_dimension`: `384`
- `normalized`: `true`
- embedding entries: `406`
- canonical products: `82`
- variant breakdown:
  - `name`: 82
  - `brand`: 82
  - `alias`: 160
  - `search_text`: 82

`products.txt` currently has `353` non-empty lines but is not used by the current `ProductMatcher` load path.

### OCR model assets (`ppocr-sdk/src/main/assets/models`)

- Detection: `det/inference.onnx`, `det/inference.yml`
- Recognition: `rec/inference.onnx`, `rec/inference.yml`

### YOLO assets (`app/src/main/assets/yolo`)

- `model.ncnn.param`
- `model.ncnn.bin`

---

## 8) Build and run

### Requirements

- Android Studio with Android SDK 37 support
- NDK + CMake installed (for app native build)
- Java toolchain support for Java 11 (app) and Java 17 (OCR module)

### Android Studio

1. Open project root.
2. Sync Gradle.
3. Build and run `:app`.
4. Wait for status: models ready.
5. Select image or capture photo.

### CLI (Linux/macOS)

```bash
./gradlew assembleDebug
./gradlew :app:installDebug
```

### CLI (Windows)

```powershell
.\gradlew.bat assembleDebug
.\gradlew.bat :app:installDebug
```

---

## 9) Testing and benchmarking

### App tests

- `app/src/test/.../ExampleUnitTest.kt` (placeholder)
- `app/src/androidTest/.../ExampleInstrumentedTest.kt` (package assertion)

### OCR benchmark tests

`ppocr-sdk/src/androidTest/java/com/paddle/ocr/benchmark/OCRBenchmarkTest.kt`

Includes:

- `testOCRExportJSON`
- `testLatencyBenchmark`
- `testMemoryBenchmark`

Run:

```bash
./gradlew :ppocr-sdk:connectedDebugAndroidTest
```

Optional instrumentation args:

- `warmup` (default 3)
- `iterations` (default 10)
- `rec_batch_size` (default 1)

Example:

```bash
./gradlew :ppocr-sdk:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.warmup=5 \
  -Pandroid.testInstrumentationRunnerArguments.iterations=20 \
  -Pandroid.testInstrumentationRunnerArguments.rec_batch_size=1
```

Benchmark JSON outputs are written to:

- `<externalFilesDir>/ocr_benchmark/`

---

## 10) Important runtime constants (current code)

- App image resize max side: `1280` (`MainActivity`)
- OCR mapping overlap threshold: `0.50` (`ProductDetectionPipeline`)
- OCR line confidence cutoff: `0.20` (`ProductDetectionPipeline`)
- Crop padding: `12%`, crop upscale: `3.0x` (`ProductDetectionPipeline`)
- YOLO input size: `640`, classes: `17` (`native-lib.cpp`)
- YOLO confidence/NMS: `0.25 / 0.45` (`native-lib.cpp`)
- OCR engine threads in app init: `2` (`MainActivity`)

---

## 11) Logging tags

- `YOLO_NCNN`
- `PADDLE_OCR`
- `TEXT_EMBEDDING_MATCHER`
- `PRODUCT_MATCHER`
- `HYBRID_PRODUCT_RESOLVER`
- `OCR_OBJECT_MAPPING`
- `OpenCVUtils`

---

## 12) Troubleshooting

- **Buttons remain disabled**
  - one or more of YOLO/OCR/embedding modules did not initialize
- **No detections**
  - verify YOLO assets present and model loaded
  - verify input bitmap reaches native as `ARGB_8888`
- **OCR init fails**
  - ensure OpenCV native load succeeds (`opencv_java4`)
  - verify ONNX/YAML files in `ppocr-sdk` assets
- **Matcher quality is poor**
  - check OCR text quality and product metadata/embedding consistency

---

## 13) Security and privacy notes

- Core pipeline is on-device in this codebase.
- No server dependency is required for model inference or matching.
- Camera images are stored in app cache and shared via non-exported `FileProvider`.

---

## 14) Known limitations

- UI is functional/demo-oriented and not production-polished.
- Detector class IDs are not surfaced in UI output (confidence + box only).
- Accuracy is sensitive to OCR quality and metadata coverage.
- Bundled model assets increase APK size.

---

## 15) Licensing and attribution

- `ppocr-sdk` source files include Apache-2.0 headers from PaddlePaddle-derived code.
- Top-level repository does not currently include a `LICENSE` file.
- Add explicit top-level license + third-party notices before distribution.


#include <jni.h>

#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "net.h"

#define LOG_TAG "YOLO_NCNN"

#define LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define LOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// =========================================================
// CONFIGURATION
// =========================================================

static const int INPUT_SIZE = 640;
static const int NUM_CLASSES = 17;

static const float CONF_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;

// =========================================================
// GLOBAL MODEL
// =========================================================

static ncnn::Net g_yolo;
static bool g_model_loaded = false;
static std::mutex g_model_mutex;

// =========================================================
// DETECTION STRUCT
// =========================================================

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
};

// =========================================================
// IOU
// =========================================================

static float calculateIoU(
        const Detection& a,
        const Detection& b) {

    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);

    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);

    const float width =
            std::max(0.0f, xx2 - xx1);

    const float height =
            std::max(0.0f, yy2 - yy1);

    const float intersection =
            width * height;

    const float areaA =
            std::max(0.0f, a.x2 - a.x1) *
            std::max(0.0f, a.y2 - a.y1);

    const float areaB =
            std::max(0.0f, b.x2 - b.x1) *
            std::max(0.0f, b.y2 - b.y1);

    const float unionArea =
            areaA + areaB - intersection;

    if (unionArea <= 0.0f) {
        return 0.0f;
    }

    return intersection / unionArea;
}

// =========================================================
// CLASS-AGNOSTIC NMS
// =========================================================

static std::vector<Detection> applyNMS(
        std::vector<Detection>& detections) {

    std::sort(
            detections.begin(),
            detections.end(),
            [](const Detection& a,
               const Detection& b) {
                return a.confidence > b.confidence;
            }
    );

    std::vector<Detection> result;

    std::vector<bool> removed(
            detections.size(),
            false
    );

    for (size_t i = 0;
         i < detections.size();
         ++i) {

        if (removed[i]) {
            continue;
        }

        result.push_back(
                detections[i]
        );

        for (size_t j = i + 1;
             j < detections.size();
             ++j) {

            if (removed[j]) {
                continue;
            }

            const float iou =
                    calculateIoU(
                            detections[i],
                            detections[j]
                    );

            if (iou > NMS_THRESHOLD) {
                removed[j] = true;
            }
        }
    }

    return result;
}

// =========================================================
// LOAD MODEL
// =========================================================

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1paddle_1poc_MainActivity_loadYoloModel(
        JNIEnv* env,
        jobject,
        jobject assetManager) {

    std::lock_guard<std::mutex>
            lock(g_model_mutex);

    LOGI("Starting YOLO model loading...");

    // ---------------------------------------------------------
    // ARM64 / Android OpenMP safety settings.
    // ---------------------------------------------------------
    setenv("KMP_AFFINITY", "disabled", 1);
    setenv("OMP_PROC_BIND", "false", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    setenv("KMP_BLOCKTIME", "0", 1);

    LOGI("OpenMP safety settings applied");

    AAssetManager* mgr =
            AAssetManager_fromJava(
                    env,
                    assetManager
            );

    if (mgr == nullptr) {

        LOGE(
                "Unable to obtain AssetManager"
        );

        return JNI_FALSE;
    }

    g_yolo.clear();

    g_model_loaded = false;

    // Emulator-safe settings.
    g_yolo.opt.use_vulkan_compute =
            false;

    // Thread count is configured here,
    // not on ncnn::Extractor.
    g_yolo.opt.num_threads =
            1;

    LOGI(
            "NCNN threads = %d",
            g_yolo.opt.num_threads
    );

    LOGI(
            "Loading model.ncnn.param..."
    );

    const int paramResult =
            g_yolo.load_param(
                    mgr,
                    "yolo/model.ncnn.param"
            );

    if (paramResult != 0) {

        LOGE(
                "Failed loading model param: %d",
                paramResult
        );

        return JNI_FALSE;
    }

    LOGI(
            "model.ncnn.param loaded successfully"
    );

    LOGI(
            "Loading model.ncnn.bin..."
    );

    const int modelResult =
            g_yolo.load_model(
                    mgr,
                    "yolo/model.ncnn.bin"
            );

    if (modelResult != 0) {

        LOGE(
                "Failed loading model weights: %d",
                modelResult
        );

        return JNI_FALSE;
    }

    LOGI(
            "model.ncnn.bin loaded successfully"
    );

    g_model_loaded = true;

    LOGI(
            "YOLO NCNN MODEL LOADED SUCCESSFULLY"
    );

    return JNI_TRUE;
}

// =========================================================
// DETECT OBJECTS
// =========================================================

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_yolo_1paddle_1poc_MainActivity_detectObjects(
        JNIEnv* env,
        jobject,
        jobject bitmap) {

    std::lock_guard<std::mutex>
            lock(g_model_mutex);

    if (!g_model_loaded) {

        LOGE(
                "Detection requested before model loaded"
        );

        return env->NewFloatArray(0);
    }

    // =====================================================
    // BITMAP INFO
    // =====================================================

    AndroidBitmapInfo info{};

    const int infoResult =
            AndroidBitmap_getInfo(
                    env,
                    bitmap,
                    &info
            );

    if (infoResult !=
        ANDROID_BITMAP_RESULT_SUCCESS) {

        LOGE(
                "AndroidBitmap_getInfo failed: %d",
                infoResult
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "Input image = %ux%u stride=%u format=%d",
            info.width,
            info.height,
            info.stride,
            info.format
    );

    if (info.format !=
        ANDROID_BITMAP_FORMAT_RGBA_8888) {

        LOGE(
                "Expected RGBA_8888 bitmap"
        );

        return env->NewFloatArray(0);
    }

    // =====================================================
    // LOCK PIXELS
    // =====================================================

    void* pixels = nullptr;

    const int lockResult =
            AndroidBitmap_lockPixels(
                    env,
                    bitmap,
                    &pixels
            );

    if (lockResult !=
        ANDROID_BITMAP_RESULT_SUCCESS ||
        pixels == nullptr) {

        LOGE(
                "AndroidBitmap_lockPixels failed: %d",
                lockResult
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "Bitmap pixels locked successfully"
    );

    const int originalWidth =
            static_cast<int>(
                    info.width
            );

    const int originalHeight =
            static_cast<int>(
                    info.height
            );

    // =====================================================
    // COPY RGBA SAFELY WITH STRIDE
    // =====================================================

    std::vector<unsigned char>
            rgba(
            static_cast<size_t>(
                    originalWidth
            ) *
            originalHeight *
            4
    );

    const auto* source =
            static_cast<
                    const unsigned char*
                    >(pixels);

    for (int y = 0;
         y < originalHeight;
         ++y) {

        memcpy(
                rgba.data() +
                static_cast<size_t>(y) *
                originalWidth *
                4,

                source +
                static_cast<size_t>(y) *
                info.stride,

                static_cast<size_t>(
                        originalWidth
                ) *
                4
        );
    }

    AndroidBitmap_unlockPixels(
            env,
            bitmap
    );

    LOGI(
            "Bitmap copied successfully"
    );

    // =====================================================
    // RESIZE
    // =====================================================

    const float scale =
            std::min(
                    INPUT_SIZE /
                    static_cast<float>(
                            originalWidth
                    ),

                    INPUT_SIZE /
                    static_cast<float>(
                            originalHeight
                    )
            );

    const int resizedWidth =
            std::max(
                    1,
                    static_cast<int>(
                            std::round(
                                    originalWidth *
                                    scale
                            )
                    )
            );

    const int resizedHeight =
            std::max(
                    1,
                    static_cast<int>(
                            std::round(
                                    originalHeight *
                                    scale
                            )
                    )
            );

    LOGI(
            "Resizing image to %dx%d",
            resizedWidth,
            resizedHeight
    );

    ncnn::Mat resized =
            ncnn::Mat::from_pixels_resize(
                    rgba.data(),
                    ncnn::Mat::PIXEL_RGBA2RGB,
                    originalWidth,
                    originalHeight,
                    resizedWidth,
                    resizedHeight
            );

    if (resized.empty()) {

        LOGE(
                "NCNN resize returned empty Mat"
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "NCNN resize successful dims=%d w=%d h=%d c=%d",
            resized.dims,
            resized.w,
            resized.h,
            resized.c
    );

    rgba.clear();
    rgba.shrink_to_fit();

    // =====================================================
    // LETTERBOX PADDING
    // =====================================================

    const int padWidth =
            INPUT_SIZE -
            resizedWidth;

    const int padHeight =
            INPUT_SIZE -
            resizedHeight;

    const int left =
            padWidth / 2;

    const int right =
            padWidth -
            left;

    const int top =
            padHeight / 2;

    const int bottom =
            padHeight -
            top;

    LOGI(
            "Padding left=%d right=%d top=%d bottom=%d",
            left,
            right,
            top,
            bottom
    );

    // =====================================================
    // MANUAL LETTERBOX
    // =====================================================

    LOGI(
            "Creating manual 640x640 letterbox..."
    );

    ncnn::Mat input(
            INPUT_SIZE,
            INPUT_SIZE,
            3
    );

    if (input.empty()) {

        LOGE(
                "Unable to allocate input tensor"
        );

        return env->NewFloatArray(0);
    }

    input.fill(
            114.f
    );

    LOGI(
            "Input tensor allocated"
    );

    for (int channel = 0;
         channel < 3;
         ++channel) {

        const ncnn::Mat srcChannel =
                resized.channel(
                        channel
                );

        ncnn::Mat dstChannel =
                input.channel(
                        channel
                );

        for (int y = 0;
             y < resizedHeight;
             ++y) {

            const float* srcRow =
                    srcChannel.row(
                            y
                    );

            float* dstRow =
                    dstChannel.row(
                            top + y
                    ) +
                    left;

            memcpy(
                    dstRow,
                    srcRow,
                    static_cast<size_t>(
                            resizedWidth
                    ) *
                    sizeof(float)
            );
        }
    }

    LOGI(
            "Manual letterbox successful: %dx%d",
            input.w,
            input.h
    );

    // =====================================================
    // MANUAL NORMALIZATION
    //
    // 0..255 -> 0..1
    // =====================================================

    LOGI(
            "Starting manual normalization..."
    );

    const float normalizationScale =
            1.0f /
            255.0f;

    for (int channel = 0;
         channel < 3;
         ++channel) {

        ncnn::Mat channelMat =
                input.channel(
                        channel
                );

        for (int y = 0;
             y < INPUT_SIZE;
             ++y) {

            float* row =
                    channelMat.row(
                            y
                    );

            for (int x = 0;
                 x < INPUT_SIZE;
                 ++x) {

                row[x] *=
                        normalizationScale;
            }
        }
    }

    LOGI(
            "Manual normalization successful"
    );

    // =====================================================
    // INFERENCE
    // =====================================================

    LOGI(
            "Starting NCNN inference..."
    );

    ncnn::Extractor extractor =
            g_yolo.create_extractor();

    extractor.set_light_mode(
            true
    );

    // IMPORTANT:
    //
    // Do NOT use:
    //
    // extractor.set_num_threads(...)
    //
    // Your NCNN version does not expose that method.
    //
    // Thread count is already controlled by:
    //
    // g_yolo.opt.num_threads = 2;

    LOGI(
            "Extractor created"
    );

    const int inputResult =
            extractor.input(
                    "in0",
                    input
            );

    LOGI(
            "extractor.input() returned %d",
            inputResult
    );

    if (inputResult != 0) {

        LOGE(
                "Failed setting input: %d",
                inputResult
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "Input tensor accepted"
    );

    ncnn::Mat output;

    LOGI(
            "Extracting out0..."
    );

    const int extractResult =
            extractor.extract(
                    "out0",
                    output
            );

    LOGI(
            "NCNN inference returned: %d",
            extractResult
    );

    if (extractResult != 0) {

        LOGE(
                "Unable to extract out0"
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "out0 dims=%d w=%d h=%d c=%d",
            output.dims,
            output.w,
            output.h,
            output.c
    );

    // =====================================================
    // YOLO OUTPUT DECODING
    //
    // 4 box values + 17 class scores = 21
    // =====================================================

    const int FEATURES =
            4 +
            NUM_CLASSES;

    int predictionCount =
            0;

    bool featuresInRows =
            false;

    if (output.h ==
        FEATURES) {

        featuresInRows =
                true;

        predictionCount =
                output.w;

    } else if (
            output.w ==
            FEATURES) {

        featuresInRows =
                false;

        predictionCount =
                output.h;

    } else {

        LOGE(
                "Unexpected output shape w=%d h=%d",
                output.w,
                output.h
        );

        return env->NewFloatArray(0);
    }

    LOGI(
            "Prediction count = %d",
            predictionCount
    );

    std::vector<Detection>
            detections;

    detections.reserve(
            128
    );

    // =====================================================
    // DECODE
    // =====================================================

    for (int i = 0;
         i < predictionCount;
         ++i) {

        float cx;
        float cy;

        float boxWidth;
        float boxHeight;

        if (featuresInRows) {

            cx =
                    output.row(0)[i];

            cy =
                    output.row(1)[i];

            boxWidth =
                    output.row(2)[i];

            boxHeight =
                    output.row(3)[i];

        } else {

            const float* row =
                    output.row(i);

            cx =
                    row[0];

            cy =
                    row[1];

            boxWidth =
                    row[2];

            boxHeight =
                    row[3];
        }

        // =================================================
        // BEST CONFIDENCE
        //
        // Ignore class ID.
        // =================================================

        float bestConfidence =
                0.f;

        for (int classIndex = 0;
             classIndex <
             NUM_CLASSES;
             ++classIndex) {

            float score;

            if (featuresInRows) {

                score =
                        output.row(
                                4 +
                                classIndex
                        )[i];

            } else {

                const float* row =
                        output.row(i);

                score =
                        row[
                                4 +
                                classIndex
                        ];
            }

            if (score >
                bestConfidence) {

                bestConfidence =
                        score;
            }
        }

        if (bestConfidence <
            CONF_THRESHOLD) {

            continue;
        }

        // =================================================
        // XYWH -> XYXY
        // =================================================

        float x1 =
                cx -
                boxWidth *
                0.5f;

        float y1 =
                cy -
                boxHeight *
                0.5f;

        float x2 =
                cx +
                boxWidth *
                0.5f;

        float y2 =
                cy +
                boxHeight *
                0.5f;

        // Remove letterbox.
        x1 -= left;
        x2 -= left;

        y1 -= top;
        y2 -= top;

        // Restore image coordinates.
        x1 /= scale;
        x2 /= scale;

        y1 /= scale;
        y2 /= scale;

        x1 =
                std::clamp(
                        x1,
                        0.f,
                        static_cast<float>(
                                originalWidth -
                                1
                        )
                );

        x2 =
                std::clamp(
                        x2,
                        0.f,
                        static_cast<float>(
                                originalWidth -
                                1
                        )
                );

        y1 =
                std::clamp(
                        y1,
                        0.f,
                        static_cast<float>(
                                originalHeight -
                                1
                        )
                );

        y2 =
                std::clamp(
                        y2,
                        0.f,
                        static_cast<float>(
                                originalHeight -
                                1
                        )
                );

        if (x2 <= x1 ||
            y2 <= y1) {

            continue;
        }

        detections.push_back(
                {
                        x1,
                        y1,
                        x2,
                        y2,
                        bestConfidence
                }
        );
    }

    LOGI(
            "Candidates before NMS = %d",
            static_cast<int>(
                    detections.size()
            )
    );

    // =====================================================
    // NMS
    // =====================================================

    std::vector<Detection>
            finalDetections =
            applyNMS(
                    detections
            );

    LOGI(
            "Final detections = %d",
            static_cast<int>(
                    finalDetections.size()
            )
    );

    // =====================================================
    // RETURN TO KOTLIN
    // =====================================================

    const int resultLength =
            static_cast<int>(
                    finalDetections.size()
            ) *
            5;

    jfloatArray result =
            env->NewFloatArray(
                    resultLength
            );

    if (resultLength == 0) {

        LOGI(
                "No detections found"
        );

        return result;
    }

    std::vector<float>
            values;

    values.reserve(
            resultLength
    );

    for (const Detection& detection :
            finalDetections) {

        values.push_back(
                detection.x1
        );

        values.push_back(
                detection.y1
        );

        values.push_back(
                detection.x2
        );

        values.push_back(
                detection.y2
        );

        values.push_back(
                detection.confidence
        );
    }

    env->SetFloatArrayRegion(
            result,
            0,
            resultLength,
            values.data()
    );

    LOGI(
            "Detection finished successfully"
    );

    return result;
}

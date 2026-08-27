plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.yolo_paddle_poc"

    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.yolo_paddle_poc"

        minSdk = 26
        targetSdk = 37

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf(
                "arm64-v8a",
                "x86_64"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    androidResources {
        noCompress += listOf(
            "bin",
            "param",
            "onnx"
        )
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {

    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)

    // PaddleOCR module
    implementation(project(":ppocr-sdk"))

    // Required directly by TextEmbeddingMatcher.kt
    implementation(
        "com.microsoft.onnxruntime:onnxruntime-android:1.21.1"
    )

    testImplementation(libs.junit)

    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
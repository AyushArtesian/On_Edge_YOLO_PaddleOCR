plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.paddle.ocr"
    compileSdk = 35

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner =
            "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            consumerProguardFiles("proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {

    implementation(
        "com.microsoft.onnxruntime:onnxruntime-android:1.21.1"
    )

    implementation(
        "com.quickbirdstudios:opencv:4.5.3.0"
    )

    implementation(
        "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0"
    )

    implementation(
        "androidx.core:core-ktx:1.15.0"
    )

    androidTestImplementation(
        "androidx.test:runner:1.6.2"
    )

    androidTestImplementation(
        "androidx.test.ext:junit-ktx:1.2.1"
    )

    androidTestImplementation(
        "androidx.test:monitor:1.7.2"
    )

    androidTestImplementation(
        "junit:junit:4.13.2"
    )
}
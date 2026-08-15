#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-}"
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 <output-dir>" >&2
    exit 1
fi

mkdir -p android/app/assets
rm -rf android/app/assets/data android/app/assets/shaders-builtin
cp -r data android/app/assets/data
cp -r vita3k/shaders-builtin android/app/assets/shaders-builtin

chmod +x android/gradlew

# Force Release build type regardless of keystore presence.
# build.gradle's release signingConfig automatically falls back to
# signingConfigs.debug when SIGNING_* secrets/env vars aren't set,
# so this still produces a valid signed apk without a real keystore.
BUILD_TYPE="Release"

pushd android > /dev/null
./gradlew --stacktrace ":app:assemble${BUILD_TYPE}"
popd > /dev/null

APK_PATH="android/app/build/outputs/apk/${BUILD_TYPE,,}/app-${BUILD_TYPE,,}.apk"
mkdir -p "$OUTPUT_DIR"
cp "$APK_PATH" "$OUTPUT_DIR/app.apk"

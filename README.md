# TacticalCanvas

## Low-latency defaults

Start TacticalCanvas with:

```powershell
.\.venv\Scripts\python.exe tc.py start
```

The vision worker requests 640x480 MJPEG at 60 FPS, keeps only the latest
camera frame, runs MediaPipe asynchronously, disables the OpenCV preview, and
pushes vision changes to browsers immediately. If the webcam only supports 30
FPS, it falls back to the negotiated rate without accumulating stale frames.

Optional overrides:

```powershell
$env:TC_CAMERA_FPS="30"       # force a lower camera request
$env:TC_DETECT_WIDTH="480"    # MediaPipe preprocessing width; 0 disables resize
$env:TC_VISION_PREVIEW="1"    # enable the diagnostic OpenCV window
```

After changing the camera resolution or physically moving the camera/projector,
run `python tc.py calibrate` again.

For the projector itself, enable Game/Low Latency mode and disable motion
interpolation, keystone correction, noise reduction, and other image processing
where the hardware permits. Prefer a direct HDMI connection and the highest
native refresh rate shared by the PC and projector.

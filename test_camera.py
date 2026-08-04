"""
Step 1 script: verify the RealSense camera is detected and streaming.
Run this BEFORE world_model.py. Press 'q' to quit.
"""

import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # NOTE: if your camera is an older SR300/R200 stick (not a D4xx),
    # depth resolution options differ. Try 640x480 @ 30fps first; if the
    # stream fails to start, open "Intel RealSense Viewer" and check what
    # resolutions/formats it lists under the Depth and Color modules,
    # then match those numbers here.
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    print("Camera connected. Streaming... press 'q' in the window to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            combined = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense - Color | Depth (press q to quit)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2


def frames_to_video(frames, output_video_path, fps=30):
    height, width, _ = frames[0].shape
    # 비디오 코덱 지
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs like 'XVID' or 'MJPG'

    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert the frame to BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write the frame to the video file
        video_writer.write(frame_bgr)

    # Release the video writer
    video_writer.release()
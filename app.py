import time
import edgeiq
"""
Use object detection and centroid tracking to count unique human
faces in the frame in realtime.

To change the computer vision model, the engine and accelerator,
and add additional dependencies read this guide:
https://docs.alwaysai.co/application_development/application_configuration.html
"""


def face_enters(object_id, prediction):
    print("Face {} enters".format(object_id))


def face_exits(object_id, prediction):
    print("Face {} exits".format(object_id))


def main():
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/res10_300x300_ssd_iter_140000")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))

    tracker = edgeiq.CorrelationTracker(
            max_objects=5, enter_cb=face_enters, exit_cb=face_exits)

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection and tracking
            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)

                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                objects = tracker.update(results.predictions, frame)

                # Update the label to reflect the object ID
                predictions = []
                for (object_id, prediction) in objects.items():
                    prediction.label = "face {}".format(object_id)
                    text.append("{}".format(prediction.label))
                    predictions.append(prediction)

                frame = edgeiq.markup_image(frame, predictions)
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()

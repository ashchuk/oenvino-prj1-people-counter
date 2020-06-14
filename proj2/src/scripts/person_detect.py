import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''

    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold

        self.num_requests = 1
        self.current_request_id = 0
        self.infer_request_handle = None
        self.input_blob = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def exec_net(self, request_id, frame):
        """
        Starts asynchronous inference for specified request.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param frame: Input image
        :return: Instance of Executable Network class
        """
        self.infer_request_handle = self.net.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        return self.net

    def wait(self, request_id):
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        wait_process = self.net.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net.requests[request_id].outputs[self.out_blob]
        return res

    def load_model(self):
        plugin = IECore()
        self.net = plugin.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)

    def predict(self, image):
        """
        Starts asynchronous inference for specified request and wait for result.
        :param image: Input image
        :return: Image with bounding boxes (or not?)
        """
        inf_start = time.time()
        preprocessed_image = self.preprocess_input(image)
        self.net.exec_net(self.current_request_id, preprocessed_image)
        # Wait for the result
        if self.net.wait(self.current_request_id) == 0:
            det_time = time.time() - inf_start
            print("detection time", det_time)
            # Results of the output layer of the network
            network_result = self.net.get_output(self.current_request_id)
            bounding_boxes = self.preprocess_outputs(network_result)
            return self.draw_outputs(bounding_boxes, image)

    def draw_outputs(self, coords, image):
        """
        Draw bounding boxes on image.

        :param coords: coordinates of bounding box corners
        :param image: image to process
        :return: bounding boxes list and processed and frame
        """
        image_width = image.get(1)
        image_height = image.get(0)

        boxes_result = []
        result_image = image
        for coords_set in coords[0][0]:
            # Draw bounding box for object
            xmin = int(coords_set[0] * image_width)
            ymin = int(coords_set[1] * image_height)
            xmax = int(coords_set[2] * image_width)
            ymax = int(coords_set[3] * image_height)
            result_image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
        return boxes_result, result_image

    def preprocess_outputs(self, outputs):
        # Pass bounding box when it's probability is more than the specified threshold
        bounding_boxes = []
        probabilities = outputs[0, 0, :, 2]
        for i, probability in enumerate(probabilities):
            if probability > self.threshold:
                box = outputs[0, 0, i, 3:]
                bounding_boxes.append(box)
        return bounding_boxes

    def preprocess_input(self, image):
        # Load the network to IE plugin to get shape of input layer
        n, c, h, w = self.net.load_model(args.model, args.device, 1, 1,
                                         self.current_request_id, args.cpu_extension)[1]
        # Change data layout from HWC to CHW
        result_image = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
        result_image = result_image.transpose((2, 0, 1))
        result_image = result_image.reshape((n, c, h, w))
        # Return preprocessed image
        return result_image


def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1

            coords, image = pd.predict(frame)
            num_people = queue.check_coords(coords)
            print(str.format("Total People in frame = {0}", len(coords)))
            print(str.format("Number of people in queue = {0}", num_people))
            out_text = ""
            y_pixel = 25

            for k, v in num_people.items():
                out_text += str.format("No. of People in Queue {0} is {1} ", k, v)
                if v >= int(max_people):
                    out_text += str.format(" Queue full; Please move to next Queue ")
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text = ""
                y_pixel += 40
            out_video.write(image)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)
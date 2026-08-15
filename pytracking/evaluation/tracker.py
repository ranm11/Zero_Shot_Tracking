import importlib
import os
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pathlib import Path
import torch


_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}


def trackerlist(name: str, parameter_name: str, run_ids = None, display_name: str = None):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, run_id, display_name) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, run_id: int = None, display_name: str = None):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
            self.segmentation_dir = '{}/{}/{}'.format(env.segmentation_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
            self.segmentation_dir = '{}/{}/{}_{:03d}'.format(env.segmentation_path, self.name, self.parameter_name, self.run_id)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', self.name))
        if os.path.isdir(tracker_module_abspath):
            tracker_module = importlib.import_module('pytracking.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

        # self.visdom = None


    # def _init_visdom(self, visdom_info, debug):
    #     visdom_info = {} if visdom_info is None else visdom_info
    #     self.pause_mode = False
    #     self.step = False
    #     if debug > 0 and visdom_info.get('use_visdom', True):
    #         try:
    #             self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
    #                                  visdom_info=visdom_info)

    #             # Show help
    #             help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
    #                         'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
    #                         'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
    #                         'block list.'
    #             self.visdom.register(help_text, 'text', 1, 'Help')
    #         except:
    #             time.sleep(0.5)
    #             print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
    #                   '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')

    # def _visdom_ui_handler(self, data):
    #     if data['event_type'] == 'KeyPress':
    #         if data['key'] == ' ':
    #             self.pause_mode = not self.pause_mode

    #         elif data['key'] == 'ArrowRight' and self.pause_mode:
    #             self.step = True


    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        # tracker.visdom = self.visdom
        return tracker

    def run_sequence(self, seq, visualization=None, debug=None, multiobj_mode=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            visdom_info: Visdom info.
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        visualization_ = visualization

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        if visualization is None:
            if debug is None:
                visualization_ = getattr(params, 'visualization', False)
            else:
                visualization_ = True if debug else False

        params.visualization = visualization_
        params.debug = debug_

        self._init_visdom(visdom_info, debug_)
        if visualization_ and self.visdom is None:
            self.init_visualization()

        # Get init information
        init_info = seq.init_info()
        is_single_object = not seq.multiobj_mode

        if multiobj_mode is None:
            multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default' or is_single_object:
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # segmentation[i] is the segmentation mask for frame i (numpy array)

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        # segmentation[i] is the multi-label segmentation mask for frame i (numpy array)

        output = {'target_bbox': [],
                  'time': [],
                  'segmentation': []}

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        if tracker.params.visualization and self.visdom is None:
            self.visualize(image, init_info.get('init_bbox'))

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)

        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'segmentation': init_info.get('init_mask')}

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            while True:
                if not self.pause_mode:
                    break
                elif self.step:
                    self.step = False
                    break
                else:
                    time.sleep(0.1)

            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)

            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

            segmentation = out['segmentation'] if 'segmentation' in out else None
            if self.visdom is not None:
                tracker.visdom_draw_tracking(image, out['target_bbox'], segmentation)
            elif tracker.params.visualization:
                self.visualize(image, out['target_bbox'], segmentation)

        for key in ['target_bbox', 'segmentation']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, None, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': OrderedDict({1: box}), 'init_object_ids': [1, ], 'object_ids': [1, ],
                    'sequence_object_ids': [1, ]}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            if isinstance(out.get('target_bbox'), (OrderedDict, dict)):
                state = [int(s) for s in next(iter(out['target_bbox'].values()))]
            else:
                state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            if tracker.params.debug > 0:
                heatmap = out.get('debug_heatmap')
                region = out.get('debug_heatmap_region')
                if isinstance(heatmap, (OrderedDict, dict)):
                    heatmap = next(iter(heatmap.values()))
                if isinstance(region, (OrderedDict, dict)):
                    region = next(iter(region.values()))
                if heatmap is not None and region is not None:
                    if isinstance(region, (list, tuple, np.ndarray)) and len(region) == 4:
                        x0, y0, heat_w, heat_h = region

                        # Desired full heatmap size for the region (integers)
                        H = int(round(heat_h))
                        W = int(round(heat_w))

                        # Resize with cubic interpolation and smooth to remove block artifacts
                        heatmap_resized = cv.resize(heatmap.astype(np.float32), (W, H), interpolation=cv.INTER_CUBIC)
                        # Apply small blur to smooth remaining grid artifacts
                        if H > 2 and W > 2:
                            heatmap_resized = cv.GaussianBlur(heatmap_resized, (5, 5), 0)

                        # Convert to color map (clip values to [0,1])
                        heatmap_color = cv.applyColorMap((np.clip(heatmap_resized, 0.0, 1.0) * 255).astype(np.uint8), cv.COLORMAP_JET)

                        # Compute visible ROI inside the frame
                        x1 = max(0, int(x0))
                        y1 = max(0, int(y0))
                        x2 = min(frame_disp.shape[1], int(x0) + W)
                        y2 = min(frame_disp.shape[0], int(y0) + H)

                        if x2 > x1 and y2 > y1:
                            # Corresponding crop inside the heatmap_color
                            sx = max(0, -int(x0))
                            sy = max(0, -int(y0))
                            sw = x2 - x1
                            sh = y2 - y1
                            heat_crop = heatmap_color[sy:sy + sh, sx:sx + sw]
                            roi = frame_disp[y1:y2, x1:x2]
                            # Blend with transparency
                            frame_disp[y1:y2, x1:x2] = cv.addWeighted(roi, 0.55, heat_crop, 0.45, 0)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()
        return params


    def init_visualization(self):
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, segmentation=None):
        self.ax.cla()
        self.ax.imshow(image)
        if segmentation is not None:
            self.ax.imshow(segmentation, alpha=0.5)

        if isinstance(state, (OrderedDict, dict)):
            boxes = [v for k, v in state.items()]
        else:
            boxes = (state,)

        for i, box in enumerate(boxes, start=1):
            col = _tracker_disp_colors[i]
            col = [float(c) / 255.0 for c in col]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor=col, facecolor='none')
            self.ax.add_patch(rect)

        if getattr(self, 'gt_state', None) is not None:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        draw_figure(self.fig)

        if self.pause_mode:
            keypress = False
            while not keypress:
                keypress = plt.waitforbuttonpress()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def _read_image(self, image_file: str):
        im = cv.imread(image_file)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)




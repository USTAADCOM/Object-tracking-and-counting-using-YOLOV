"""
Module track the objects with yolo_v5 for flask Web display.
"""
# Python Module
import os
import sys
import time
import argparse
import datetime
import warnings
from pathlib import Path
from collections import Counter
# Path Setup
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
try:
    from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
except Exception:
    sys.path.append('yolov5/utils')
    from dataloaders import VID_FORMATS, LoadImages, LoadStreams
# Module Import
import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression,
                                   scale_coords, cv2,check_imshow, xyxy2xywh,
                                    increment_path, print_args, check_file, 
                                    colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from track_video import (write_text_file, write_srt_file,
                         create_subtitle_video)
# Import Libraries
import cv2
import pandas as pd
warnings.filterwarnings('ignore')
import torch
import torch.backends.cudnn as cudnn
# Setup Environment Variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
logging.getLogger().removeHandler(logging.getLogger().handlers[0])
@torch.no_grad()
def parse_arguments():
    """
    parse_arguments method parse the argument variables.

    Parameters
    ----------
    None

    Return
    ------
    opt: Any
        All the argument for differemt operations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs = '+', type = str,
                        default = WEIGHTS / 'yolov5n.pt',
                        help = 'model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type = str,
                        default = WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type = str,
                        default = 'strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type = str, default = '0',
                        help = 'file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs = '+',
                        type = int, default = [640], help = 'inference size h,w')
    parser.add_argument('--conf-thres', type = float, default = 0.5,
                        help = 'confidence threshold')
    parser.add_argument('--iou-thres', type = float, default = 0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type = int, default = 1000,
                        help = 'maximum detections per image')
    parser.add_argument('--device', default = '',
                        help =  'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action = 'store_true',
                        help = 'display tracking video results')
    parser.add_argument('--save-txt', action = 'store_true',
                        help = 'save results to *.txt')
    parser.add_argument('--save-conf', action = 'store_true',
                        help = 'save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action = 'store_true',
                        help = 'save cropped prediction boxes')
    parser.add_argument('--save-vid', action = 'store_true',
                        help='save video tracking results')
    parser.add_argument('--nosave', action = 'store_true',
                        help='do not save images/videos')
    parser.add_argument('--count', action = 'store_true',
                        help = 'display all MOT counts results on screen')
    parser.add_argument('--draw', action = 'store_true',
                        help = 'display object trajectory lines')
    parser.add_argument('--classes', nargs ='+', type=int,
                        help = 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action = 'store_true',
                        help = 'class-agnostic NMS')
    parser.add_argument('--augment', action = 'store_true',
                        help = 'augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help = 'visualize features')
    parser.add_argument('--update', action = 'store_true',
                        help = 'update all models')
    parser.add_argument('--project', default =ROOT / 'runs/track',
                        help = 'save results to project/name')
    parser.add_argument('--name', default = 'exp',
                        help = 'save results to project/name')
    parser.add_argument('--exist-ok', action = 'store_true',
                        help = 'existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default = 2, type = int,
                        help = 'bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default = False, action = 'store_true',
                        help = 'hide labels')
    parser.add_argument('--hide-conf', default = False, action = 'store_true',
                        help = 'hide confidences')
    parser.add_argument('--hide-class', default = False, action = 'store_true',
                        help = 'hide IDs')
    parser.add_argument('--half', action = 'store_true',
                        help = 'use FP16 half-precision inference')
    parser.add_argument('--dnn', action = 'store_true',
                        help = 'use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt
class VideoCamera:
    """
    Class Doc string.
    """
    def __init__(self, src):
        self.src = src

    def get_frame(self):
        """
        method yiled frames after object detcetion.

        Parameters
        ----------
        None

        Return
        ------
        frame: bytes 
            return live streaming frame with object tracking and 
            counting in bytes form.
        """
        opt = parse_arguments()
        source = self.src
        yolo_weights = WEIGHTS / 'yolov5n.pt'
        strong_sort_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
        imgsz = (640, 640)
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        device = ''
        show_vid = False
        save_txt = True
        save_crop = False
        save_vid = True
        classes = None
        agnostic_nms = False
        augment = False
        visualize = False
        update = False
        project = ROOT / 'runs/track'
        name = 'exp'
        exist_ok = False
        hide_labels = False
        hide_conf = False
        hide_class = False
        half = False
        dnn = False
        count = True
        draw = False
        source = str(source)
        time_stamp_name = str(datetime.datetime.now().timestamp()).replace(".", "")
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)
        if not isinstance(yolo_weights, list):
            exp_name = yolo_weights.stem
        exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok = exist_ok)
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents = True, exist_ok=True)
        device = select_device(device)
        model = DetectMultiBackend(yolo_weights, device = device,
                                   dnn = dnn, data = None, fp16 = half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s = stride)
        trajectory = {}
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size = imgsz, stride = stride, auto = pt)
            nr_sources = len(dataset)
        else:
            dataset = LoadImages(source, img_size = imgsz, stride = stride, auto = pt)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources,[None] * nr_sources, [None] * nr_sources
        cfg = get_config()
        cfg.merge_from_file(opt.config_strongsort)
        strongsort_list = []
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    strong_sort_weights,
                    device,
                    max_dist = cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance = cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age = cfg.STRONGSORT.MAX_AGE,
                    n_init = cfg.STRONGSORT.N_INIT,
                    nn_budget = cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda = cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha = cfg.STRONGSORT.EMA_ALPHA,
                )
            )
        outputs = [None] * nr_sources
        model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            t2 = time_sync()
            dt[0] += t2 - t1
            visualize = increment_path(save_dir / Path(path[0]).stem,
                                       mkdir=True) if visualize else False
            pred = model(im, augment = augment, visualize = visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            pred = non_max_suppression(pred, conf_thres, iou_thres,
                                       classes, agnostic_nms,
                                       max_det = max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / time_stamp_name)
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / time_stamp_name)
                    else:
                        txt_file_name = p.parent.name
                        save_path = str(save_dir / time_stamp_name)
                curr_frames[i] = im0
                txt_path = str(save_dir / 'tracks' / txt_file_name)
                s += '%gx%g ' % im.shape[2:]
                imc = im0.copy() if save_crop else im0
                annotator = Annotator(im0, line_width = 2, pil = not ascii)
                if cfg.STRONGSORT.ECC:
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                if det is not None and len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    temp_dict = {}
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        temp_dict[names[int(c)]] = f"{n}"
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    write_text_file(temp_dict, p.name, time_stamp_name)
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    t4 = time_sync()
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(),
                                                           confs.cpu(), 
                                                           clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes 
                            if draw:
                                center = ((int(bboxes[0]) + int(bboxes[2])) // 2,
                                          (int(bboxes[1]) + int(bboxes[3])) // 2)
                                if id not in trajectory:
                                    trajectory[id] = []
                                trajectory[id].append(center)
                                for i1 in range(1,len(trajectory[id])):
                                    if trajectory[id][i1-1] is None or trajectory[id][i1] is None:
                                        continue
                                    thickness = 2
                                    try:
                                        cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                                    except Exception:
                                        pass
                            if save_txt:
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                with open(txt_path+'.txt', 'a', encoding = "utf8") as f:
                                    f.write(('%g ' * 11 + '\n') % (frame_idx + 1, cls, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
                            if save_vid or save_crop or show_vid:
                                c = int(cls)
                                id = int(id)
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                    (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color = colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
                else:
                    strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')
                if count:
                    item_dict = {}
                    try:
                        df = pd.read_csv(txt_path +'.txt' , header = None, delim_whitespace = True)
                        df = df.iloc[:,0:3]
                        df.columns=["frameid" ,"class","trackid"]
                        df = df[['class','trackid']]
                        df = (df.groupby('trackid')['class']
                                .apply(list)
                                .apply(lambda x:sorted(x))
                                ).reset_index()
                        df.colums = ["trackid","class"]
                        df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                        vc = df['class'].value_counts()
                        vc = dict(vc)
                        vc2 = {}
                        for key, val in enumerate(names):
                            vc2[key] = val
                        item_dict = dict((vc2[key], value) for (key, value) in vc.items())
                        item_dict  = dict(sorted(item_dict.items(), key=lambda item: item[0]))
                    except Exception:
                        pass
                    if save_txt:
                        display = im0.copy()
                        h, w = im0.shape[0], im0.shape[1]
                        x1 = 10
                        y1 = 10
                        x2 = 10
                        y2 = 70
                        txt_size = cv2.getTextSize(str(temp_dict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                        cv2.putText(im0, '{}'.format(temp_dict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                        cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)
                if show_vid:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        break
                if save_vid:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path,
                                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                                        fps, (w, h))
                    vid_writer[i].write(im0)
                prev_frames[i] = curr_frames[i]
            _, jpeg = cv2.imencode('.jpg', im0)
            yield jpeg.tobytes()
        vid_writer[i].release()
        t = tuple(x / seen * 1E3 for x in dt)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        time.sleep(4)
        write_srt_file(save_path, p.name, time_stamp_name)
        create_subtitle_video(save_path, p.name, time_stamp_name)

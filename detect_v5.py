import argparse
import os
import sys
from pathlib import Path
import statistics

import pandas as pd
import torch
from torchvision import transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, print_args, scale_coords
from utils.torch_utils import select_device

import classifier
import iou_cal


def cls_to_char(t_data, cls_idx):
    row = t_data[t_data[0] == int(cls_idx)]
    if len(row) > 0:
        return str(row.iloc[0, 1])
    return f"[{cls_idx}]"


def _col_center_x(col):
    xs = [item["cx"] for item in col]
    return statistics.median(xs)


def _merge_close_columns(columns, merge_thresh):
    if not columns:
        return columns

    columns = sorted(columns, key=lambda col: -_col_center_x(col))
    merged = [columns[0]]

    for col in columns[1:]:
        prev = merged[-1]
        if abs(_col_center_x(prev) - _col_center_x(col)) <= merge_thresh:
            prev.extend(col)
        else:
            merged.append(col)

    return merged


def reorder_for_vertical_rtl(pred_bbox_list, pred_class_list, t_data):
    items = []
    widths = []

    for bbox, cls_idx in zip(pred_bbox_list, pred_class_list):
        x1, y1, x2, y2 = map(int, bbox[:4])
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ch = cls_to_char(t_data, cls_idx)

        widths.append(w)
        items.append({
            "bbox": [x1, y1, x2, y2],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "char": ch,
        })

    if not items:
        return []

    median_w = statistics.median(widths)
    assign_thresh = max(18, median_w * 0.9)
    merge_thresh = max(12, median_w * 0.55)

    items.sort(key=lambda d: -d["cx"])

    columns = []
    for item in items:
        best_idx = None
        best_dist = float("inf")

        for idx, col in enumerate(columns):
            col_x = _col_center_x(col)
            dist = abs(item["cx"] - col_x)
            if dist <= assign_thresh and dist < best_dist:
                best_idx = idx
                best_dist = dist

        if best_idx is None:
            columns.append([item])
        else:
            columns[best_idx].append(item)

    columns = _merge_close_columns(columns, merge_thresh)
    columns.sort(key=lambda col: -_col_center_x(col))

    for col in columns:
        col.sort(key=lambda d: (d["cy"], d["cx"]))

    ordered = []
    for col in columns:
        ordered.extend(col)

    return ordered


def build_ancient_text(pred_bbox_list, pred_class_list, t_data):
    ordered = reorder_for_vertical_rtl(pred_bbox_list, pred_class_list, t_data)
    chars = [item["char"] for item in ordered]
    return "".join(chars)


def load_models(device_name=""):
    weights = ROOT / "weights_detector.pt"
    imgsz = (1280, 1280)

    device = select_device(device_name)
    detector = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, pt = detector.stride, detector.pt
    imgsz = check_img_size(imgsz, s=stride)

    t_data = pd.read_csv(ROOT / "class.csv", header=None)
    t_data = t_data.iloc[:, [0, 1]]

    with open(ROOT / "class.csv", "r", encoding="UTF-8") as f:
        class_num = int(f.readlines()[-1].split(",")[0]) + 1

    cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model = classifier.mnistsimple_Classifier_Model(class_num).to(cls_device)
    classifier_model.load_state_dict(torch.load(ROOT / "weights_classifier.pth", map_location=cls_device))
    classifier_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return detector, classifier_model, t_data, transform, device, cls_device, imgsz, stride, pt


@torch.no_grad()
def extract_ancient_text(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"이미지 경로가 파일이 아닙니다: {image_path}")

    detector, classifier_model, t_data, transform, device, _, imgsz, stride, pt = load_models()

    dataset = LoadImages(str(image_path), img_size=imgsz, stride=stride, auto=pt)
    detector.warmup(imgsz=(1, 3, *imgsz))

    ancient_text = ""

    for path, im, im0s, vid_cap, s, origin_img, x_pad, y_pad in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if detector.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = detector(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.25, None, False, max_det=1000)

        for det in pred:
            if not len(det):
                ancient_text = ""
                continue

            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            pred_bbox_list = iou_cal.tensor_to_list(det)

            ocrdataset = classifier.mnistsimple_Dataset(
                im0s,
                pred_bbox_list,
                [],
                transforms=transform,
            )
            ocrloader = torch.utils.data.DataLoader(
                ocrdataset,
                batch_size=1,
                shuffle=False,
            )

            pred_class_list = classifier.get_predictions(
                classifier_model,
                device,
                ocrloader,
                [],
            )

            ancient_text = build_ancient_text(pred_bbox_list, pred_class_list, t_data)

    return ancient_text


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="OCR을 수행할 이미지 파일 경로")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    ancient_text = extract_ancient_text(opt.image)
    print("ancient_text =", ancient_text)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

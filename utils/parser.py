'''
Argument parser file
'''
import argparse


def merge_others_parser():
    '''
    Parse arguments of merge landmark
    '''
    parser = argparse.ArgumentParser(
        description='Merge RetinaFace lm predictions')
    parser.add_argument('--input-train-path', type=str,
                        help='Path to training file to merge lm')
    parser.add_argument('--input-val-path', type=str,
                        help='Path to validating file to merge lm')
    parser.add_argument('--input-test-path', type=str,
                        help='Path to testing file to merge lm')
    parser.add_argument('--pixta-lm-train-path', type=str, default='',
                        help='Path to training lm prediction file')
    parser.add_argument('--pixta-lm-val-path', type=str, default='',
                        help='Path to validating lm prediction file')
    parser.add_argument('--pixta-lm-test-path', type=str, default='',
                        help='Path to testing lm prediction file')
    parser.add_argument('--pixta-human-train-path', type=str,
                        help='Path to Pixta human training prediction file')
    parser.add_argument('--pixta-human-val-path', type=str,
                        help='Path to Pixta human validating prediction file')
    parser.add_argument('--pixta-human-test-path', type=str,
                        help='Path to Pixta human testing prediction file')
    parser.add_argument('--pixta-root-train-path', type=str,
                        help='Path to Pixta training image root')
    parser.add_argument('--pixta-root-val-path', type=str,
                        help='Path to Pixta validating image root')
    parser.add_argument('--pixta-root-test-path', type=str,
                        help='Path to Pixta testing image root')
    parser.add_argument('--fake-face-paths', type=str,
                        help='Path to fake face prediction file')
    parser.add_argument('--fake-root-paths', type=str,
                        help='Path to fake root images')
    parser.add_argument('--wider-human-path', type=str,
                        help='Path to Widerface human prediction file')
    parser.add_argument('--wider-gt-path', type=str,
                        help='Path to Widerface GT file')
    parser.add_argument('--wider-root-path', type=str,
                        help='Path to Widerface image root')
    parser.add_argument('--output-ann-path', type=str,
                        help='Path to output training file')
    parser.add_argument('--output-img-path', type=str,
                        help='Path to output img folder')
    parser.add_argument('--train-val-split', type=float,
                        help='Train Val split ratio of other datasets')
    parser.add_argument('--lm-threshold', type=int,
                        help='Number of lms belong to each bbox')
    parser.add_argument('--lm-conf-threshold', type=float,
                        help='Confidence threshold to choose lm')
    parser.add_argument('--fake-conf-threshold', type=float,
                        help='Confidence threshold to choose fake face predictions')
    parser.add_argument('--human-conf-threshold', type=float,
                        help='Confidence threshold to choose human predictions')
    parser.add_argument('--human-label-names', type=str,
                        help='Label name of human class')
    args = parser.parse_args()
    return args


def focus_generator_parser():
    '''
    Parse arguments of focus generator
    '''
    parser = argparse.ArgumentParser(
        description='Generating chip')
    parser.add_argument('--input-path', type=str,
                        help='Path to file to generate focus mask')
    parser.add_argument('--output-path', type=str,
                        help='Path to output file')
    parser.add_argument('--dont-care-low', type=int,
                        help='')
    parser.add_argument('--dont-care-high', type=int,
                        help='')
    parser.add_argument('--small-threshold', type=int,
                        help='')
    parser.add_argument('--label-w', type=int,
                        help='Label mask width')
    parser.add_argument('--label-h', type=int,
                        help='Label mask height')
    parser.add_argument('--stride', type=int,
                        help='Stride to downsample from image to mask')
    args = parser.parse_args()
    return args


def chip_generator_parser():
    '''
    Parse arguments of chip generator
    '''
    parser = argparse.ArgumentParser(
        description='Generating chip')
    parser.add_argument('--input-train-path', type=str,
                        help='Path to training file to generate chip')
    parser.add_argument('--input-val-path', type=str,
                        help='Path to validating file to generate chip')
    parser.add_argument('--input-test-path', type=str,
                        help='Path to testing file to generate chip')
    parser.add_argument('--root-path', type=str,
                        help='Path to root images path')
    parser.add_argument('--ori-img-test-path', type=str,
                        help='Path to save original test images')
    parser.add_argument('--ori-json-test-path', type=str,
                        help='Path to save original test json')
    parser.add_argument('--out-train-path', type=str,
                        help='Path to output training json file')
    parser.add_argument('--out-val-path', type=str,
                        help='Path to output validating json file')
    parser.add_argument('--out-test-path', type=str,
                        help='Path to output testing json file')
    parser.add_argument('--chip-save-path', type=str,
                        help='Path to output chips folder')
    parser.add_argument('--valid-range', type=int,
                        help='Valid range of size of bbox for each chip size')
    parser.add_argument('--c-stride', type=int,
                        help='Stride while sliding chip')
    parser.add_argument('--mapping-threshold', type=float,
                        help='Threshold to map our data to WIDERFACE data')
    parser.add_argument('--training-size', type=int,
                        help='Size of image to training detection')
    parser.add_argument('--n-threads', type=int,
                        help='Num of threads to run multi processing')
    parser.add_argument('--use-neg', type=int,
                        help='Generate negative chip or not')
    args = parser.parse_args()
    return args


def preprocessing_coco_parser():
    '''
    Parse arguments of preprocessing coco
    '''
    parser = argparse.ArgumentParser(
        description='Preprocess COCO dataset form')
    parser.add_argument('--input', type=str,
                        help='Path to file or folder to preprocess')
    parser.add_argument('--root-path', type=str,
                        help='Path to root images path')
    parser.add_argument('--final-out', type=str,
                        help='Path to merged COCO json file')
    parser.add_argument('--n-per-chunk', type=int,
                        help='Num of images each chunk to run multi processing')
    parser.add_argument('--n-threads', type=int,
                        help='Num of threads to run multi processing')
    parser.add_argument('--split-ratio', type=str,
                        help='Ratio to split in to train, val and test')
    args = parser.parse_args()
    return args


def cvat_tool_parser():
    '''
    Parse arguments of cvat tool
    '''
    parser = argparse.ArgumentParser(
        description='Getting annotations from CVAT')
    parser.add_argument('--username', type=str,
                        help='Username to login to CVAT')
    parser.add_argument('--password', type=str,
                        help='Password to login to CVAT')
    parser.add_argument('--data-name', type=str,
                        help='Name of data to download')
    parser.add_argument('--id-list', type=str,
                        help='List of task id to download')
    parser.add_argument('--out-folder', type=str,
                        help='Output folder to save annotations')

    args = parser.parse_args()
    return args


def train_parser():
    '''
    Parse arguments of training process
    '''
    parser = argparse.ArgumentParser(description='RetinaFocus training script.')
    parser.add_argument('--train-json-path', type=str, required=True,
                        help='Training dataset.')
    parser.add_argument('--val-json-path', type=str, required=True,
                        help='Val dataset.')
    parser.add_argument('--image-train-dir', type=str, required=True,
                        help='The directory that contains train images.')
    parser.add_argument('--image-val-dir', type=str, required=True,
                        help='The directory that contains validation images.')
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--retinaface-weights-path', type=str,
                        help='Path to RetinaFace weights.')
    parser.add_argument('--exclude-top-retinaface', action='store_true',
                        help='Exclude loading weights from top layers \
                             (BboxHead, ClassHead, LandmarkHead) of the RetinaFace model.')
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not using cuda')

    args = parser.parse_args()
    return args


def infer_parser():
    '''
    Parse arguments of inference process
    '''
    parser = argparse.ArgumentParser(description='RetinaFocus inference script.')
    parser.add_argument('--test-json-path', type=str,
                        help='Testing dataset.')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='The directory that contains images.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to RetinaFocus model.')
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to save output images')
    parser.add_argument('--early-stop', action='store_true',
                        help='Use early stopping to save inferene time')
    parser.add_argument('--model-type', type=str,
                        help='Type of model [torch, onnx].')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not using cuda')

    args = parser.parse_args()
    return args


def demo_parser():
    '''
    Parse arguments of the demo
    '''
    parser = argparse.ArgumentParser(description='RetinaFocus demo script.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to RetinaFocus model.')
    parser.add_argument('--cfg-path', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to save output images')
    parser.add_argument('--model-type', type=str,
                        help='Type of model [torch, onnx].')
    parser.add_argument('--no-cuda', action='store_true',
                        help='do not using cuda')
    parser.add_argument('--img-root', type=str,
                        help="Image root folder / file for demo")

    args = parser.parse_args()
    return args
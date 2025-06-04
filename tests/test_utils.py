import sys
import types
import importlib.util
import os
import numpy as np


def load_inf_utils():
    dummy = types.ModuleType('dummy')
    # Basic stubs
    sys.modules.setdefault('cv2', dummy)
    sys.modules.setdefault('tensorflow', dummy)
    PIL = types.ModuleType('PIL')
    PIL.Image = dummy
    sys.modules.setdefault('PIL', PIL)
    # deep_sort and submodules
    deep_sort = types.ModuleType('deep_sort')
    nn_matching = types.ModuleType('nn_matching')
    tracker = types.ModuleType('tracker')
    tracker.Tracker = object
    utils = types.ModuleType('utils')
    utils.create_obj_infos = dummy
    utils.linear_inter_bbox = dummy
    utils.filter_short_objs = dummy
    deep_sort.nn_matching = nn_matching
    deep_sort.tracker = tracker
    deep_sort.utils = utils
    sys.modules.setdefault('deep_sort', deep_sort)
    sys.modules.setdefault('deep_sort.nn_matching', nn_matching)
    sys.modules.setdefault('deep_sort.tracker', tracker)
    sys.modules.setdefault('deep_sort.utils', utils)
    application_util = types.ModuleType('application_util')
    application_util.preprocessing = dummy
    sys.modules.setdefault('application_util', application_util)
    class_ids = types.ModuleType('class_ids')
    class_ids.targetClass2id_new_nopo = dummy
    class_ids.coco_obj_to_actev_obj = dummy
    sys.modules.setdefault('class_ids', class_ids)
    nn = types.ModuleType('nn')
    nn.resnet_fpn_backbone = dummy
    nn.fpn_model = dummy
    sys.modules.setdefault('nn', nn)
    pred_models = types.ModuleType('pred_models')
    pred_models.Model = dummy
    sys.modules.setdefault('pred_models', pred_models)
    pred_utils = types.ModuleType('pred_utils')
    pred_utils.activity2id = dummy
    sys.modules.setdefault('pred_utils', pred_utils)

    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'code', 'inference', 'inf_utils.py')
    with open(file_path, 'r') as f:
        lines = f.readlines()

    selected = []
    selected.extend(lines[344:349])
    selected.extend(lines[776:791])
    namespace = {'np': np}
    exec(''.join(selected), namespace)
    return types.SimpleNamespace(**{k: namespace[k] for k in ['get_nearest', 'relative_to_abs']})


def test_relative_to_abs():
    inf_utils = load_inf_utils()
    rel = np.array([[1, 0], [1, 1], [0, 1]])
    start = [0, 0]
    expected = np.array([[1, 0], [2, 1], [2, 2]])
    out = inf_utils.relative_to_abs(rel, start)
    np.testing.assert_array_equal(out, expected)


def test_get_nearest():
    inf_utils = load_inf_utils()
    frames = [0, 10, 20, 30]
    assert inf_utils.get_nearest(frames, 17) == 20

"""DEV-0050：W1 的 Python 沙箱（src/w1/sandbox.py）。

钉住：白名单代码跑通并取回 mask；任务书点名的每一种被禁止的 import 与调用都被拒（AST 拒绝或运行期判违规，
绝不 ok）；超时与超内存被结束；子进程环境变量为空；stdout 截断；不用 pickle；结果确定。
"""

import numpy as np
import pytest

from src.w1 import sandbox
from src.w1.sandbox import (
    STDOUT_LIMIT_CHARS,
    SandboxConfig,
    check_code,
    from_import_allowed,
    module_allowed,
    run_code,
)

H, W = 24, 32


@pytest.fixture(scope="module")
def image():
    img = np.zeros((H, W, 3), np.uint8)
    img[4:16, 8:24] = (200, 30, 30)
    return img


@pytest.fixture(scope="module")
def masks():
    m = np.zeros((H, W), bool)
    m[5:15, 9:23] = True
    m[9, 15] = False  # 一个洞
    return {"a": m}


def test_module_docstring_states_threat_model():
    doc = sandbox.__doc__
    assert "研究诚信的防线" in doc and "不是对抗性安全边界" in doc


# ---------------------------------------------------------------------------
# 白名单
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["numpy", "numpy.linalg", "scipy", "scipy.ndimage", "scipy.signal", "cv2", "skimage", "skimage.segmentation",
     "skimage.feature", "math", "itertools", "collections", "collections.abc", "functools", "json", "re", "statistics"],
)
def test_module_allowed(name):
    assert module_allowed(name)


@pytest.mark.parametrize(
    "name",
    ["torch", "transformers", "os", "sys", "subprocess", "socket", "pathlib", "importlib", "io", "builtins", "ctypes",
     "pickle", "random", "PIL", "cv2.dnn", "cv2.ml", "skimage.data", "skimage.io", "skimage.future", "scipy.io",
     "scipy.datasets", "numpy.ctypeslib", "numpy.testing", "numpy._core", "skimage.feature._cascade"],
)
def test_module_rejected(name):
    assert not module_allowed(name)


def test_from_import_rules():
    assert from_import_allowed("scipy", "ndimage") and from_import_allowed("scipy", "signal")
    assert from_import_allowed("numpy", "select") and from_import_allowed("skimage", "segmentation")
    for mod, name in [("cv2", "dnn"), ("cv2", "dnn_Net"), ("cv2", "CascadeClassifier"), ("cv2", "HOGDescriptor"),
                      ("cv2", "imread"), ("cv2", "os"), ("skimage", "data"), ("skimage.feature", "Cascade"),
                      ("numpy", "load"), ("numpy", "ctypeslib"), ("scipy", "io"), ("cv2", "*"), ("numpy", "_core")]:
        assert not from_import_allowed(mod, name), (mod, name)


def test_check_code_accepts_ordinary_code():
    code = (
        "import numpy as np\nimport scipy.ndimage as ndi\nfrom scipy import signal\nimport cv2\n"
        "from skimage import segmentation, measure\nimport math, itertools, collections, functools, json, re, statistics\n"
        "x = np.select([image[..., 0] > 100], [1])\nclass A:\n    def __init__(self, v):\n        self.v = v\n"
        "result_mask = masks['a']\n"
    )
    assert check_code(code) == []


@pytest.mark.parametrize(
    "code",
    [
        "import torch", "import transformers", "import os", "import socket", "import subprocess", "import pathlib",
        "import importlib", "from os import path", "import cv2.dnn", "from cv2 import dnn", "import cv2\ncv2.dnn.readNet('m.onnx')",
        "import cv2\nc = cv2.CascadeClassifier('f.xml')", "import cv2\nh = cv2.HOGDescriptor()", "import skimage.data",
        "from skimage import data", "import skimage.feature\nc = skimage.feature.Cascade('x')", "f = open('/etc/hosts')",
        "m = __import__('os')", "eval('1')", "exec('x = 1')", "compile('1', 'f', 'eval')", "import cv2\nd = getattr(cv2, 'dnn')",
        "x = (1).__class__", "import numpy as np\nf = np.lib._npyio_impl", "from numpy import *", "from . import x",
        "import cv2\no = cv2.os", "import numpy as np\nnp.zeros(3).tofile('x')", "g = globals()", "b = __builtins__",
    ],
)
def test_check_code_rejects(code):
    assert check_code(code), code


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------


def test_whitelisted_code_runs_and_returns_mask(image, masks):
    code = (
        "import numpy as np\nimport scipy.ndimage as ndi\nfrom scipy import signal\nimport cv2\n"
        "from skimage import color, filters, segmentation, morphology, measure\n"
        "g = color.rgb2gray(image)\ne = filters.sobel(g)\nd = morphology.binary_dilation(masks['a'], morphology.disk(1))\n"
        "lab = segmentation.slic(image, n_segments=4, start_label=1)\nedges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)\n"
        "k = signal.convolve2d(g, np.ones((3, 3)), mode='same')\ns = np.select([g > 0.2], [1])\n"
        "print('H', H, 'W', W, 'area', int(masks['a'].sum()))\nresult_mask = ndi.binary_fill_holes(masks['a'])\n"
    )
    r = run_code(code, image=image, masks=masks)
    assert r.status == "ok", r.error
    assert r.violations == ()
    assert r.stdout.strip() == f"H {H} W {W} area {int(masks['a'].sum())}"
    assert r.result_mask.dtype == bool and r.result_mask.shape == (H, W)
    assert np.array_equal(r.result_mask, np.pad(np.ones((10, 14), bool), ((5, H - 15), (9, W - 23))))
    assert r.code == code


def test_result_mask_optional_and_int01_accepted(image, masks):
    r = run_code("print(1)", image=image, masks=masks)
    assert r.ok and r.result_mask is None
    r = run_code("import numpy as np\nresult_mask = masks['a'].astype(np.uint8)", image=image, masks=masks)
    assert r.ok and np.array_equal(r.result_mask, masks["a"])


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\nresult_mask = np.zeros((3, 3), bool)",
        "import numpy as np\nresult_mask = np.full((H, W), 2)",
        "import numpy as np\nresult_mask = np.empty((H, W), dtype=object)",  # 不用 pickle：对象数组不收
    ],
)
def test_bad_result_mask_is_error(image, masks, code):
    r = run_code(code, image=image, masks=masks)
    assert r.status == "error" and r.result_mask is None and "result_mask" in r.error


# 任务书点名的禁止项（torch、transformers、os、socket、cv2.dnn、CascadeClassifier、skimage.data、open、__import__），
# 外加只能在运行期拦下的别名写法。
FORBIDDEN_RUNS = [
    ("import torch", "rejected"),
    ("import transformers", "rejected"),
    ("import os", "rejected"),
    ("import socket", "rejected"),
    ("import cv2\nnet = cv2.dnn.readNet('model.onnx')", "rejected"),
    ("import cv2\nc = cv2.CascadeClassifier('face.xml')", "rejected"),
    ("import skimage.data", "rejected"),
    ("f = open('/etc/hosts')", "rejected"),
    ("m = __import__('os')", "rejected"),
    # 运行期（第 2–4 层）：AST 看不出来的写法
    ("import cv2\nnet = cv2.dnn_Net()", "violation"),
    ("import cv2 as c\nm = c.ml.SVM_create()", "violation"),
    ("import cv2\nt = cv2.TrackerMIL_create()", "violation"),
    ("import cv2\np = cv2.data.haarcascades", "violation"),
    ("import skimage\nx = skimage.data.camera()", "violation"),
    ("import numpy as np\nx = np.load('/tmp/x.npy')", "violation"),
    ("import numpy\nf = numpy.lib.npyio\nx = f.load('/tmp/x.npy')", "violation"),
    ("import numpy as np\nnp.testing.assert_equal(1, 1)", "violation"),
    ("import scipy\nx = scipy.io.loadmat('x.mat')", "rejected"),
    ("import skimage.feature as F\nc = F.learn_gmm", "rejected"),
]


@pytest.mark.parametrize("code,status", FORBIDDEN_RUNS)
def test_forbidden_imports_and_calls_are_refused(image, masks, code, status):
    r = run_code(code + "\nimport numpy as np\nresult_mask = np.ones((H, W), bool)", image=image, masks=masks)
    assert r.status == status, (r.status, r.error)
    assert r.result_mask is None
    assert r.violations


def test_violation_cannot_be_swallowed(image, masks):
    code = "import numpy as np\ntry:\n    np.load('/tmp/x.npy')\nexcept Exception:\n    pass\nresult_mask = masks['a']"
    r = run_code(code, image=image, masks=masks)
    assert r.status == "violation" and r.result_mask is None and r.violations == ("numpy.load",)
    code = "import numpy as np\ntry:\n    np.load('/tmp/x.npy')\nexcept:\n    pass\nresult_mask = masks['a']"
    r = run_code(code, image=image, masks=masks)  # 光秃的 except 能接住异常，但违规照样记下，结果作废
    assert r.status == "violation" and r.result_mask is None


def test_library_internals_keep_their_own_file_reads(image, masks):
    """skimage.morphology 导入时用 np.load 读自带的整数表：库内部调用不算违规（USER_ONLY 一类）。"""
    r = run_code("from skimage import morphology\nd = morphology.disk(3)\nprint(int(d.sum()))", image=image, masks=masks)
    assert r.status == "ok", r.error
    assert r.stdout.strip() == "29"


# ---------------------------------------------------------------------------
# RUN-0059 回归：库内部的导入（私有子模块、可选依赖）不算违规；用户代码的导入照旧按白名单判
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["y = masks['a'].mean()", "s = image.std()", "mx = image.max()", "v = image.var()"])
def test_c_level_lazy_import_of_private_numpy_submodule_is_allowed(image, masks, code):
    """这些 ndarray 方法第一次调用时，numpy 的 C 代码经 PyImport_Import 导入 numpy._core._methods；当前帧是用户代码，
    所以导入经过受限的 __import__。修复前这里判成 "violation: 'import numpy._core._methods'"（HPC 与开发机都能复现）。"""
    r = run_code(code + "\nresult_mask = masks['a']", image=image, masks=masks)
    assert r.status == "ok", r.error
    assert r.violations == () and r.result_mask is not None


def _fake_library_dir(root):
    """假的 pooch、torch（被执行就留下标记文件并抛 RuntimeError）与一个模拟库代码的模块 w1test_lib。"""
    lib = root / "lib"
    lib.mkdir()
    marks = root / "marks"
    marks.mkdir()
    for name in ("pooch", "torch"):
        (lib / f"{name}.py").write_text(
            f"open({str(marks / (name + '.loaded'))!r}, 'w').close()\nraise RuntimeError('fake {name} was executed')\n"
        )
    (lib / "w1test_lib.py").write_text(
        f"open({str(marks / 'w1test_lib.loaded')!r}, 'w').close()\n"
        "import importlib\n"
        "\n"
        "def optional():\n"
        "    out = {}\n"
        "    for name in ('pooch', 'torch'):\n"
        "        try:\n"
        "            importlib.import_module(name)\n"
        "            out[name] = 'loaded'\n"
        "        except ImportError as e:\n"
        "            out[name] = type(e).__name__\n"
        "    try:\n"
        "        import torch  # 语句形式的可选导入\n"
        "        out['torch_stmt'] = 'loaded'\n"
        "    except ImportError as e:\n"
        "        out['torch_stmt'] = type(e).__name__\n"
        "    return out\n"
        "\n"
        "def private_numpy():\n"
        "    m = importlib.import_module('numpy._core._methods')\n"
        "    import numpy._core._internal as internal\n"
        "    return m.__name__, internal.__name__\n"
    )
    return lib, marks


def test_library_optional_imports_of_blocked_modules_behave_as_missing(image, masks, tmp_path):
    """假的 pooch、torch 就在子进程的 sys.path 上；库代码（非用户代码）可选地导入它们：得到 ModuleNotFoundError，
    运行仍是 ok、没有违规，它们的模块代码从未执行。库代码导入 numpy 的私有子模块也照常可用。"""
    lib, marks = _fake_library_dir(tmp_path)
    cfg = SandboxConfig(test_sys_path=(str(lib),), test_preload=("w1test_lib",))
    code = "print(w1test_lib.optional())\nprint(w1test_lib.private_numpy())\nresult_mask = masks['a']"
    r = run_code(code, image=image, masks=masks, config=cfg)
    assert r.status == "ok", r.error
    assert r.violations == () and np.array_equal(r.result_mask, masks["a"])
    out = r.stdout.splitlines()
    assert eval(out[0]) == {"pooch": "ModuleNotFoundError", "torch": "ModuleNotFoundError", "torch_stmt": "ModuleNotFoundError"}
    assert eval(out[1]) == ("numpy._core._methods", "numpy._core._internal")
    assert {"pooch", "torch"} <= set(r.blocked_imports)
    assert (marks / "w1test_lib.loaded").exists()  # 标记机制本身有效
    assert not (marks / "pooch.loaded").exists() and not (marks / "torch.loaded").exists()  # 被挡的模块从未加载


def test_real_library_with_fake_pooch_installed(image, masks, tmp_path):
    """HPC 上装了 pooch：skimage 导入时会可选地 import pooch。在开发机上用假的 pooch 复现：slic 照常跑完，不算违规。"""
    lib, marks = _fake_library_dir(tmp_path)
    code = "from skimage.segmentation import slic\nlab = slic(image, n_segments=6, start_label=1)\nresult_mask = lab == lab[10, 16]"
    r = run_code(code, image=image, masks=masks, config=SandboxConfig(test_sys_path=(str(lib),)))
    assert r.status == "ok", r.error
    assert r.violations == () and "pooch" in r.blocked_imports
    assert not (marks / "pooch.loaded").exists()


@pytest.mark.parametrize("code", ["import numpy._core", "import pooch", "import torch", "from numpy._core import _methods"])
def test_user_imports_of_private_or_blocked_modules_are_still_refused(image, masks, tmp_path, code):
    lib, marks = _fake_library_dir(tmp_path)
    r = run_code(code, image=image, masks=masks, config=SandboxConfig(test_sys_path=(str(lib),)))
    assert r.status == "rejected" and r.violations, (r.status, r.error)
    assert not (marks / "pooch.loaded").exists() and not (marks / "torch.loaded").exists()


@pytest.mark.parametrize("code,ok", [
    ("import numpy", True), ("from numpy import linalg", True), ("import math", True),
    ("import numpy._core", False), ("import pooch", False), ("import torch", False), ("import os", False),
    ("from numpy import load", False), ("from numpy._core import _methods", False),
])
def test_guarded_import_alone_polices_user_import_statements(monkeypatch, code, ok):
    """第 2 层单独（不经 AST 检查）也能拦下用户代码里的 import 语句。在本进程里直接执行，下毒换成空操作，免得改动测试进程的 numpy。"""
    monkeypatch.setattr(sandbox, "_poison_now", lambda: None)
    monkeypatch.setattr(sandbox, "_VIOLATIONS", [])
    g = {"__builtins__": {"__import__": sandbox._guarded_import}, "__name__": "__w1_sandbox__"}
    compiled = compile(code, sandbox.CODE_FILENAME, "exec", dont_inherit=True)
    if ok:
        exec(compiled, g)
        assert sandbox._VIOLATIONS == []
    else:
        with pytest.raises(sandbox.SandboxViolation):
            exec(compiled, g)
        assert sandbox._VIOLATIONS


def test_user_exception_and_syntax_error(image, masks):
    r = run_code("x = 1\ny = x / 0", image=image, masks=masks)
    assert r.status == "error" and "ZeroDivisionError" in r.error and "line 2" in r.error
    r = run_code("def f(:\n  pass", image=image, masks=masks)
    assert r.status == "error" and r.error.startswith("SyntaxError")


def test_stdout_is_truncated(image, masks):
    r = run_code("print('x' * 10000)", image=image, masks=masks)
    assert r.ok and len(r.stdout) == STDOUT_LIMIT_CHARS and r.stdout_truncated


def test_timeout_kills_process(image, masks):
    r = run_code("while True:\n    pass", image=image, masks=masks, config=SandboxConfig(timeout_s=3.0))
    assert r.status == "timeout" and r.result_mask is None
    assert r.wall_time_s < 10.0


def test_memory_limit_kills_process(image, masks):
    # 限额 2 GB，按 50 MB 一块分配到 5 GB：Linux 上 RLIMIT_AS 让分配失败（MemoryError），macOS 上看门狗结束进程。
    # 限额不取更小：Linux 上 RLIMIT_AS 管的是虚拟地址空间，python + numpy 启动本身就要几百 MB。
    code = "import numpy as np\nxs = []\nfor i in range(100):\n    xs.append(np.ones(50 * 2**20, np.uint8))\nprint(len(xs))"
    r = run_code(code, image=image, masks=masks, config=SandboxConfig(memory_bytes=2 * 2**30, timeout_s=60.0))
    assert r.status == "memory", (r.status, r.error)
    assert r.result_mask is None


def test_environment_is_empty(image, masks, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/should/not/leak")
    monkeypatch.setenv("SECRET_TOKEN", "x")
    r = run_code("print(1)", image=image, masks=masks)
    assert r.ok
    # 父进程传 env={}；唯一可能出现的键是解释器自己按 PEP 538 设的 LC_CTYPE（C 语言环境强制改成 UTF-8）
    assert set(r.env_keys) <= {"LC_CTYPE"}, r.env_keys
    assert any(g.startswith("watchdog:") for g in r.memory_guard), r.memory_guard


def test_results_are_deterministic(image, masks):
    code = (
        "import numpy as np\nfrom skimage.segmentation import slic\nlab = slic(image, n_segments=6, start_label=1)\n"
        "print(lab.max(), int(lab.sum()))\nresult_mask = lab == lab[10, 16]"
    )
    a = run_code(code, image=image, masks=masks)
    b = run_code(code, image=image, masks=masks)
    assert a.ok and b.ok
    assert a.stdout == b.stdout and np.array_equal(a.result_mask, b.result_mask)


def test_invalid_inputs_raise(image):
    with pytest.raises(ValueError):
        run_code("print(1)", image=image[..., 0], masks={})
    with pytest.raises(ValueError):
        run_code("print(1)", image=image, masks={"bad name!": np.zeros((H, W), bool)})
    with pytest.raises(ValueError):
        run_code("print(1)", image=image, masks={"a": np.zeros((H, W), np.uint8)})
    with pytest.raises(ValueError):
        SandboxConfig(timeout_s=0)

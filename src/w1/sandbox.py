"""W1 工作台的 Python 沙箱：`run_python` 工具的执行端（DEV-0050；D103 §二.2"编排"一行）。

定位（先读这一段）：
- 这是**研究诚信的防线**：防止被测的 LLM 借代码调用学习型视觉模型、读取预训练权重、联网下载或读写任意文件，
  保证"能力只来自 LLM 加不含学习的算法"这一主张成立（D103 §二.2 的禁止项）。
- 它**不是对抗性安全边界**：拦截针对的是正常写出来的代码（包括无意中写出的 `import torch`、
  `cv2.dnn.readNet(...)`、`np.load(...)`），不保证挡住刻意构造的逃逸。被测模型不是攻击者；
  真正的隔离（容器、seccomp、无网络的命名空间）不在本模块的范围内。每段代码原文都进工作台的日志，事后可审计。

执行方式：
- 子进程 `python -I -B <本文件> <临时目录>`：`-I` 不读 PYTHON* 环境变量、不加用户 site、不把脚本目录放进 sys.path；
  环境变量为空（父进程传 `env={}`，不带 HF_HOME、PATH 等）；工作目录是一次性的临时目录。
  本文件既是父进程的模块，也是子进程的脚本，规则只在这里定义一次，两边共用。
- 限时：默认 20 s，到时由父进程结束子进程。
- 限内存：默认 4 GB。Linux 上用 `resource.setrlimit(RLIMIT_AS)`；另有子进程内的看门狗线程每 10 ms 读一次内存占用
  （Linux 是峰值常驻内存，macOS 是含压缩页的 phys_footprint），超了就以退出码 86 结束。
  macOS 不支持 RLIMIT_AS（setrlimit 报错），只有看门狗。
- 子进程在导入 numpy 之前把 BLAS/OpenMP 线程数设为 1（写进它自己的 os.environ，不来自父进程）：
  结果与线程数无关、逐位确定，RLIMIT_AS 也不会被每线程的内存预留撑爆。
- 输入：图像（uint8 HxWx3 RGB）与命名 mask（bool），经 npz 传入；输出：stdout（截到 4,000 字符），
  以及代码里赋给 `result_mask` 的 mask，经 npz 取回。读写一律 `allow_pickle=False`，不用 pickle。

拦截分四层：
1. 启动前的 AST 检查（父进程）：import 白名单；禁用名字（open、exec、eval、compile、__import__、getattr 等，
   以及一切双下划线名字）；禁用属性（以下划线开头的属性、标准库模块名、帧对象属性、明确的学习型 / 文件读写 API 名）。
2. 受限的内建（子进程）：用户代码的 `__builtins__` 只含白名单里的内建；`import` 语句走带白名单的 `__import__`
   （名字以双下划线开头，第 1 层不让用户代码直接引用它）。白名单只管用户代码里的 import 语句；用户代码调用的
   C 函数在内部发起的导入（如 `image.max()` 第一次导入 numpy._core._methods）属于库内部，私有子模块照常导入。
3. `sys.meta_path` 拦截器（子进程，对所有导入者生效，包括白名单库的内部）：学习型模型框架与下载器一律不能加载，
   导入它们只得到 ModuleNotFoundError（库的可选依赖回退照常工作），记进 blocked_imports，不算违规；
   白名单包（numpy、scipy、cv2、skimage）的每个模块加载完，都重新执行第 4 层。
4. 属性下毒（子进程）：白名单包里被屏蔽的属性（`cv2.dnn`、`cv2.CascadeClassifier`、`skimage.data`、`np.load` 等）
   换成一调用、一取属性就报违规的占位对象；公开子模块里对同一对象的别名也一并换掉。
任何一次违规都记进结果的 violations，这次运行判为 violation，返回的 mask 作废。
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# 默认限额与文件约定
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEMORY_BYTES = 4 * 1024**3
STDOUT_LIMIT_CHARS = 4000
ERROR_LIMIT_CHARS = 2000
CODE_FILENAME = "<w1-sandbox>"
MASK_KEY_PREFIX = "mask__"
MASK_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
STATUSES = ("ok", "error", "rejected", "violation", "timeout", "memory", "crash")

_EXIT_MEMORY = 86
_WATCHDOG_INTERVAL_S = 0.01
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_CONFIG_FILE = "config.json"
_CODE_FILE = "code.py"
_INPUT_FILE = "input.npz"
_OUTPUT_FILE = "output.npz"
_RESULT_FILE = "result.json"

# ---------------------------------------------------------------------------
# 规则（唯一定义处；父进程的 AST 检查与子进程的运行期拦截共用）
# ---------------------------------------------------------------------------

# 用户代码可以 import 的顶层模块（任务书第 3 项）。白名单包的子模块默认允许，下面屏蔽的除外。
ALLOWED_MODULES = frozenset(
    {"numpy", "scipy", "cv2", "skimage", "math", "itertools", "collections", "functools", "json", "re", "statistics"}
)

# 白名单包里被屏蔽的属性（也包括子模块：`import a.b` 与 `from a import b` 同样按这里判定）。分两类：
# - STRICT：学习型模型、预训练权重、样例数据与下载。谁调用都报违规。
# - USER_ONLY：文件 / 设备读写、编译与加载本地代码。只对用户代码屏蔽，白名单库内部照常可用：
#   例如 skimage.morphology 导入时用 np.load 读它自带的圆盘分解表（整数表，不是权重）。
#   判定看直接调用者的代码文件名是不是用户代码（CODE_FILENAME）。
BLOCKED_STRICT: dict[str, tuple[str, ...]] = {
    "scipy": ("datasets", "misc"),
    "skimage": ("data", "future", "data_dir"),  # data 含预训练的级联分类器文件；future 是可训练的分割器
    "skimage.feature": ("Cascade", "learn_gmm", "fisher_vector"),
    "cv2": (
        # DNN 与带预训练模型的模块 / 类（OpenCV 4.10 与 5.x 的名字取并集；当前版本没有的照样占位）
        "dnn", "dnn_superres", "face", "text", "ml", "gapi", "ximgproc", "xfeatures2d", "xobjdetect", "saliency",
        "wechat_qrcode", "mcc", "barcode", "xphoto", "optflow", "legacy",
        "CascadeClassifier", "CascadeClassifier_convert",
        "HOGDescriptor", "HOGDescriptor_getDefaultPeopleDetector", "HOGDescriptor_getDaimlerPeopleDetector",
        "FaceDetectorYN", "FaceDetectorYN_create", "FaceRecognizerSF", "FaceRecognizerSF_create",
        # 预训练文件与样例数据的路径
        "data", "samples", "utils",
    ),
}
BLOCKED_USER_ONLY: dict[str, tuple[str, ...]] = {
    "numpy": (
        "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "fromregex",
        "memmap", "DataSource",
        "ctypeslib", "f2py", "distutils",
        "testing",  # numpy.testing.extbuild 会编译并加载 C 扩展
    ),
    "numpy.lib": ("DataSource",),
    "numpy.lib.npyio": ("load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromregex", "DataSource", "NpzFile"),
    "numpy.lib.format": ("open_memmap", "read_array", "write_array"),
    "scipy": ("io", "LowLevelCallable"),
    "scipy.sparse": ("load_npz", "save_npz"),
    "skimage": ("io",),
    "cv2": (
        "FileStorage", "FileNode", "VideoCapture", "VideoWriter", "haveImageReader", "haveImageWriter", "imcount",
        "readOpticalFlow", "writeOpticalFlow", "loadMesh", "saveMesh", "loadPointCloud", "savePointCloud",
        "loadChromaticAberrationParams",
    ),
}
# 按前缀屏蔽（cv2 的 Python 绑定把子模块的类也平铺成 `dnn_Net`、`ml_SVM` 这样的顶层名字）。
BLOCKED_PREFIXES_STRICT: dict[str, tuple[str, ...]] = {
    "cv2": (
        "dnn_", "ml_", "face_", "text_", "gapi_", "GInfer", "GNet", "barcode_", "mcc_", "wechat_qrcode_", "xphoto_",
        "ximgproc_", "xfeatures2d_", "saliency_", "legacy_", "optflow_",
        "Tracker",  # 视频跟踪器：GOTURN、DaSiamRPN、Nano、Vit 要 DNN 模型，MIL 是在线学习
    ),
}
BLOCKED_PREFIXES_USER_ONLY: dict[str, tuple[str, ...]] = {
    "cv2": ("imread", "imwrite", "VideoCapture_", "VideoWriter_", "FileStorage_", "FileNode_"),
}

# 任何导入者都不能加载的顶层模块（第 3 层）：学习型模型框架、带预训练模型的库、下载器。
# socket、subprocess 等标准库模块不在这里：scipy.ndimage、numpy.testing 的导入链本身就会加载它们；
# 用户代码碰不到它们靠的是第 1、2 层的白名单。
FORBIDDEN_EVERYWHERE = frozenset(
    {
        "torch", "torchvision", "torchaudio", "transformers", "tensorflow", "keras", "jax", "jaxlib", "flax",
        "onnx", "onnxruntime", "openvino", "tflite_runtime", "sklearn", "timm", "open_clip", "clip",
        "segment_anything", "sam2", "ultralytics", "mmcv", "mmdet", "mmseg", "mmengine", "detectron2",
        "paddle", "paddleocr", "easyocr", "pytesseract", "tesserocr", "kornia", "accelerate", "peft",
        "safetensors", "huggingface_hub", "diffusers", "mediapipe", "dlib", "face_recognition", "insightface",
        "cellpose", "stardist", "xgboost", "lightgbm", "requests", "urllib3", "httpx", "aiohttp", "pooch", "dask",
        # skimage 里只服务于被屏蔽的 Cascade / learn_gmm 的私有模块（懒加载，平时不会被导入）
        "skimage.feature._cascade", "skimage.feature._fisher_vector",
    }
)

# AST：不论属于哪个对象都拒绝的属性名（明确的学习型 / 文件读写 API、标准库模块名、帧对象属性）。
# 以下划线开头的属性另行一律拒绝。
FORBIDDEN_ATTR_NAMES = frozenset(
    {
        "dnn", "dnn_superres", "CascadeClassifier", "HOGDescriptor", "Cascade", "learn_gmm", "fisher_vector",
        "FaceDetectorYN", "FaceRecognizerSF", "readNet", "readNetFromONNX", "readNetFromCaffe",
        "readNetFromTensorflow", "readNetFromTorch", "readNetFromDarknet", "readNetFromTFLite",
        "readNetFromModelOptimizer", "blobFromImage", "blobFromImages",
        "imread", "imreadmulti", "imwrite", "imwritemulti", "FileStorage", "VideoCapture", "VideoWriter",
        "tofile", "dump", "fromfile", "memmap", "open_memmap", "loadtxt", "savetxt", "genfromtxt", "fromregex",
        "load_npz", "save_npz", "loadmat", "savemat", "ctypeslib", "f2py", "DataSource", "LowLevelCallable",
        "data_dir",
        # 标准库与第三方模块名：防止经白名单库的属性（如 cv2.os、numpy.testing 的 socket）拿到它们。
        # signal、select 不在内：它们与 scipy.signal、np.select 同名（"io" 在内，顺带挡住 scipy.io、skimage.io）。
        "os", "sys", "subprocess", "socket", "pathlib", "importlib", "builtins", "io", "shutil", "ctypes",
        "pickle", "glob", "tempfile", "inspect", "platform", "threading", "multiprocessing", "marshal",
        "gc", "resource", "site", "sysconfig", "urllib", "http", "ssl", "requests", "pooch", "torch",
        "transformers", "lazy_loader", "posixpath", "zipfile", "tarfile", "gzip", "bz2", "lzma", "unittest",
        "asyncio", "selectors", "mmap", "runpy", "pkgutil", "zipimport",
        # 帧与代码对象：拿到它们就能摸到沙箱自身的全局变量
        "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "gi_frame", "gi_code", "cr_frame", "cr_code",
        "ag_frame", "ag_code", "tb_frame", "tb_next", "co_code",
    }
)

# AST：禁用的名字（另外，一切以双下划线开头的名字都禁用）。
FORBIDDEN_NAMES = frozenset(
    {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
     "input", "breakpoint", "help", "exit", "quit"}
)

# 用户代码可用的内建（其余一律没有）。
SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod", "complex",
    "dict", "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hasattr", "hash", "hex", "id",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "oct", "ord",
    "pow", "print", "property", "range", "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod", "str",
    "sum", "super", "tuple", "type", "zip", "NotImplemented", "Ellipsis",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "FloatingPointError", "ImportError",
    "IndexError", "KeyError", "LookupError", "MemoryError", "ModuleNotFoundError", "NameError",
    "NotImplementedError", "OverflowError", "RecursionError", "RuntimeError", "StopIteration", "TypeError",
    "UnboundLocalError", "ValueError", "ZeroDivisionError", "Warning", "RuntimeWarning", "UserWarning",
)

# 第 3、4 层作用的包。
_POISON_PACKAGES = ("numpy", "scipy", "cv2", "skimage")


def _blocked_kind(module: str, attr: str) -> str | None:
    """'strict'、'user_only' 或 None（不屏蔽）。"""
    if attr in BLOCKED_STRICT.get(module, ()) or any(attr.startswith(p) for p in BLOCKED_PREFIXES_STRICT.get(module, ())):
        return "strict"
    if attr in BLOCKED_USER_ONLY.get(module, ()) or any(attr.startswith(p) for p in BLOCKED_PREFIXES_USER_ONLY.get(module, ())):
        return "user_only"
    return None


def attr_blocked(module: str, attr: str) -> bool:
    """白名单包 `module` 的属性（或子模块）`attr` 对用户代码是否被屏蔽（两类都算）。"""
    return _blocked_kind(module, attr) is not None


def module_allowed(name: str) -> bool:
    """用户代码能否 `import name`：顶层在白名单里，任何一段都不以下划线开头，路径上没有被屏蔽的子模块。"""
    parts = name.split(".")
    if parts[0] not in ALLOWED_MODULES:
        return False
    if any(not p or p.startswith("_") for p in parts):
        return False
    return not any(attr_blocked(".".join(parts[:i]), parts[i]) for i in range(1, len(parts)))


def from_import_allowed(module: str, name: str) -> bool:
    """`from module import name` 里的 name 是否允许（module 本身另行用 module_allowed 判定）。"""
    if name == "*" or name.startswith("_") or name in FORBIDDEN_ATTR_NAMES:
        return False
    return not attr_blocked(module, name)


def check_code(code: str) -> list[str]:
    """启动前的 AST 检查（第 1 层）。返回违规说明的列表（空 = 通过）；语法错误照常抛 SyntaxError。"""
    tree = ast.parse(code, filename=CODE_FILENAME, mode="exec")
    problems: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                if not module_allowed(a.name):
                    problems.append(f"line {line}: import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"line {line}: relative imports are not allowed")
            elif not module_allowed(node.module or ""):
                problems.append(f"line {line}: import from '{node.module}' is not allowed")
            else:
                for a in node.names:
                    if not from_import_allowed(node.module, a.name):
                        problems.append(f"line {line}: 'from {node.module} import {a.name}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                problems.append(f"line {line}: name '{node.id}' is not available")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTR_NAMES:
                problems.append(f"line {line}: attribute '.{node.attr}' is not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.asname is not None and (a.asname.startswith("__") or a.asname in FORBIDDEN_NAMES):
                    problems.append(f"line {line}: alias '{a.asname}' is not allowed")
    return list(dict.fromkeys(problems))


# ---------------------------------------------------------------------------
# 父进程接口
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxConfig:
    """沙箱限额。python 为 None 时用当前解释器（sys.executable）。"""

    timeout_s: float = DEFAULT_TIMEOUT_S
    memory_bytes: int = DEFAULT_MEMORY_BYTES
    stdout_chars: int = STDOUT_LIMIT_CHARS
    python: str | None = None
    # 仅供回归测试（RUN-0059）：在开发机上不装任何东西，复现"库内部导入可选依赖、私有子模块"的情形。工作台从不设置。
    # test_sys_path：追加到子进程 sys.path 末尾的目录（放假的 pooch、torch 与模拟库代码的模块）；
    # test_preload：沙箱在用户代码之前导入的模块名（按库代码导入，不过白名单），以同名变量交给用户代码。
    test_sys_path: tuple[str, ...] = ()
    test_preload: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (isinstance(self.timeout_s, (int, float)) and self.timeout_s > 0):
            raise ValueError(f"timeout_s 必须为正，收到 {self.timeout_s!r}")
        if isinstance(self.memory_bytes, bool) or not isinstance(self.memory_bytes, int) or self.memory_bytes <= 0:
            raise ValueError(f"memory_bytes 必须是正整数，收到 {self.memory_bytes!r}")
        if isinstance(self.stdout_chars, bool) or not isinstance(self.stdout_chars, int) or self.stdout_chars < 0:
            raise ValueError(f"stdout_chars 必须是非负整数，收到 {self.stdout_chars!r}")


@dataclass(frozen=True, eq=False)
class SandboxResult:
    """一次 run_code 的结果。status ∈ STATUSES；只有 ok 时 result_mask 可能非空。"""

    status: str
    code: str
    stdout: str = ""
    stdout_truncated: bool = False
    error: str | None = None
    result_mask: Any = None  # np.ndarray (H, W) bool 或 None
    violations: tuple[str, ...] = ()
    blocked_imports: tuple[str, ...] = ()  # 白名单库内部试图加载、被第 3 层挡掉的模块（不算违规，只记录）
    env_keys: tuple[str, ...] = ()  # 子进程启动时 os.environ 的键（父进程传空环境）
    memory_guard: tuple[str, ...] = ()  # 实际生效的内存限制方式：rlimit_as、watchdog
    wall_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> dict:
        """可写 JSON 的摘要（不含 mask 与代码原文）。"""
        return {
            "status": self.status,
            "stdout_chars": len(self.stdout),
            "stdout_truncated": self.stdout_truncated,
            "error": self.error,
            "violations": list(self.violations),
            "blocked_imports": list(self.blocked_imports),
            "env_keys": list(self.env_keys),
            "memory_guard": list(self.memory_guard),
            "has_mask": self.result_mask is not None,
            "wall_time_s": self.wall_time_s,
        }


def _check_inputs(image, masks):
    import numpy as np

    img = np.asarray(image)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image 必须是 (H, W, 3) uint8，收到 dtype {img.dtype}、形状 {img.shape}")
    out = {}
    for name, m in (masks or {}).items():
        if not isinstance(name, str) or not MASK_NAME_RE.match(name):
            raise ValueError(f"mask 名字不合法: {name!r}")
        a = np.asarray(m)
        if a.dtype != bool or a.shape != img.shape[:2]:
            raise ValueError(f"mask {name!r} 必须是与图像同尺寸的 bool 数组，收到 dtype {a.dtype}、形状 {a.shape}")
        out[name] = a
    return img, out


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _tail(b: bytes, n: int = ERROR_LIMIT_CHARS) -> str:
    return b.decode("utf-8", errors="replace")[-n:].strip()


def run_code(code: str, *, image, masks: Mapping[str, Any] | None = None, config: SandboxConfig | None = None) -> SandboxResult:
    """在子进程沙箱里执行 code。用户代码可见的变量：image（HxWx3 uint8 RGB）、masks（名字 → HxW bool）、H、W；
    赋给 result_mask 的 HxW 布尔数组作为结果取回。出错不抛异常，按 status 返回（输入本身不合法时抛 ValueError）。"""
    import numpy as np

    if not isinstance(code, str):
        raise TypeError(f"code 必须是 str，收到 {type(code).__name__}")
    cfg = config or SandboxConfig()
    img, ms = _check_inputs(image, masks)
    t0 = time.perf_counter()

    def _done(status: str, **kw) -> SandboxResult:
        return SandboxResult(status=status, code=code, wall_time_s=time.perf_counter() - t0, **kw)

    try:
        problems = check_code(code)
    except SyntaxError as e:
        return _done("error", error=f"SyntaxError: {e.msg} (line {e.lineno})")
    if problems:
        return _done("rejected", error="; ".join(problems)[:ERROR_LIMIT_CHARS], violations=tuple(problems))

    with tempfile.TemporaryDirectory(prefix="w1sbx_") as tmp:
        with open(os.path.join(tmp, _CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(
                {"memory_bytes": cfg.memory_bytes, "stdout_chars": cfg.stdout_chars,
                 "test_sys_path": list(cfg.test_sys_path), "test_preload": list(cfg.test_preload)},
                f,
            )
        with open(os.path.join(tmp, _CODE_FILE), "w", encoding="utf-8") as f:
            f.write(code)
        np.savez(os.path.join(tmp, _INPUT_FILE), image=img, **{MASK_KEY_PREFIX + k: v for k, v in ms.items()})
        cmd = [cfg.python or sys.executable, "-I", "-B", os.path.abspath(__file__), tmp]
        proc = subprocess.Popen(
            cmd, cwd=tmp, env={}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            out, err = proc.communicate(timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
            out, err = proc.communicate()
        finally:
            if proc.poll() is None:
                _kill(proc)
                proc.wait()
        if timed_out:
            return _done("timeout", error=f"timed out after {cfg.timeout_s:g} s; the process was killed")
        if proc.returncode == _EXIT_MEMORY:
            return _done("memory", error=f"memory limit exceeded ({cfg.memory_bytes / 1024**3:g} GB); the process was killed")
        result_path = os.path.join(tmp, _RESULT_FILE)
        if not os.path.exists(result_path):
            return _done("crash", error=f"sandbox process exited with code {proc.returncode}: {_tail(err) or _tail(out)}")
        with open(result_path, encoding="utf-8") as f:
            r = json.load(f)
        mask = None
        if r["status"] == "ok" and r.get("has_mask"):
            with np.load(os.path.join(tmp, _OUTPUT_FILE), allow_pickle=False) as z:
                mask = np.array(z["result_mask"], dtype=bool)
            if mask.shape != img.shape[:2]:  # 子进程已校验；这里再挡一次，防止形状错位的 mask 进工作台
                return _done("crash", error=f"result_mask shape {mask.shape} != image {img.shape[:2]}")
        return _done(
            r["status"],
            stdout=r.get("stdout", ""),
            stdout_truncated=bool(r.get("stdout_truncated", False)),
            error=r.get("error"),
            result_mask=mask,
            violations=tuple(r.get("violations", ())),
            blocked_imports=tuple(r.get("blocked_imports", ())),
            env_keys=tuple(r.get("env_keys", ())),
            memory_guard=tuple(r.get("memory_guard", ())),
        )


# ---------------------------------------------------------------------------
# 子进程端（本文件作为脚本运行时）
# ---------------------------------------------------------------------------


class SandboxViolation(BaseException):
    """用户代码碰了被禁止的东西。继承 BaseException，`except Exception` 吞不掉。"""


_VIOLATIONS: list[str] = []
_BLOCKED_LOADS: list[str] = []
_STUBS: dict[int, "_Blocked"] = {}
_KEEP_ALIVE: list[Any] = []  # 被换掉的原对象：留着引用，id 不会被复用
_SWEPT: set[str] = set()
_SWEEP_STATE = {"n_stubs": -1, "depth": 0}  # depth：正在执行的白名单模块加载的嵌套层数


def _violation(what: str):
    _VIOLATIONS.append(what)
    raise SandboxViolation(
        f"'{what}' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)"
    )


class _Blocked:
    """被屏蔽属性的占位。

    - strict：调用、取属性、设属性，谁做都报违规。
    - user_only：直接调用者是用户代码（CODE_FILENAME）时报违规；白名单库内部的调用与取属性转给原对象。
      原对象还没加载（懒加载的子模块）时，库内部第一次用到再按 lazy 的模块名导入。
    - 双下划线属性一律照常报 AttributeError，免得库内部的探测（hasattr(x, "__array__") 之类）出错。
    """

    __slots__ = ("_w1_name", "_w1_orig", "_w1_strict", "_w1_lazy")

    def __init__(self, name: str, orig: Any, strict: bool, lazy: str | None = None) -> None:
        object.__setattr__(self, "_w1_name", name)
        object.__setattr__(self, "_w1_orig", orig)
        object.__setattr__(self, "_w1_strict", strict)
        object.__setattr__(self, "_w1_lazy", lazy)

    def _w1_target(self, what: str, caller_depth: int = 2):
        """违规就抛 SandboxViolation；否则返回原对象。帧 0 是这里、1 是 __call__ 等、2 是真正的调用者。"""
        if self._w1_strict or sys._getframe(caller_depth).f_code.co_filename == CODE_FILENAME:
            _violation(what)
        orig = self._w1_orig
        if orig is None and self._w1_lazy is not None:
            import importlib

            orig = importlib.import_module(self._w1_lazy)
            object.__setattr__(self, "_w1_orig", orig)
        if orig is None:
            raise AttributeError(self._w1_name)
        return orig

    def __call__(self, *args, **kwargs):
        return self._w1_target(self._w1_name)(*args, **kwargs)

    def __getattr__(self, attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return getattr(self._w1_target(f"{self._w1_name}.{attr}"), attr)

    def __setattr__(self, attr: str, value) -> None:
        setattr(self._w1_target(f"{self._w1_name}.{attr}"), attr, value)

    def __repr__(self) -> str:
        return f"<blocked: {self._w1_name}>"


def _is_identifiable(obj: Any) -> bool:
    """能按 id 识别别名的对象：模块、类、可调用对象。int、str 等常量不算（小整数的 id 是共享的）。"""
    import types

    return isinstance(obj, (types.ModuleType, type)) or (callable(obj) and not isinstance(obj, (int, float, str, bytes)))


def _poison_now() -> None:
    """第 4 层：把白名单包里被屏蔽的属性换成占位对象；再把公开子模块里指向同一对象的别名也换掉。幂等。"""
    for strict, table, prefixes in (
        (True, BLOCKED_STRICT, BLOCKED_PREFIXES_STRICT),
        (False, BLOCKED_USER_ONLY, BLOCKED_PREFIXES_USER_ONLY),
    ):
        for modname in sorted(set(table) | set(prefixes)):
            mod = sys.modules.get(modname)
            if mod is None:
                continue
            d = mod.__dict__
            cands = set(table.get(modname, ()))
            for p in prefixes.get(modname, ()):
                cands.update(n for n in list(d) if n.startswith(p))
            for n in sorted(cands):
                cur = d.get(n)
                if isinstance(cur, _Blocked):
                    continue
                full = f"{modname}.{n}"
                if cur is None:
                    cur = sys.modules.get(full)  # 已加载、但还没设成父包属性的子模块
                if cur is not None and _is_identifiable(cur):
                    stub = _STUBS.get(id(cur))
                    if stub is None:
                        stub = _STUBS[id(cur)] = _Blocked(full, cur, strict)
                        _KEEP_ALIVE.append(cur)
                else:
                    stub = _Blocked(full, cur, strict, lazy=full if cur is None else None)
                setattr(mod, n, stub)
    if len(_STUBS) != _SWEEP_STATE["n_stubs"]:  # 有新的被屏蔽对象：全部公开模块重扫一遍
        _SWEPT.clear()
        _SWEEP_STATE["n_stubs"] = len(_STUBS)
    for name, mod in list(sys.modules.items()):
        if name in _SWEPT or mod is None:
            continue
        parts = name.split(".")
        if parts[0] not in _POISON_PACKAGES or any(p.startswith("_") for p in parts):
            continue
        _SWEPT.add(name)
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for k, v in items:
            if k.startswith("_") or isinstance(v, _Blocked):
                continue
            stub = _STUBS.get(id(v))
            if stub is not None:
                setattr(mod, k, stub)


class _PoisonLoader:
    """包一层加载器：最外层的白名单模块执行完，重新下毒一次。

    只在最外层下毒：包自己初始化的过程中（例如 cv2 的 __init__ 逐个加载 cv2.dnn 等子模块的 Python 部分）
    不能把它要用的属性换掉。嵌套加载的子模块在最外层执行完之前就已设成父包的属性，所以最后这一次下毒能覆盖它们。
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        _SWEEP_STATE["depth"] += 1
        try:
            self._inner.exec_module(module)
        finally:
            _SWEEP_STATE["depth"] -= 1
        if _SWEEP_STATE["depth"] == 0:
            _poison_now()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _Finder:
    """第 3 层：sys.meta_path 的第一个拦截器。"""

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname in FORBIDDEN_EVERYWHERE or fullname.split(".")[0] in FORBIDDEN_EVERYWHERE:
            _BLOCKED_LOADS.append(fullname)
            raise ModuleNotFoundError(f"No module named '{fullname}' (blocked in the W1 sandbox)", name=fullname)
        if fullname.split(".")[0] not in _POISON_PACKAGES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PoisonLoader(spec.loader)
                return spec
        return None


_IMPORT_NAME_OPCODE = None  # dis.opmap["IMPORT_NAME"]，子进程里第一次用到时取


def _import_from_user_statement(depth: int = 2) -> bool:
    """这次 `__import__` 是不是用户代码里的 import 语句发起的。

    用户代码的 `__builtins__` 是受限字典，所以凡是"当前帧是用户代码"时发生的导入都会经过 _guarded_import，包括两种：
    - 用户写的 import 语句：调用者帧是用户代码，且正在执行 IMPORT_NAME 指令；
    - 用户代码调用的 C 函数在内部导入（PyImport_Import 取当前帧的 builtins）：例如 `image.max()` 第一次调用时
      numpy 的 C 代码导入 numpy._core._methods。此时调用者帧也是用户代码，但当前指令是 CALL，不是 IMPORT_NAME。
      这属于库内部的导入（RUN-0059 在 HPC 上发现的误判），不按用户代码的规则判。
    帧 0 是这里，1 是 _guarded_import，2 是发起导入的帧。
    """
    global _IMPORT_NAME_OPCODE
    if _IMPORT_NAME_OPCODE is None:
        import dis

        _IMPORT_NAME_OPCODE = dis.opmap["IMPORT_NAME"]
    f = sys._getframe(depth)
    if f.f_code.co_filename != CODE_FILENAME:
        return False
    code, i = f.f_code.co_code, f.f_lasti
    return 0 <= i < len(code) and code[i] == _IMPORT_NAME_OPCODE


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """用户代码的 `__import__`（第 2 层），导入之后重新下毒。

    - 用户代码里的 import 语句：与 AST 检查同一套白名单，运行期再判一次，不合规就是违规。
    - 库内部的导入（用户代码调用的 C 函数里发起的）：不套用户规则，私有子模块（numpy._core.* 等）照常导入。
      第 3 层照样生效：学习型框架与下载器（torch、pooch 等）谁导入都只得到 ModuleNotFoundError，
      记进 blocked_imports，不算违规，库的"可选依赖"回退照常工作；这些模块永远不会被加载。
    """
    if _import_from_user_statement():
        if level:
            _violation("relative import")
        if not module_allowed(name):
            _violation(f"import {name}")
        for item in fromlist or ():
            if not from_import_allowed(name, item):
                _violation(f"from {name} import {item}")
    mod = __import__(name, globals, locals, fromlist, level)
    _poison_now()
    return mod


def _safe_builtins() -> dict:
    import builtins

    d = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    d["__import__"] = _guarded_import
    d["__build_class__"] = builtins.__build_class__
    return d


class _CappedWriter(io.TextIOBase):
    """只留前 limit 个字符的 stdout；超出部分只计数，不占内存。"""

    def __init__(self, limit: int) -> None:
        self._parts: list[str] = []
        self._n = 0
        self._limit = limit
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, s) -> int:
        s = str(s)
        room = self._limit - self._n
        if room > 0:
            self._parts.append(s[:room])
            self._n += min(len(s), room)
        if len(s) > max(room, 0):
            self.truncated = True
        return len(s)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _format_user_error(e: BaseException) -> str:
    lines = [fr.lineno for fr in traceback.extract_tb(e.__traceback__) if fr.filename == CODE_FILENAME]
    msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    if lines:
        msg += f" (line {lines[-1]})"
    return msg[:ERROR_LIMIT_CHARS]


def _memory_reader():
    """(名字, 读当前进程内存占用的函数，字节)。

    - macOS：phys_footprint（proc_pid_rusage，RUSAGE_INFO_V2，含被内存压缩器压缩的页，即活动监视器的"内存"）。
      不用 ru_maxrss：macOS 会压缩内容重复的页，3 GB 全 1 的数组常驻内存只显示约 0.9 GB。
    - 其他（Linux）：峰值常驻内存 ru_maxrss（KB）。Linux 上主要靠 RLIMIT_AS，看门狗只是兜底。
    """
    import resource

    def maxrss() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)

    if sys.platform == "darwin":
        try:
            import ctypes

            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            buf = ctypes.create_string_buffer(512)
            pid = os.getpid()

            def footprint() -> int:
                # struct rusage_info_v2：ri_uuid[16]，随后 7 个 uint64，第 8 个（偏移 72）是 ri_phys_footprint
                if libc.proc_pid_rusage(pid, 2, buf) != 0:
                    return maxrss()
                return int.from_bytes(buf.raw[72:80], "little")

            if footprint() > 0:
                return "phys_footprint", footprint
        except (OSError, AttributeError):
            pass
    return "maxrss", maxrss


def _watchdog(limit: int, read) -> None:
    while True:
        if read() > limit:
            os._exit(_EXIT_MEMORY)
        time.sleep(_WATCHDOG_INTERVAL_S)


def _child_main(workdir: str) -> None:
    env_keys = sorted(os.environ)  # 先记下启动时的环境变量，再做任何改动
    with open(os.path.join(workdir, _CONFIG_FILE), encoding="utf-8") as f:
        cfg = json.load(f)
    limit = int(cfg["memory_bytes"])
    guards = []
    import resource
    import threading

    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        guards.append("rlimit_as")
    except (ValueError, OSError):
        pass  # macOS 不支持 RLIMIT_AS
    how, read = _memory_reader()
    threading.Thread(target=_watchdog, args=(limit, read), daemon=True, name="w1-sandbox-watchdog").start()
    guards.append(f"watchdog:{how}")
    os.environ.update(_THREAD_ENV)

    result: dict[str, Any] = {"env_keys": env_keys, "memory_guard": guards, "has_mask": False}

    def _finish() -> None:
        result["violations"] = list(_VIOLATIONS)
        result["blocked_imports"] = list(dict.fromkeys(_BLOCKED_LOADS))
        with open(os.path.join(workdir, _RESULT_FILE), "w", encoding="utf-8") as f:
            json.dump(result, f)

    try:
        import contextlib
        import warnings

        import numpy as np

        np_load, np_savez = np.load, np.savez_compressed  # 下毒之前留下沙箱自己要用的两个函数
        with np_load(os.path.join(workdir, _INPUT_FILE), allow_pickle=False) as z:
            image = np.array(z["image"])
            masks = {k[len(MASK_KEY_PREFIX):]: np.array(z[k]) for k in z.files if k.startswith(MASK_KEY_PREFIX)}
        with open(os.path.join(workdir, _CODE_FILE), encoding="utf-8") as f:
            code = f.read()
        h, w = int(image.shape[0]), int(image.shape[1])
        sys.path.extend(cfg.get("test_sys_path", []))
        sys.meta_path.insert(0, _Finder())
        _poison_now()
        warnings.simplefilter("ignore")
        g = {"__builtins__": _safe_builtins(), "__name__": "__w1_sandbox__", "image": image, "masks": masks, "H": h, "W": w}
        for modname in cfg.get("test_preload", []):  # 仅测试：按库代码导入（经第 3 层，但不经用户白名单）
            import importlib

            g[modname.split(".")[-1]] = importlib.import_module(modname)
            _poison_now()
        out = _CappedWriter(int(cfg["stdout_chars"]))
        status, error = "ok", None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(code, CODE_FILENAME, "exec", dont_inherit=True), g)
        except SandboxViolation as e:
            status, error = "violation", str(e)
        except MemoryError:
            status, error = "memory", "MemoryError: memory limit exceeded"
        except BaseException as e:  # noqa: BLE001 —— 用户代码的任何异常都转成 error 返回
            status, error = "error", _format_user_error(e)
        if _VIOLATIONS and status != "violation":
            status, error = "violation", f"'{_VIOLATIONS[0]}' is not allowed in the W1 sandbox" + (f"; then: {error}" if error else "")
        result.update(stdout=out.getvalue(), stdout_truncated=out.truncated)
        if status == "ok":
            rm = g.get("result_mask")
            if rm is not None:
                arr = np.asarray(rm)
                if arr.shape != (h, w):
                    status, error = "error", f"result_mask must have shape (H, W) = ({h}, {w}); got {arr.shape}"
                elif arr.dtype == bool:
                    pass
                elif arr.dtype.kind in "iuf" and bool(np.all((arr == 0) | (arr == 1))):
                    arr = arr.astype(bool)
                else:
                    status, error = "error", f"result_mask must be boolean (or 0/1); got dtype {arr.dtype}"
                if status == "ok":
                    np_savez(os.path.join(workdir, _OUTPUT_FILE), result_mask=np.ascontiguousarray(arr, dtype=bool))
                    result["has_mask"] = True
        result.update(status=status, error=error)
    except MemoryError:
        result.update(status="memory", error="MemoryError: memory limit exceeded")
    _finish()


if __name__ == "__main__":
    _child_main(sys.argv[1])

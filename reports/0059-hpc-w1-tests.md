# RUN-0059 报告：HPC 环境里跑 W1 工作台、沙箱与多轮对话接口的测试

任务书 `prompts/server_hpc/0059-hpc-w1-tests.md`（origin b2d21f6a）。经 hpcmgt 上脱离会话的 `$VI/tmp/run0059.sh` 依次做，输出 `$VI/logs/run0059.txt`、`$VI/logs/pytest0059.txt`。没有碰 06 的 RUN-0056、07 的 RUN-0054；没有往 `$VI/repo` 里 pull。

## 工作树与环境

- 10-21 08:45:14Z squeue：157296、157297 都是 RUNNING。
- 临时工作树 `wt0059`：fetch rc=0；`worktree add --no-checkout --detach … origin/main` rc=0；`sparse-checkout set` rc=0；`read-tree -mu HEAD` rc=0。
- **commit：`cb16aa0c40428169db1350f43509eb72e6757029`**（当时的 origin/main）；status 空；包含 a5f1fc28：是。
- 环境：`$VI/env`（numpy 2.5.2、scipy 1.18.0、Pillow 12.3.0、torch 2.13.0+cu130、opencv-python-headless 4.10.0.84、scikit-image 0.24.0，RUN-0058）。
- 命令同任务书第 3 步（06 上经 srun，`CUDA_VISIBLE_DEVICES=` 空，`HF_HOME`、`HF_HUB_CACHE`、`HF_HUB_OFFLINE=1`、`TMPDIR`）。08:45:30Z – 08:49:41Z，**pytest_rc=1**。
- 之后删了 `$VI/tmp/pt_0059`；`git worktree remove $VI/wt/wt0059` rc=0；08:49:43Z squeue 两条 RUNNING。

## 结果：4 failed, 211 passed

- 没有因缺 HF 缓存而失败的（test_adapter_chat.py 全部通过）。
- 4 个失败都在 W1 的沙箱路径上，报错里都有 `violation: 'import numpy._core._methods' is not allowed in the W1 sandbox`；`test_results_are_deterministic` 的 SandboxResult 还记着 `imports=('pooch',)`、`env_keys=('LC_CTYPE',)`、`memory_guard=('rlimit_as', 'watchdog:maxrss')`。

## 失败的完整报错（pytest 输出原文）

```
=================================== FAILURES ===================================
______________________ test_run_python_through_workbench _______________________

scene = (array([[[111, 108, 117],
        [111, 103, 114],
        [125, 121, 101],
        ...,
        [102, 121, 118],
    ...lse, False, False, ..., False, False, False]], shape=(96, 128)), <src.w1.workbench.Workbench object at 0x145518114dd0>)

    def test_run_python_through_workbench(scene):
        _, disk, wb = scene
        wb._set_mask("d", disk)
>       r = _ok(wb.call("run_python", {"code": "import numpy as np\nprint(int(masks['d'].sum()), H, W)\nresult_mask = ~masks['d']", "out": "inv"}))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_w1_workbench.py:421: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

r = ToolResult(text="Error: violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, ..._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)")

    def _ok(r: ToolResult) -> ToolResult:
>       assert r.error is None, r.text
E       AssertionError: Error: violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)
E       assert "violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)" is None
E        +  where "violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)" = ToolResult(text="Error: violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, ..._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)").error

tests/test_w1_workbench.py:51: AssertionError
_______________________ test_workbench_is_deterministic ________________________

    def test_workbench_is_deterministic():
>       a = _run_sequence(perturb_rng=False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_w1_workbench.py:615: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

perturb_rng = False

    def _run_sequence(perturb_rng: bool):
        img, _ = _scene(seed=3)
        wb = Workbench(Image.fromarray(img))
        out = []
        for name, args in _SEQUENCE:
            if perturb_rng:  # 打乱 OpenCV 的全局随机数状态：GrabCut 每次调用前重设种子，结果不受影响
                cv2.setRNGSeed(12345)
                cv2.randu(np.zeros((10, 10), np.float32), 0, 1)
            r = wb.call(name, args)
>           assert r.error is None, (name, r.text)
E           AssertionError: ('run_python', "Error: violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)")
E           assert "violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)" is None
E            +  where "violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)" = ToolResult(text="Error: violation: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, ..._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)").error

tests/test_w1_workbench.py:604: AssertionError
_______________ test_library_internals_keep_their_own_file_reads _______________

image = array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0...     [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], shape=(24, 32, 3), dtype=uint8)
masks = {'a': array([[False, False, False, False, False, False, False, False, False,
        False, False, False, False, False...e,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False]])}

    def test_library_internals_keep_their_own_file_reads(image, masks):
        """skimage.morphology 导入时用 np.load 读自带的整数表：库内部调用不算违规（USER_ONLY 一类）。"""
        r = run_code("from skimage import morphology\nd = morphology.disk(3)\nprint(int(d.sum()))", image=image, masks=masks)
>       assert r.status == "ok", r.error
E       AssertionError: 'import numpy._core._methods' is not allowed in the W1 sandbox (no learned models, pretrained weights, files, network or processes)
E       assert 'violation' == 'ok'
E         
E         - ok
E         + violation

tests/test_w1_sandbox.py:191: AssertionError
________________________ test_results_are_deterministic ________________________

image = array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0...     [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], shape=(24, 32, 3), dtype=uint8)
masks = {'a': array([[False, False, False, False, False, False, False, False, False,
        False, False, False, False, False...e,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False]])}

    def test_results_are_deterministic(image, masks):
        code = (
            "import numpy as np\nfrom skimage.segmentation import slic\nlab = slic(image, n_segments=6, start_label=1)\n"
            "print(lab.max(), int(lab.sum()))\nresult_mask = lab == lab[10, 16]"
        )
        a = run_code(code, image=image, masks=masks)
        b = run_code(code, image=image, masks=masks)
>       assert a.ok and b.ok
E       AssertionError: assert (False)
E        +  where False = SandboxResult(status='violation', code='import numpy as np\nfrom skimage.segmentation import slic\nlab = slic(image, n...orts=('pooch',), env_keys=('LC_CTYPE',), memory_guard=('rlimit_as', 'watchdog:maxrss'), wall_time_s=0.6022018864750862).ok

tests/test_w1_sandbox.py:239: AssertionError
```

## pytest 输出最后 30 行原文

```

    def test_results_are_deterministic(image, masks):
        code = (
            "import numpy as np\nfrom skimage.segmentation import slic\nlab = slic(image, n_segments=6, start_label=1)\n"
            "print(lab.max(), int(lab.sum()))\nresult_mask = lab == lab[10, 16]"
        )
        a = run_code(code, image=image, masks=masks)
        b = run_code(code, image=image, masks=masks)
>       assert a.ok and b.ok
E       AssertionError: assert (False)
E        +  where False = SandboxResult(status='violation', code='import numpy as np\nfrom skimage.segmentation import slic\nlab = slic(image, n...orts=('pooch',), env_keys=('LC_CTYPE',), memory_guard=('rlimit_as', 'watchdog:maxrss'), wall_time_s=0.6022018864750862).ok

tests/test_w1_sandbox.py:239: AssertionError
=============================== warnings summary ===============================
tests/test_adapter_chat.py::test_turn_fields[natural]
  /home/mail/chenyuma/visint/env/lib/python3.12/site-packages/fla/utils/_device.py:100: UserWarning: Triton is not supported on current platform, roll back to CPU.
    _cpu_device_warning()

tests/test_adapter_chat.py: 14 warnings
  /home/mail/chenyuma/visint/env/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_w1_workbench.py::test_run_python_through_workbench - Assert...
FAILED tests/test_w1_workbench.py::test_workbench_is_deterministic - Assertio...
FAILED tests/test_w1_sandbox.py::test_library_internals_keep_their_own_file_reads
FAILED tests/test_w1_sandbox.py::test_results_are_deterministic - AssertionEr...
4 failed, 211 passed, 15 warnings in 241.96s (0:04:01)
srun: error: hpcgpu06: task 0: Exited with exit code 1
```

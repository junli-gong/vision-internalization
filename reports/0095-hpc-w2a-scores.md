# RUN-0095 报告：hpcmgt（CPU，只读），RUN-0072 的 W2-a 逐句交与并（D152）

任务书 `prompts/server_hpc/0095-hpc-w2a-scores.md`。不用 GPU，不进 srun。只贴数字，不解读。

## 运行

- 经 hpcmgt 上脱离会话的 `$VI/tmp/run0095.sh`（10-29 09:18:12Z 起）。前 squeue：157296 RUNNING 74-20:43:58 hpcgpu06；157297 RUNNING 76-18:16:07 hpcgpu07。
- fetch 第 1 次 rc=0；wt0095（稀疏）HEAD `399a4d5068d4a74e5e21e7dcbe6d3ddfe5f3975a`（含 40aff2c1），status 空；`extract_scores.py` sha256 `4816c65288253d5c17f605710e932b55ec7e826f09d71f9c505c120b9cbf605f`，与任务书相同。
- 命令（照任务书）：`nice -n 10 $VI/env/bin/python -I $VI/wt/wt0095/prompts/server_hpc/0095-hpc-w2a-scores/extract_scores.py --run $VI/repo/experiments/W2/0072_w2_test_hpcgpu06 --setting w2a --out $VI/repo/experiments/W2/0095_w2a_scores`
- 09:18:18Z 开始，09:18:21Z 结束，**rc=0**。

## stdout 原文

```
test_refcoco_unc_testA 1000 0.583182 0.583182 match
test_refcoco_unc_testB 1000 0.595192 0.595192 match
test_refcoco+_unc_testA 1000 0.563036 0.563036 match
test_refcoco+_unc_testB 1000 0.520797 0.520797 match
test_refcocog_umd_test 1000 0.575052 0.575052 match
```

## summary.json 的每个划分

| 划分 | n | source_sha256（predictions.jsonl，已与 manifest 核对） | out_sha256 | match |
|---|---|---|---|---|
| test_refcoco_unc_testA | 1000 | `f5a854ffd5c396d630f751a6e0450c7edb14570c2835f558f1f0c880d6f09c18` | `14e160b6ecb2d15ee8968022100972b8a600f6e14487902c187aae4d32948a55` | true |
| test_refcoco_unc_testB | 1000 | `3208817df93e41f9cf7f0643bb11fae1b20f26ba9f3cb4ca014e18615dc6291a` | `bda6578d4f580af2338dd3bd96cb90284a0ca795fb897bbb1b8f9d6994ce95fd` | true |
| test_refcoco+_unc_testA | 1000 | `b139454673d2743b92c4ed53c4ccb3fa453005e296c45ad233f6a9d4bf44c16c` | `0bf36cca8eff4c5042e56adcd6457e2d05f967307dd6d14441a50885bffbb14e` | true |
| test_refcoco+_unc_testB | 1000 | `5f748a79edcd3a39ad1a84d5cac30ea839348cda16b09cdd8d27a03333da9f6a` | `1aabb7e57320f16a25d9cc9a003c121945f6ad03ee39bbd8bd9636dff732612b` | true |
| test_refcocog_umd_test | 1000 | `5cc7c76a90ca922a986a3738de1a9e4fda38a0ae5447ef645eb9ad29218bc095` | `bd39b1f597f52efffe4a3aaac13792348c835ae1a7b53ad1843116a34513774a` | true |

- 输出 6 个文件：5 个 jsonl（91,592–93,203 字节）与 `summary.json`（1,739 字节）。

## 推送

- 用一次性同步（`ONCE=1 sync_push.sh … 0095 experiments/W2/0095_w2a_scores`），09:18:30Z ok。服务器上不在 `$VI/repo` 里手动 commit，所以提交信息是同步脚本的固定格式 `[results] 0095: sync <时间>`，不是任务书写的 `[results] 0095: W2-a per-sentence scores`。

## 收尾

- 后 squeue（09:18:30Z）：157296 RUNNING 74-20:44:16 hpcgpu06；157297 RUNNING 76-18:16:25 hpcgpu07。
- 没进 srun，没碰 RUN-0093（07）与 RUN-0094（06）。

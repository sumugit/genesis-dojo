"""Prompts for the reward writer (LLM) and the video evaluator (VLM)."""

import inspect
import json

from . import success, task_env

REWARD_SYSTEM = """あなたは四脚ロボットの強化学習の報酬設計者です。
日本語の指示に合う動作をロボットに覚えさせる報酬関数を、Pythonで書きます。
出力は必ず次の形式にしてください（それ以外の文章は不要）:
要点: <報酬の狙いを日本語1〜2文で>
```python
<モジュール全体のコード>
```"""

ENV_SPEC = """# 環境
- ロボット: Unitree Go2（四脚、関節12個）。シミュレータはGenesis、PPO（rsl_rl）、並列{num_envs}環境、{iterations}イテレーション。
- 制御周期 dt=0.02秒、1エピソード{episode_s}秒。毎エピソード、立った姿勢から開始（初期高さ0.42mから落ちて立つ）。
- 方策の出力は関節の目標角度の差分（既定の立ち姿勢からのずれ）。PD制御で追従する。
- 実測値: 立っているときの胴体中心の高さ ≈ 0.29m。足が接地しているときの足先中心の高さ ≈ 0.023m。
- 座標: ワールドz軸が上。初期状態でロボットの頭は+x方向を向く。
- 関節の順序: {joint_names}
- 既定の関節角度[rad]: {default_angles}

# 報酬関数が読める状態（s: TaskState、すべてtorch.Tensor、先頭の次元が環境数N）
```python
{task_state_src}
{helpers_src}
```
knee_pos は膝（前足なら肘）の関節、head_pos は鼻先の位置で、これらの高さ（z）が小さいときは膝や頭が地面に着いている。足先以外で体を支える抜け道（頭や肘で支える逆立ち）は報酬で避けること。
上の nose_up_angle / roll_angle は `from pai_loop.task_env import nose_up_angle, roll_angle` で使えます。

# 書くモジュールの仕様
- `PARAMS: dict` … 目標値・重み・終了条件の閾値など、数値はすべてここに置く（JSONにできる値のみ）。
- `compute(s, p) -> dict[str, Tensor]` … p は PARAMS。各項は重みを掛けた済みの (N,) テンソル。
  環境は全項の和に dt を掛けて報酬にする。項名はログに出るので短い英語にする。
- `failed(s, p) -> Tensor` … (N,) bool。Trueの環境はその場でエピソードを打ち切りリセットする（転倒など）。
- import してよいのは torch, math, pai_loop.task_env のみ。NaNが出ないこと（asin/sqrtの入力はclampする）。
- 正の報酬は「生きているだけで得」になるので、失敗時の打ち切りと組み合わせて使う。
- 目標から遠い初期状態でも勾配が消えないよう、指数型だけでなく線形の項も検討する。"""

SUCCESS_NOTE = """# 数値の成功基準（人が承認済み・変更不可）
学習した方策は、録画した動作の数値指標が、次の基準を満たすかどうかで機械的に判定されます。
系列の指標は、`window_s` の時間帯のサンプルのうち `min_fraction` 以上が範囲内であることが条件です。
この基準を満たす動作を学習させる報酬を設計してください。基準そのものを変えることはできません。
```yaml
{spec_text}
```"""

EVAL_SYSTEM = """あなたは四脚ロボットの動作を評価する審査員です。
学習済みの方策を動かした動画から抜き出したフレームと、数値の指標を見て、
日本語の指示どおりの動作になっているかを採点します。
報酬関数や学習の設定は見えません。先入観なく、見える動作だけで判断してください。
画像だけで姿勢を判断すると向きを取り違えやすいので、必ず数値の指標と突き合わせてください。
出力はJSONのみ（コードブロック不要）:
{"score": 0〜10の整数, "success": true/false, "observations": "見えたこと(日本語2〜3文)",
 "problems": ["問題点", ...], "next_fix": "動作・姿勢の観点での改善案(日本語。報酬コードには触れない)"}"""


def env_spec(cfg: dict) -> str:
    ecfg = cfg["env"]
    return ENV_SPEC.format(
        num_envs=cfg["num_envs"],
        iterations=cfg["max_iterations"],
        episode_s=ecfg["episode_length_s"],
        joint_names=", ".join(ecfg["joint_names"]),
        default_angles=json.dumps(ecfg["default_joint_angles"]),
        task_state_src=inspect.getsource(task_env.TaskState),
        helpers_src=inspect.getsource(task_env.nose_up_angle) + "\n" + inspect.getsource(task_env.roll_angle),
    )


def reward_request(instruction: str, cfg: dict, history: list[dict], success_spec_text: str | None = None,
                   hint_text: str | None = None, prior_text: str | None = None) -> str:
    parts = [f"# 日本語の指示\n「{instruction}」", env_spec(cfg)]
    if hint_text:
        parts.append("# 設計上のヒント（人が書いた助言。従う義務はないが、報酬設計の参考にすること）\n" + hint_text.strip())
    if success_spec_text:
        parts.append(SUCCESS_NOTE.format(spec_text=success_spec_text.strip()))
    if history:
        last = history[-1]
        parts.append("# 前回のループの結果")
        if last.get("reward_code"):
            parts.append(f"## 前回の報酬コード\n```python\n{last['reward_code']}\n```")
        if last.get("train_terms"):
            parts.append("## 学習終盤の各報酬項の平均（1エピソード・1秒あたり）\n" + json.dumps(last["train_terms"], ensure_ascii=False))
        if last.get("error"):
            parts.append(f"## エラー\n{last['error']}")
        if last.get("success_check"):
            parts.append("## 数値の成功判定で満たせなかった基準\n" + success.failure_summary(last["success_check"]))
        if last.get("evaluation"):
            parts.append("## 動画評価（審査員。報酬コードは見ていない）\n" + json.dumps(last["evaluation"], ensure_ascii=False))
        older = [h.get("evaluation", {}).get("next_fix") or h.get("error", "") for h in history[:-1]]
        if older:
            parts.append("## それ以前の修正指示（古い順）\n" + "\n".join(f"- {x}" for x in older))
        parts.append("前回の結果を踏まえて報酬関数を修正し、モジュール全体を書き直してください。")
    else:
        if prior_text:   # the first loop of a re-run: what the earlier experiment found (words and code only)
            parts.append(prior_text)
        parts.append("この指示と成功基準に合う報酬関数モジュールを書いてください。")
    return "\n\n".join(parts)


def eval_request(instruction: str, metrics: dict) -> str:
    """Blind scoring request: no reward code, no reward terms, no numeric verdict."""
    frames = ", ".join(f"{i + 1}枚目={f['t']:.2f}秒" for i, f in enumerate(metrics["frames"]))
    shown = {k: v for k, v in metrics.items() if k not in ("frames", "video", "record_wall_seconds")}
    return f"""# 日本語の指示
「{instruction}」

# 画像
{len(metrics['frames'])}枚のフレーム（{frames}）。カメラはロボットの位置だけを追従し、向きは世界座標に固定されている（初期状態では、ロボットの右側面を真横から見る）。
初期状態ではロボットの頭は画面の右を向いている。ロボットが回ると見える面が変わるので、向きは yaw_deg（開始時からの累積回転角）と合わせて判断すること。頭は黒い顔（センサー部）がある側の端で、胴体側面に「GO2」の文字がある。
頭と尻のどちらが高いかは、画像より nose_up_deg を優先すること。

# 数値の指標（{metrics['metrics_after_s']}秒以降の平均と、0.1秒ごとの時系列 series_0p1s）
- nose_up_deg: 体の前後の傾き。正なら頭が尻より高い。roll_deg: 横の傾き。
- height_m: 胴体中心の高さ（立つと約0.29m）。foot_z_*: 足先中心の高さ（接地で約0.023m）。FL/FR=前左/前右、RL/RR=後左/後右。
- yaw_deg: 開始時からの累積回転角（反時計回りが正。機体が垂直に近いと定義できず、値は据え置かれる）。yaw_total_deg: 終了時の累積回転。
- xy_max_dist_m: 開始位置からの最大水平距離。xy_path_length_m: 水平移動の総距離。xy_drift_m: 開始と終了の直線距離（円を描いて戻ると小さくなる）。
- front_feet_asym_m / rear_feet_asym_m: 前（後ろ）の左右の足の前後位置の差（左右対称なら0）。dof_vel_rms: 関節速度の大きさ。
- foot_z_* から導く: 前の2本の高い方（front_feet_z_max_m）、後ろの2本の低い方（rear_feet_z_min_m）。逆立ちなら、前足が低いまま・後ろ足が高く、nose_up_deg は大きな負になる。
- head_z: 鼻先の高さ。knee_z_FL / knee_z_FR / knee_z_RL / knee_z_RR: 各脚の膝（前足では肘）の関節の高さ。
  逆立ちの評価では、head_z が地面すれすれ（0.1m前後）、または前の膝が地面近く（0.05m以下）なら、足先ではなく頭や肘で体を支えている。
  それは本来の逆立ち（前足の足先だけで支える。本物では head_z が約0.29m、前の膝が約0.22m）とは違う姿勢なので、成功とは判定しないこと。
- x_m / y_m: 開始位置を原点とした水平位置（軌跡。円を描くかどうかは、これで判断する）。
- early_terminations: 途中で打ち切られた回数。
{json.dumps(shown, ensure_ascii=False)}"""

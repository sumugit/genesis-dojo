# pai_loop/sim — シミュレータの抽象化

タスク環境（[task_env.py](../task_env.py)）と報酬コードが、特定のシミュレータ（Genesis）に依存しないようにするための層。
Genesis が使えなくなった場合は、この層に別のバックエンドを足すだけで済む。

## ファイル

| ファイル | 役割 |
|---|---|
| `interface.py` | 契約の定義。`SimState`（状態のデータクラス）と `QuadrupedSim`（バックエンドが実装すべきメソッド） |
| `genesis_backend.py` | `GenesisGo2Sim`。Unitree Go2 を Genesis で動かす実装 |
| `__init__.py` | 工場関数 `make_sim(name, **kwargs)`。**タスク側コードが import してよいのはこれだけ** |

## 連携の仕方

```text
TaskEnv ── make_sim(cfg["sim"], ...) ──▶ GenesisGo2Sim（interface.QuadrupedSim を実装）
   │                                         └─ Genesis の Scene / Go2 / カメラ
   └─ sim.step(目標関節角) / sim.get_state() / sim.reset(...) / sim.error_mask()
      録画時のみ sim.start_recording() / render_frame() / stop_recording()
```

`make_sim` は、`name == "genesis"` のときだけ `genesis_backend` を**遅延 import** する。
Genesis を入れていない環境でも、報酬コードの検証（`reward_check.py`）などは動く。

## interface.py の契約

- クォータニオンは `(w, x, y, z)` 順。
- `SimState` の速度は**世界座標**。ボディ座標への変換は `TaskEnv` が行う。
- 関節の順序は、バックエンド生成時に渡す `joint_names` に従う。
- `SimState` の各フィールドの先頭次元は環境数 N:

| フィールド | 形 | 内容 |
|---|---|---|
| `base_pos` / `base_quat` | (N,3) / (N,4) | 胴体の位置・姿勢（世界座標） |
| `base_lin_vel` / `base_ang_vel` | (N,3) | 胴体の並進・角速度（世界座標） |
| `dof_pos` / `dof_vel` | (N,12) | 関節角・関節速度 |
| `feet_pos` | (N,4,3) | 足先の位置（世界座標）。順序は FL, FR, RL, RR |

- `QuadrupedSim` のメソッド:
  `step(joint_targets)`（PDの目標角を設定して1制御周期進める）／ `get_state()` ／
  `reset(envs_mask, base_pos, base_quat, dof_pos)`（指定した環境を、速度ゼロで指定姿勢へ移す）／
  `error_mask()`（物理ソルバが発散した環境）／
  録画用に `start_recording(path, fps)`・`render_frame(cam_pos, lookat)`・`stop_recording()`（カメラ付きで作った場合のみ有効）。

## genesis_backend.py の実装上のポイント

- シーンは地面（plane）と Go2（URDF）。`dt`（既定0.02秒）、`substeps`（既定2）、自己衝突なし。
  描画する環境は先頭の1つだけ（`rendered_envs_idx=[0]`）。
- 関節は PD 制御（`kp` / `kd` を全関節に設定）。`control_dofs_position` で目標角を与える。
- **足先位置は直接取れない。** Genesis が足先のリンクをふくらはぎ（calf）に統合するため、
  calf リンクの位置・姿勢と、固定のオフセット `(0, 0, -0.213)` から計算する。
- `reset` は、自由関節が6自由度・7要素であることを考慮して `qpos = [位置(3), 姿勢(4), 関節角]` を組み立てて設定する。
- カメラは `camera` 引数が渡されたときだけ作る（学習時は不要で、録画時のみ）。
- Genesis のバージョンによって録画APIが異なるため、バージョンは固定している（[VERSIONS.md](../../VERSIONS.md)）。

## 新しいバックエンドを足すには

1. `QuadrupedSim` の全メソッドを持つクラスを、新しいファイルに実装する
2. `__init__.py` の `make_sim` に分岐を追加する
3. 設定ファイルの `sim:` にその名前を指定する

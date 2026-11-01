# configs — 実験設定（YAML）

1つの実験（1回の学習）を再現するための**完全な設定**。`scripts/train.py` と `scripts/record.py` が読む。

## ファイル

| ファイル | 内容 |
|---|---|
| `base.yaml` | 自動ループ共通の設定（環境・PPO・録画）。`scripts/loop.py` が読み、各ループの設定の土台にする |
| `exp01_sit.yaml` | **人手版**の「お座り」。コメント付きで手書きされ、報酬は `pai_loop/rewards/sit.py` |
| `tasks/<芸>.success.yaml` | **数値の成功条件**（芸ごと）。人が承認（`approved: true`）して凍結する。`sit`（お座り）と `spin`（その場で回る）は草案（`approved: false`） |

## 設定の構造

| キー | 内容 |
|---|---|
| `exp_name` / `instruction` | 実験名（`runs/<exp_name>/` の名前になる）と、日本語の指示 |
| `seed` | 乱数シード（`gs.init` に渡す）。再現性のため固定する |
| `sim` | シミュレータのバックエンド名（`pai_loop/sim` の `make_sim` に渡す。現在は `genesis`） |
| `num_envs` / `max_iterations` | 並列環境数（4096）と PPO のイテレーション数（300） |
| `env` | ロボットと環境: 関節名の順序、既定の関節角度、PDゲイン（`kp`/`kd`）、初期の位置・姿勢、エピソード長（秒）、`action_scale`、`clip_actions` |
| `obs_scales` | 観測のスケール（角速度・関節角・関節速度） |
| `train` | rsl_rl の `OnPolicyRunner` の設定（PPO、MLP 512-256-128、`num_steps_per_env`、保存間隔、TensorBoard） |
| `record` | 録画: 秒数、解像度、画角、カメラ位置（ロボットからの相対）、指標を平均し始める時刻 `settle_s`。任意項目: `lookat_z`（注視点の高さ、既定0.25）、`frame_times`（キーフレームを切り出す時刻のリスト。既定は 0, 0.5, 1, 2, 4 秒と終端） |
| `reward.module` / `reward.params` | 報酬モジュールの import 名と、そのパラメータ（重み・目標値・終了条件） |

## 生成の仕組み（自動ループ）

```text
base.yaml ──┐
            ├─ loop.py が合成 ─▶ configs/<exp>_L<i>.yaml
LLMの報酬     ┘   + exp_name, instruction,
(PARAMS)         + reward.module = pai_loop.rewards.generated.<exp>_L<i>
                 + reward.params = そのモジュールの PARAMS
```

- 生成された設定は、`base.yaml` の内容に `exp_name` / `instruction` / `reward` を足したもの。
- **`reward.params` が、学習時に実際に使われる値。** 報酬モジュール内の `PARAMS` は書き出しの元にすぎず、
  後から `.py` を書き換えても、この YAML を変えない限り学習結果は変わらない。
- `scripts/train.py` は、この YAML を `runs/<exp_name>/config.yaml` にコピーする。
  `scripts/record.py` は、そのコピーを読んで録画する（学習時と同じ設定を使うため）。

## 注意

- `exp01_sit.yaml` のカメラは側前方（`cam_offset: [0.8, -1.8, 0.5]`）。自動ループ（`base.yaml`）は、評価役に渡す画像のために、
  初期状態でロボットの右側面を**真横**から見る位置（`[0.0, -1.4, 0.35]`）にカメラを置く。カメラは位置だけを追従し、向きは世界座標に固定される（回る芸では見える面が変わる）。評価用プロンプトはこの前提で書かれている。
- 同じ `exp_name` で学習し直すと `runs/<exp_name>/` が既にあるため `train.py` は終了する。実験名を変えるか、削除する。

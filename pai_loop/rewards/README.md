# pai_loop/rewards — 報酬モジュール

「1つの芸 = 1つの報酬モジュール」。**LLMが書き換える唯一の部分**であり、環境・学習・録画のコードは変えない。
環境は設定ファイルの `reward.module`（例: `pai_loop.rewards.sit`）から、これらを動的に読み込む。

## ファイル

| ファイル / ディレクトリ | 内容 |
|---|---|
| `sit.py` | **手書き**の「お座り」報酬。LLM版の比較基準。設定は `configs/exp01_sit.yaml` |
| `generated/` | **LLMが生成した**報酬の置き場。[詳細](generated/README.md) |
| `__init__.py` | 説明のみ |

## 報酬モジュールの契約

どの報酬モジュールも、次の3つを定義する。契約は `task_env.py` の docstring と `prompts.py` の仕様文に書かれ、
`reward_check.load_and_check` が検証する。

```python
PARAMS: dict                                    # 目標値・重み・終了条件の閾値。JSONにできる値のみ
def compute(s: TaskState, p: dict) -> dict[str, Tensor]   # 重み付き済みの報酬項。各項は形 (N,)
def failed(s: TaskState, p: dict) -> Tensor               # 形 (N,) の bool。Trueなら打ち切ってリセット
```

- `s`（`TaskState`）は環境の全状態で、先頭次元が環境数 N。定義は [task_env.py](../task_env.py)。
- `p` は `PARAMS`。ただし**学習時に使われるのは設定ファイルの `reward.params`**（`loop.py` が `PARAMS` を書き出したもの）。
  モジュール内の `PARAMS` を後から書き換えても、既存の設定ファイルの学習結果は変わらない。
- 環境は全項の合計に `dt` を掛けたものを報酬とする。項名は短い英語にする（学習ログの `Mean episode rew_<項名>` に出る）。
- import してよいのは `torch`、`math`、`pai_loop.task_env` のみ。`asin` や `sqrt` の入力は `clamp` して、NaN を出さない。
- 正の報酬は「生きているだけで得」になるので、失敗による打ち切り（`failed`）と組み合わせて使う。

## sit.py（手書きのお座り）の中身

目標姿勢: 頭を約40°上げ、胴体を約0.22mに下げ、4本の足を接地させ、その場から動かない。

| 項名 | 内容 |
|---|---|
| `pitch` | 前後の傾きが目標に近いほど高い。**線形**（指数にすると、立っている状態で勾配が消えるため） |
| `height` | 胴体の高さが目標に近いほど高い（指数型） |
| `feet_lift` | 足が接地の高さより浮いた量への罰 |
| `roll` | 横の傾きへの罰 |
| `drift` | 横移動・旋回への罰 |
| `action_rate` | 行動の急変への罰 |
| `dof_vel` | 関節速度への罰 |

終了条件（`failed`）: 頭が上がりすぎ／下がりすぎ、横に傾きすぎ、胴体が低すぎる。

`sit.py` は `nose_up_angle(s)` と `roll_angle(s)` を**自前で定義**している（`TaskState` を受け取る）。
`task_env.py` 側のヘルパーは `gravity_b` を受け取る別の関数で、LLM版はこちらを import して使う。

## 新しい芸を足すには

1. 手書きなら、このディレクトリに契約どおりのモジュールを作り、設定ファイルの `reward.module` に指定する
2. LLMに書かせるなら、`scripts/loop.py` を実行する（生成物は `generated/` に入る）

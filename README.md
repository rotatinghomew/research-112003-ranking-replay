# 公益家教入口的排序策略离线回放

各表为本地合成业务记录，身份与机构代号不对应真实对象。

## 资料格式

资料为 UTF-8 编码、首行为列名的 CSV。空字段表示没有记录，不表示零。连续日号采用同一份内部日历。

| 文件 | 列 | 登记口径 |
| --- | --- | --- |
| `requests.csv` | `id`、`user`、`day`、`entry`、`slot`、`region`、`slate`、`probability`、`logger` | probability 存放记录策略选中整屏顺序的联合概率。候选目录分地区，历史流程可能留下不符合新规则的曝光。 |
| `candidates.csv` | `request`、`card`、`provider`、`catalog`、`region` | 曝光当时的候选及服务点，region=all 不限地区。 |
| `support.csv` | `request`、`slate`、`probability` | 记录策略在当次请求下可产生的全部整屏次序及各自概率。 |
| `outcomes.csv` | `receipt`、`request`、`card`、`click_day`、`known`、`state`、`booking` | 点击和办理记录；booking 为接续标识，只有同一服务点和同一次接续才能去重。 |
| `proposals.csv` | `plan`、`entry`、`region`、`from_day`、`to_day`、`slate`、`horizon`、`min_support`、`max_weight`、`holdout`、`traffic`、`asof` | 拟比较顺序及上线候选范围；预算 traffic 为小流量请求量，holdout 为至少保留的对照比例。 |
| `capacity.csv` | `provider`、`slot`、`known`、`limit` | 服务点每时段可兑现的办结额度，可在生效前改版。 |
| `running.csv` | `approval`、`entry`、`provider`、`slot`、`active`、`committed`、`rule` | 已批准的流量兑现占用和实际线上规则。 |

## 检查资料

使用 Python 3.11 和标准库。在项目根目录运行：

```sh
python3 -m unittest discover -s tests -v
```

这项检查核对资料格式与引用，不生成业务结论。

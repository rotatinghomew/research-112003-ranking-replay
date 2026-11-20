"""公益家教入口排序策略的离线回放。

只使用标准库。全部比较都落在「曝光当时能产生的整屏次序」上：

* 候选与服务点来自 ``candidates`` 的当次快照。后来把卡片迁到别的服务点，
  不拿今天的目录改写旧曝光；需要上线时再单独核对当前目录的额度。
* 次序必须在 ``support`` 中作为一条联合概率出现，权重取该联合概率的倒数。
  各位置边际概率相乘会把一条从未产生过的搭配算成高置信方案。
* 同一请求、同一服务点的重复回执按 ``(provider, booking)`` 去重；不同请求
  复用同一 booking 不算重复，那是同一服务点被多次接续。
* 观察窗口未结、又缺办结证据的领取，下界记 0、上界记 1，并按用户整体
  重抽样给出区间。区间同时覆盖正负时，极端权重和未成熟回执足以改变建议。

用法::

    python3 -m analysis.replay          # 逐条候选的比较结论
    python3 -m analysis.replay --json   # 机器可读
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'

BLOCK_CAP = 'support_weight_capped'
BLOCK_MIN = 'support_below_minimum'
BLOCK_NEW = 'order_never_producible'
BLOCK_NONE = 'no_eligible_requests'
BLOCK_EXPOSURE = 'no_exposed_orders'
BLOCK_MIXED = 'mixed_propensity'


def read(table: str) -> list[dict[str, str]]:
    with (DATA / f'{table}.csv').open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


@dataclass
class Snapshot:
    """一次离线回放所见的全部资料，均为文件里的原样记录。"""

    requests: dict[str, dict[str, str]] = field(default_factory=dict)
    candidates: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    support: dict[str, dict[str, float]] = field(default_factory=dict)
    outcomes: list[dict[str, str]] = field(default_factory=list)
    proposals: list[dict[str, str]] = field(default_factory=list)
    capacity: list[dict[str, str]] = field(default_factory=list)
    running: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls) -> 'Snapshot':
        snap = cls()
        snap.requests = {row['id']: row for row in read('requests')}
        for row in read('candidates'):
            snap.candidates.setdefault(row['request'], {})[row['card']] = row
        for row in read('support'):
            snap.support.setdefault(row['request'], {})[row['slate']] = float(row['probability'])
        snap.outcomes = read('outcomes')
        snap.proposals = read('proposals')
        snap.capacity = read('capacity')
        snap.running = read('running')
        return snap

    # ---------------------------------------------------------- 目录 ----
    def card_available(self, request: str, card: str) -> bool:
        """这张卡片在当次请求中确实能展示，并且地区可用。"""
        row = self.candidates.get(request, {}).get(card)
        if row is None:
            return False
        return row['region'] in ('all', self.requests[request]['region'])

    def slate_producible(self, request: str, slate: str) -> bool:
        """这个整屏次序在当次请求下能产生。"""
        if slate not in self.support.get(request, {}):
            return False
        return all(self.card_available(request, card) for card in slate.split('|'))

    def marginal_product(self, request: str, slate: str) -> float:
        """各位置边际概率相乘 —— 只用于说明为什么不能这样算。"""
        sequence = slate.split('|')
        table = self.support.get(request, {})
        product = 1.0
        for position, card in enumerate(sequence):
            marginal = sum(
                p for option, p in table.items()
                if len(option.split('|')) == len(sequence) and option.split('|')[position] == card
            )
            product *= marginal
        return product

    def order_providers(self, request: str, slate: str) -> list[str]:
        """该请求下这个整屏依次涉及的服务点（同一点可能出现两次）。"""
        return [self.candidates[request][card]['provider'] for card in slate.split('|')]

    def unique_providers(self, request: str, slate: str) -> list[str]:
        return sorted(set(self.order_providers(request, slate)))

    def card_catalog(self, request: str, card: str) -> str:
        return self.candidates[request][card]['catalog']


# ---------------------------------------------------------------- 回执 ----

@dataclass
class ReceiptView:
    """某个截止日、某个观察窗口下，每个请求能看到的接续情况。"""

    asof: int
    horizon: int
    states: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def at(cls, snap: Snapshot, asof: int, horizon: int) -> 'ReceiptView':
        by_request: dict[str, list[dict]] = defaultdict(list)
        for row in snap.outcomes:
            if int(row['known']) <= asof:
                by_request[row['request']].append(row)

        view = cls(asof=asof, horizon=horizon)
        for request, meta in snap.requests.items():
            deduped: dict[tuple[str, str], dict] = {}
            duplicates = 0
            for row in by_request.get(request, []):
                provider = snap.candidates[row['request']][row['card']]['provider']
                key = (provider, row['booking'])
                previous = deduped.get(key)
                if previous is not None:
                    duplicates += 1
                    if row['state'] == 'completed' and previous['state'] != 'completed':
                        deduped[key] = row
                    continue
                deduped[key] = row

            services = list(deduped.values())
            completed = any(row['state'] == 'completed' for row in services)
            clicked = bool(services)
            open_window = int(meta['day']) + horizon > asof
            view.states[request] = {
                'completed': completed,
                'clicked': clicked,
                'open': open_window,
                'unresolved': clicked and not completed,
                'services': services,
                'duplicate_receipts': duplicates,
            }
        return view

    def closed(self, request: str) -> int:
        """保守下界：有办结证据才是 1。缺证不记为成功，也不把未结当 0。"""
        return 1 if self.states[request]['completed'] else 0

    def open_upper(self, request: str) -> int:
        """乐观上界：窗口未结、尚未办结，仍可能办成，记 1。"""
        state = self.states[request]
        if state['completed']:
            return 1
        return 1 if state['open'] else 0


# ---------------------------------------------------------------- 比较 ----

@dataclass
class Comparison:
    plan: str
    entry: str
    region: str
    slate: str
    slot: str
    eligible: list[str]
    producible: list[str]
    usable: list[str]
    exposed: list[str]
    probability: float | None
    min_support: float
    weight_cap: float
    asof: int
    horizon: int
    receipts: ReceiptView
    snap: Snapshot
    blockers: list[str] = field(default_factory=list)

    # -- 点估计 --
    def rate(self, ids: list[str], upper: bool) -> float:
        getter = self.receipts.open_upper if upper else self.receipts.closed
        return sum(getter(r) for r in ids) / len(ids)

    def estimate(self, upper: bool) -> dict:
        if not self.usable:
            return {'baseline': None, 'target': None, 'gain': None,
                    'sum_weight': 0.0, 'capped': 0}
        baseline = self.rate(self.usable, upper)
        if not self.exposed or not self.probability:
            return {'baseline': baseline, 'target': None, 'gain': None,
                    'sum_weight': 0.0, 'capped': 0}
        getter = self.receipts.open_upper if upper else self.receipts.closed
        total = 0.0
        sum_weight = 0.0
        capped = 0
        for request in self.exposed:
            weight = 1.0 / self.probability
            if weight > self.weight_cap:
                weight = self.weight_cap
                capped += 1
            total += weight * getter(request)
            sum_weight += weight
        # 逆概率加权的分母是全部可用人群，而不是曝光人数：
        # 目标策略若对所有人生效，期望总办结 = (1/N) Σ曝光 (1/p) y。
        target = total / len(self.usable)
        return {'baseline': baseline, 'target': target, 'gain': target - baseline,
                'sum_weight': sum_weight, 'capped': capped}

    def interval(self, upper: bool, reps: int = 4000,
                 seed: int = 20261120) -> tuple[float, float] | None:
        """按用户整体重抽样的百分位区间，处理同一用户的重复访问。"""
        if not self.usable or not self.exposed or not self.probability:
            return None
        by_user: dict[str, list[str]] = defaultdict(list)
        for request in self.usable:
            by_user[self.snap.requests[request]['user']].append(request)
        users = sorted(by_user)
        rng = random.Random(seed)
        getter = self.receipts.open_upper if upper else self.receipts.closed
        exposed = set(self.exposed)
        weight = 1.0 / self.probability
        if weight > self.weight_cap:
            weight = self.weight_cap
        draws: list[float] = []
        for _ in range(reps):
            sample: list[str] = []
            for _ in users:
                sample.extend(by_user[rng.choice(users)])
            baseline = sum(getter(r) for r in sample) / len(sample)
            # 与点估计同一口径：逆概率加权落在全人群的期望总办结上。
            target = sum(weight * getter(r) for r in sample if r in exposed) / len(sample)
            draws.append(target - baseline)
        if not draws:
            return None
        draws.sort()

        def percentile(value: float) -> float:
            rank = (len(draws) - 1) * value
            low = math.floor(rank)
            high = math.ceil(rank)
            if low == high:
                return draws[low]
            return draws[low] * (high - rank) + draws[high] * (rank - low)

        return percentile(0.025), percentile(0.975)

    # -- 按候选目录版本复算 --
    def by_catalog(self, upper: bool) -> dict[str, dict]:
        """按目录版本拆开：换版前后不是同一件事，不能合并报一个增益。

        与池化口径一致，目标侧用联合概率的逆概率加权，分母是该版本的可用人群。
        """
        groups: dict[str, list[str]] = defaultdict(list)
        exposed_groups: dict[str, list[str]] = defaultdict(list)
        first = self.slate.split('|')[0]
        for request in self.usable:
            groups[self.snap.card_catalog(request, first)].append(request)
        for request in self.exposed:
            exposed_groups[self.snap.card_catalog(request, first)].append(request)
        getter = self.receipts.open_upper if upper else self.receipts.closed
        result = {}
        for version, ids in sorted(groups.items()):
            exposed = exposed_groups.get(version, [])
            baseline = self.rate(ids, upper)
            if exposed and self.probability:
                weight = min(1.0 / self.probability, self.weight_cap)
                target = sum(weight * getter(r) for r in exposed) / len(ids)
                gain = target - baseline
            else:
                target = gain = None
            result[version] = {
                'usable': len(ids),
                'exposed': len(exposed),
                'baseline': baseline,
                'target': target,
                'gain': gain,
                'open_usable': sum(1 for r in ids if self.receipts.states[r]['open']),
                'open_exposed': sum(1 for r in exposed if self.receipts.states[r]['open']),
                'completions_exposed': sum(getter(r) for r in exposed),
            }
        return result

    def same_provider_orders(self) -> int:
        """一张屏上两张卡片落到同一服务点的请求数：两次点击不等于两个可用服务。"""
        return sum(
            1 for request in self.usable
            if len(self.snap.unique_providers(request, self.slate)) < len(self.slate.split('|'))
        )

    def exposure_services(self) -> dict:
        receipts = sum(
            len(self.receipts.states[r]['services']) for r in self.exposed
        )
        unique = sum(
            len({(s['card'], s['booking']) for s in self.receipts.states[r]['services']})
            for r in self.exposed
        )
        return {'click_receipts': receipts,
                'unique_services': unique,
                'duplicate_receipts': receipts - unique}

    def completion_share(self) -> dict[str, float]:
        """按曝光当时目录，办结落在各卡片上的份额。

        只用于把预测兑现分散到服务点，避免把全部需求押在同一家；
        卡片后来迁移服务点的，会在 ``mapping_transfer`` 里单独标出。
        """
        counts: Counter = Counter()
        for request in self.exposed:
            for service in self.receipts.states[request]['services']:
                if service['state'] == 'completed':
                    counts[service['card']] += 1
        total = sum(counts.values())
        if not total:
            return {}
        return {card: count / total for card, count in counts.items()}


def build(snap: Snapshot, proposal: dict[str, str]) -> Comparison:
    entry = proposal['entry']
    region = proposal['region']
    slate = proposal['slate']
    sequence = slate.split('|')
    from_day = int(proposal['from_day'])
    to_day = int(proposal['to_day'])
    asof = int(proposal['asof'])
    horizon = int(proposal['horizon'])
    min_support = float(proposal['min_support'])
    weight_cap = float(proposal['max_weight'])

    eligible = [
        rid for rid, row in snap.requests.items()
        if row['entry'] == entry
        and from_day <= int(row['day']) <= to_day
        and (region == 'all' or row['region'] == region)
        and all(snap.card_available(rid, card) for card in sequence)
    ]
    producible = [rid for rid in eligible if snap.slate_producible(rid, slate)]
    usable = [rid for rid in producible if snap.support[rid][slate] >= min_support]
    exposed = [rid for rid in usable if snap.requests[rid]['slate'] == slate]
    probabilities = sorted({snap.support[rid][slate] for rid in usable})
    probability = probabilities[0] if len(probabilities) == 1 else None
    slots = sorted({snap.requests[rid]['slot'] for rid in eligible})

    comparison = Comparison(
        plan=proposal['plan'], entry=entry, region=region, slate=slate,
        slot=slots[0] if len(slots) == 1 else '/'.join(slots),
        eligible=eligible, producible=producible, usable=usable, exposed=exposed,
        probability=probability, min_support=min_support, weight_cap=weight_cap,
        asof=asof, horizon=horizon,
        receipts=ReceiptView.at(snap, asof, horizon),
        snap=snap,
    )

    if not eligible:
        comparison.blockers.append(BLOCK_NONE)
    if not producible:
        comparison.blockers.append(BLOCK_NEW)
    if producible and not usable:
        comparison.blockers.append(BLOCK_MIN)
    if len(probabilities) > 1:
        comparison.blockers.append(BLOCK_MIXED)
    if not exposed:
        comparison.blockers.append(BLOCK_EXPOSURE)
    if comparison.probability and 1 / comparison.probability > weight_cap:
        comparison.blockers.append(BLOCK_CAP)
    return comparison


# ------------------------------------------------------------ 兑现额度 ----

def effective_capacity(snap: Snapshot) -> dict[tuple[str, str], dict]:
    """服务点每时段额度取文件里最新的一版。

    ``capacity.known`` 是该额度版本的截止日；同一 provider/slot 出现多版时取
    ``known`` 最大的一版。q1/evening 在日 24 记录 limit=75，评审后的日 31 版
    降为 limit=30，审批必须按 30。

    ``running.active`` 是已批准流量的生效日，``running.committed`` 才是已批准的
    兑现占用。剩余额度 = limit − Σcommitted，已有批准流量优先于新入口。
    """
    rows: dict[tuple[str, str], dict] = {}
    revisions: Counter = Counter()
    for row in snap.capacity:
        key = (row['provider'], row['slot'])
        revisions[key] += 1
        previous = rows.get(key)
        if previous is None or int(row['known']) > int(previous['known']):
            rows[key] = row
    committed = Counter()
    active_days: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in snap.running:
        key = (row['provider'], row['slot'])
        committed[key] += int(row['committed'])
        active_days[key].append(int(row['active']))
    result = {}
    for key, row in rows.items():
        limit = int(row['limit'])
        result[key] = {
            'known': int(row['known']),
            'committed': committed[key],
            'active_from': sorted(active_days.get(key, [])),
            'limit': limit,
            'used': committed[key],
            'headroom': limit - committed[key],
            'revised': revisions[key] > 1,
        }
    return result


def latest_mapping(snap: Snapshot, entry: str) -> dict[str, dict[str, str]]:
    """该入口最近一天的卡片 -> 服务点/目录，用于上线时的额度核对。

    仅用于「现在上线会用哪个服务点」；历史曝光仍按曝光当时的快照归属。
    """
    days = [int(row['day']) for row in snap.requests.values() if row['entry'] == entry]
    latest_day = max(days)
    mapping: dict[str, dict[str, str]] = {}
    for rid, row in snap.requests.items():
        if row['entry'] != entry or int(row['day']) != latest_day:
            continue
        for card, meta in snap.candidates[rid].items():
            mapping[card] = {'provider': meta['provider'], 'catalog': meta['catalog']}
    return mapping


def mapping_transfer(snap: Snapshot, comparison: Comparison) -> dict:
    """曝光时的卡片-服务点归属是否仍等于当前目录。"""
    current = latest_mapping(snap, comparison.entry)
    changes = {}
    for request in comparison.exposed:
        for card in comparison.slate.split('|'):
            old = snap.candidates[request][card]['provider']
            new = current.get(card, {}).get('provider')
            if new is not None and old != new:
                changes[card] = {'exposure': old, 'current': new}
    return {
        'current': current,
        'changed_cards': changes,
        'transferable': not changes,
    }


def capacity_check(snap: Snapshot, comparison: Comparison,
                   traffic: int, holdout: float,
                   completion_rate: float | None = None) -> dict:
    """按入口、时段、服务点核对额度，不用一个入口的高估透支另一个入口。

    ``traffic`` 是小流量请求量，不能直接当办结数。这里同时给出：
    - ``request_ceiling``：请求量全部压在同一点的最坏屏，只用于发现硬冲突；
    - ``expected_redemptions``：按离线目标办结率折算的兑现占用，用于审批。
    ``holdout`` 是对照保留份额，小流量中必须有这么多继续走旧规则。
    """
    capacities = effective_capacity(snap)
    control = math.ceil(traffic * holdout)
    treatment = traffic - control
    # 没有证据时不假装知道兑现率，只保留「预测兑现未知」；不把缺证当成 0 占用。
    expected = (math.ceil(treatment * completion_rate)
                if completion_rate is not None else None)
    mapping = latest_mapping(snap, comparison.entry)
    share = comparison.completion_share()
    providers = sorted({mapping[card]['provider'] for card in comparison.slate.split('|')
                        if card in mapping})
    rows = []
    for provider in providers:
        cards = [card for card in comparison.slate.split('|')
                 if card in mapping and mapping[card]['provider'] == provider]
        provider_share = sum(share.get(card, 0.0) for card in cards)
        provider_expected = (math.ceil(expected * provider_share)
                             if expected is not None else None)
        key = (provider, comparison.slot)
        row = capacities.get(key)
        if row is None:
            rows.append({'provider': provider, 'slot': comparison.slot,
                         'status': 'no_quota_record', 'request_ceiling': traffic,
                         'treatment': treatment, 'control': control,
                         'expected_redemptions': provider_expected,
                         'completion_share': provider_share})
            continue
        if provider_expected is None:
            # 没有可用的兑现估计：只能标注「需求未知」，不能据此判定够或不够。
            status = 'demand_unknown'
        elif row['headroom'] <= 0:
            status = 'exhausted'
        elif provider_expected > row['headroom']:
            status = 'insufficient'
        else:
            status = 'ok'
        rows.append({
            'provider': provider, 'slot': comparison.slot,
            'limit': row['limit'], 'known': row['known'],
            'committed': row['committed'],
            'headroom': row['headroom'], 'request_ceiling': traffic,
            'treatment': treatment, 'control': control,
            'expected_redemptions': provider_expected,
            'completion_share': provider_share,
            # 即便办结率未知，请求量本身也远超余量时标为「余量吃紧」；
            # 这不是预测，只是提醒不能把整段流量都压到该点。
            'thin': row['headroom'] < treatment,
            'status': status, 'revised': row['revised'],
        })
    return {'control': control, 'treatment': treatment,
            'completion_rate': completion_rate, 'providers': rows}


def shared_quota(snap: Snapshot, plans: list[dict]) -> dict[tuple[str, str], dict]:
    """同一服务点/时段被多个入口同时申请时，额度不能重复使用。"""
    capacities = effective_capacity(snap)
    by_key: dict[tuple[str, str], dict] = defaultdict(
        lambda: {'entries': [], 'running': [], 'demand': 0, 'expected': 0}
    )
    for plan in plans:
        for row in plan['capacity']['providers']:
            key = (row['provider'], row['slot'])
            if row['status'] == 'no_quota_record':
                continue
            expected = row.get('expected_redemptions') or 0
            by_key[key]['entries'].append({
                'plan': plan['plan'], 'entry': plan['entry'],
                'request_ceiling': row['request_ceiling'],
                'expected_redemptions': row.get('expected_redemptions'),
            })
            by_key[key]['demand'] += row['request_ceiling']
            by_key[key]['expected'] += expected
    for row in snap.running:
        key = (row['provider'], row['slot'])
        by_key[key]['running'].append({
            'approval': row['approval'], 'entry': row['entry'],
            'active_from': int(row['active']),
            'committed': int(row['committed']),
        })
    for key, item in by_key.items():
        cap = capacities.get(key, {})
        item['limit'] = cap.get('limit')
        item['used'] = cap.get('used')
        item['headroom'] = cap.get('headroom')
        item['overcommitted'] = item['expected'] > max(cap.get('headroom', 0), 0)
    return dict(by_key)


# ---------------------------------------------------------------- 汇总 ----

def verdict(comparison: Comparison) -> tuple[str, str]:
    """返回 (类别, 说明)。类别只描述证据状态，不代替运营决策。"""
    if BLOCK_NEW in comparison.blockers:
        return 'no_counterfactual', (
            '目标整屏次序在窗口内从未进入 support，属于没有反事实数据的入口，'
            '不能比较；也不能用两张卡片的边际点击率拼出置信增益。'
        )
    if BLOCK_MIN in comparison.blockers:
        return 'no_counterfactual', (
            '次序虽然出现过，但支持概率低于 min_support，无法做稳定的反事实比较；'
            '支持不足不等于没人需要，不应记为 0。'
        )
    if BLOCK_MIXED in comparison.blockers:
        return 'mixed_support', '同一窗口内支持概率不一致，不能当作单一策略比较。'
    if BLOCK_CAP in comparison.blockers:
        return 'inexact', (
            '该次序的支持概率过低，逆概率权重超过 max_weight 被截断；'
            '截断本身会把目标估计抬离 0–1 范围，不能作为发布依据，'
            '只能说明该入口需要更多曝光。'
        )
    if BLOCK_EXPOSURE in comparison.blockers:
        return 'no_counterfactual', '窗口内没有该次序的实际曝光，无法给出比较。'

    lower = comparison.estimate(False)
    upper = comparison.estimate(True)
    interval_lower = comparison.interval(False)
    interval_upper = comparison.interval(True)
    by_catalog = comparison.by_catalog(False)
    gains = [item['gain'] for item in by_catalog.values() if item['gain'] is not None]
    sign_flip = len(gains) > 1 and min(gains) < 0 < max(gains)
    moved = mapping_transfer(comparison.snap, comparison)['changed_cards']
    if sign_flip and moved:
        cards = '、'.join(f'{card}（{old["exposure"]}→{old["current"]}）'
                          for card, old in moved.items())
        return 'not_transferable', (
            f'按候选目录版本拆开后增益方向相反；{cards} 的服务点归属已经迁移，'
            '旧版曝光的证据不能直接用到现在的服务点上；现版人群也还不够成熟。'
        )
    if sign_flip:
        return 'mixed_support', (
            '按候选目录版本拆开后增益方向相反，版前版后不是同一个策略，'
            '不能合并成单一增益。'
        )
    immature = any(
        item['open_usable'] for item in by_catalog.values()
    )
    if immature:
        return 'immature', (
            '仍有观察窗口未结，未结且缺证的领取不是零；当前点估计不足以定论。'
        )
    spans_zero = (
        (interval_lower is None or interval_lower[0] <= 0 <= interval_lower[1])
        and (interval_upper is None or interval_upper[0] <= 0 <= interval_upper[1])
    )
    if spans_zero:
        return 'inconclusive', (
            '点估计与区间覆盖正负，极端权重和未成熟回执足以改变建议；'
            '尚不足以支持小流量替换。'
        )
    if (lower['gain'] or 0) > 0:
        return 'supported', '下界与上界同向为正，可作为小流量候选，仍需额度核对。'
    if (upper['gain'] or 0) < 0:
        return 'negative', '上下界同向为负，不建议替换。'
    return 'inconclusive', '证据方向不一致，不能据此替换。'


def summarise(snap: Snapshot) -> list[dict]:
    results = []
    for proposal in snap.proposals:
        comparison = build(snap, proposal)
        category, reason = verdict(comparison)
        lower = comparison.estimate(False)
        upper = comparison.estimate(True)
        # 审批按保守下界的目标办结率折算兑现占用；没有证据时不给折扣。
        completion_rate = lower['target'] if lower['target'] is not None else None
        capacity = capacity_check(
            snap, comparison, int(proposal['traffic']), float(proposal['holdout']),
            completion_rate,
        )
        transfer = mapping_transfer(snap, comparison)
        naive = (round(comparison.snap.marginal_product(
            comparison.usable[0] if comparison.usable else comparison.eligible[0],
            comparison.slate), 4) if comparison.eligible else None)
        joint = comparison.probability
        results.append({
            'plan': proposal['plan'],
            'entry': comparison.entry,
            'region': comparison.region,
            'slot': comparison.slot,
            'slate': comparison.slate,
            'asof': comparison.asof,
            'traffic': int(proposal['traffic']),
            'holdout': float(proposal['holdout']),
            'users': sorted({snap.requests[r]['user'] for r in comparison.eligible}),
            'counts': {'eligible': len(comparison.eligible),
                       'producible': len(comparison.producible),
                       'usable': len(comparison.usable),
                       'exposed': len(comparison.exposed)},
            'probability': joint,
            'min_support': comparison.min_support,
            'weight_cap': comparison.weight_cap,
            'max_uncapped_weight': (round(1 / joint, 2) if joint else None),
            'cap_binds': bool(joint and 1 / joint > comparison.weight_cap),
            'naive_product': naive,
            'naive_differs_from_joint': bool(joint is not None and naive is not None
                                             and abs(naive - joint) > 1e-9),
            'blockers': comparison.blockers,
            'verdict': category,
            'reason': reason,
            'lower': lower,
            'upper': upper,
            'interval_lower': comparison.interval(False),
            'interval_upper': comparison.interval(True),
            'by_catalog': comparison.by_catalog(False),
            'transfer': transfer,
            'open_windows': sum(1 for r in comparison.usable
                                if comparison.receipts.states[r]['open']),
            'open_unresolved_exposed': sum(
                1 for r in comparison.exposed
                if comparison.receipts.states[r]['open']
                and comparison.receipts.states[r]['unresolved']),
            'duplicate_receipts': sum(comparison.receipts.states[r]['duplicate_receipts']
                                      for r in comparison.usable),
            'same_provider_orders': comparison.same_provider_orders(),
            'exposure_services': comparison.exposure_services(),
            'capacity': capacity,
            'over_capacity': any(
                row['status'] in ('exhausted', 'insufficient', 'no_quota_record')
                for row in capacity['providers']),
            'unusable': not comparison.usable,
            'comparable': category not in ('no_counterfactual',),
        })
    return results


def overlap(plans: list[dict]) -> list[dict]:
    """发布候选之间的重叠：用户人群与共享服务点/时段。"""
    pairs = []
    for index, left in enumerate(plans):
        for right in plans[index + 1:]:
            left_providers = {(r['provider'], r['slot'])
                              for r in left['capacity']['providers']}
            right_providers = {(r['provider'], r['slot'])
                               for r in right['capacity']['providers']}
            pairs.append({
                'left': left['plan'], 'right': right['plan'],
                'shared_users': len(set(left['users']) & set(right['users'])),
                'shared_quota': sorted(left_providers & right_providers),
            })
    return pairs


def running_handover(snap: Snapshot) -> list[dict]:
    """已在跑的入口保留当时发出的规则；需要降额只能出运营可交接的变更。

    服务点评审后减额时：
    - 仍在额度内的入口保留线上规则，但标出「评审后改版」，提醒运营复核共享余量；
    - 承诺已超过新版额度的入口，不静默停掉，而是留下一份可交接的降额变更。
    """
    capacities = effective_capacity(snap)
    rows = []
    for row in snap.running:
        key = (row['provider'], row['slot'])
        cap = capacities.get(key, {})
        committed = int(row['committed'])
        active_from = int(row['active'])
        headroom = cap.get('headroom')
        revised = cap.get('revised', False)
        others = cap.get('committed', 0) - committed
        notes = []
        if headroom is not None and committed > cap.get('limit', 0):
            notes.append('承诺已超过评审后额度：保留现有规则，另出运营可交接的降额变更。')
        elif revised:
            notes.append('服务点在评审后改版减额：保留现有规则，'
                         '新入口的共享余量需重新核对。')
        if headroom is not None and others > headroom:
            notes.append('同一服务点的其他已批准流量已占去余量：不能再用剩余额度批新入口。')
        rows.append({
            'approval': row['approval'], 'entry': row['entry'],
            'provider': row['provider'], 'slot': row['slot'],
            'active_from': active_from, 'committed': committed,
            'rule': row['rule'],
            'limit': cap.get('limit'), 'used': cap.get('used'),
            'other_committed': others,
            'headroom': headroom, 'revised': revised,
            'note': '；'.join(notes) if notes else '当前额度可覆盖。',
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='公益家教入口的离线回放')
    parser.add_argument('--json', action='store_true', help='输出 JSON')
    args = parser.parse_args()
    snap = Snapshot.load()
    plans = summarise(snap)
    payload = {
        'plans': plans,
        'overlap': overlap(plans),
        'running': running_handover(snap),
        'capacity': {f'{p}/{s}': row for (p, s), row in effective_capacity(snap).items()},
        'shared_quota': {f'{p}/{s}': row for (p, s), row in shared_quota(snap, plans).items()},
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=list))
        return
    for item in payload['plans']:
        print(f"\n[{item['plan']}] {item['entry']}/{item['region']} {item['slate']}"
              f"  asof={item['asof']}  结论={item['verdict']}")
        print(f"  请求 可用 {item['counts']['eligible']} / 可产生次序 {item['counts']['producible']}"
              f" / 支持达标 {item['counts']['usable']} / 实际曝光 {item['counts']['exposed']}")
        print(f"  联合概率 {item['probability']} 边际相乘 {item['naive_product']}"
              f" 权重上限 {item['weight_cap']} 上限是否截断 {item['cap_binds']}")
        print(f"  未结 {item['open_windows']} 去重回执 {item['duplicate_receipts']}"
              f" 同点双卡 {item['same_provider_orders']}"
              f" 曝光回执/唯一服务 {item['exposure_services']['click_receipts']}"
              f"/{item['exposure_services']['unique_services']}")
        for label in ('lower', 'upper'):
            est = item[label]
            if est['gain'] is None:
                print(f'  {label}: 无法估计')
                continue
            ci = item['interval_' + label]
            ci_text = (f" 按用户区间 [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                       if ci else '')
            print(f"  {label}: 基线 {est['baseline']:.4f} 目标 {est['target']:.4f}"
                  f" 增益 {est['gain']:+.4f}{ci_text}")
        for version, group in item['by_catalog'].items():
            gain = '无' if group['gain'] is None else f"{group['gain']:+.4f}"
            print(f"  目录 {version}: 可用 {group['usable']} 曝光 {group['exposed']}"
                  f" 未结 {group['open_usable']} 增益 {gain}")
        changed = item['transfer']['changed_cards']
        print('  卡片归属: ' + ('未变' if not changed else
                               '；'.join(f'{c} {v["exposure"]}->{v["current"]}'
                                        for c, v in changed.items())))
        for row in item['capacity']['providers']:
            print(f"  额度 {row['provider']}/{row['slot']}: {row['status']}"
                  f" 余 {row.get('headroom')} 请求上限 {row['request_ceiling']}"
                  f" 预计兑现 {row['expected_redemptions']}"
                  f" 治疗 {row['treatment']} 对照 {row['control']}"
                  + ('（余量偏薄）' if row.get('thin') else ''))
        print('  ' + item['reason'])
    print('\n候选之间的重叠：')
    for pair in payload['overlap']:
        print(f"  {pair['left']} vs {pair['right']}: 共享用户 {pair['shared_users']}"
              f" 共享额度 {pair['shared_quota']}")
    print('\n共享额度（多个入口不能重复使用同一剩余额度）：')
    for key, row in payload['shared_quota'].items():
        print(f"  {key}: 余 {row['headroom']} 请求上限合计 {row['demand']}"
              f" 预计兑现合计 {row['expected']}"
              f" 入口 {[e['plan'] for e in row['entries']]}"
              f" 已跑 {[r['approval'] for r in row['running']]}"
              f" 超用 {row['overcommitted']}")
    print('\n已批准流量与额度交接：')
    for row in payload['running']:
        print(f"  {row['approval']} {row['entry']} {row['provider']}/{row['slot']}"
              f" 生效日={row['active_from']} committed={row['committed']}"
              f" 其他已批={row['other_committed']}"
              f" 余={row['headroom']} {row['note']}")


if __name__ == '__main__':
    main()

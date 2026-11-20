"""核对离线回放的口径：反事实、去重、未结窗口与额度。"""
import unittest

from analysis.replay import (
    ReceiptView,
    Snapshot,
    build,
    capacity_check,
    effective_capacity,
    latest_mapping,
    overlap,
    running_handover,
    summarise,
    verdict,
)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snap = Snapshot.load()
        cls.plans = {plan['plan']: plan for plan in summarise(cls.snap)}


class Counterfactual(Fixture):
    def test_only_producible_orders_support_counterfactual(self):
        # guide 只在 support 中出现 d|a，从未出现 a|d。
        v4 = self.plans['v4']
        self.assertEqual(v4['counts']['eligible'], 48)
        self.assertEqual(v4['counts']['producible'], 0)
        self.assertEqual(v4['verdict'], 'no_counterfactual')
        self.assertIn('order_never_producible', v4['blockers'])

    def test_insufficient_support_is_not_zero_demand(self):
        v1 = self.plans['v1']
        # chat 的 c|a 每次都以 0.02 出现，低于 min_support 0.18。
        self.assertEqual(v1['counts']['producible'], 160)
        self.assertEqual(v1['counts']['usable'], 0)
        self.assertEqual(v1['probability'], None)
        self.assertEqual(v1['verdict'], 'no_counterfactual')
        self.assertIn('support_below_minimum', v1['blockers'])

    def test_marginal_product_is_not_the_joint_probability(self):
        # 只用边际概率拼 a|b 会得到 0.64，实际整屏联合概率是 0.80。
        chat = next(rid for rid, row in self.snap.requests.items() if row['entry'] == 'chat')
        joint = self.snap.support[chat]['a|b']
        product = self.snap.marginal_product(chat, 'a|b')
        self.assertAlmostEqual(joint, 0.80)
        self.assertAlmostEqual(product, 0.64)
        self.assertNotAlmostEqual(product, joint)
        # 也说明不能用它给从未出现的搭配报高置信增益。
        self.assertAlmostEqual(self.snap.marginal_product(chat, 'c|a'), 0.004)

    def test_v4_not_producible_even_at_high_weight(self):
        proposal = next(p for p in self.snap.proposals if p['plan'] == 'v4')
        comparison = build(self.snap, proposal)
        self.assertEqual(comparison.producible, [])
        self.assertIsNone(comparison.estimate(False)['gain'])


class Receipts(Fixture):
    def test_duplicate_redirect_is_one_service(self):
        # r0001 有两条回执，但同一服务点 q1 与同一接续 k001：重复跳转，不是两次办理。
        # 第二条回执 known=31，只有到该截止日才看得见。
        view = ReceiptView.at(self.snap, asof=31, horizon=7)
        state = view.states['r0001']
        self.assertEqual(len(state['services']), 1)
        self.assertEqual(state['duplicate_receipts'], 1)
        self.assertTrue(state['completed'])
        # 评审 asof=28 时第二条尚未回报，不应当作重复或新办理。
        early = ReceiptView.at(self.snap, asof=28, horizon=7)
        self.assertEqual(early.states['r0001']['duplicate_receipts'], 0)
        self.assertEqual(len(early.states['r0001']['services']), 1)

    def test_cross_request_booking_is_not_merged(self):
        # 89 个 booking 出现在多个请求里：同一服务点被不同请求接续，不能当重复。
        by_booking = {}
        for row in self.snap.outcomes:
            by_booking.setdefault(row['booking'], set()).add(row['request'])
        shared = [booking for booking, requests in by_booking.items() if len(requests) > 1]
        self.assertEqual(len(shared), 89)
        view = ReceiptView.at(self.snap, asof=28, horizon=7)
        # k001 同时服务 r0001、r0114、r0227，这三个请求都各自保留自己的接续。
        for request in ('r0001', 'r0114', 'r0227'):
            self.assertTrue(view.states[request]['clicked'])

    def test_open_window_is_not_zero(self):
        # v2 的 home 在 day 23 的请求窗口未结：未办结的记 0/1 两端，不当成已办结或零需求。
        proposal = next(p for p in self.snap.proposals if p['plan'] == 'v2')
        comparison = build(self.snap, proposal)
        open_requests = [r for r in comparison.usable
                         if comparison.receipts.states[r]['open']]
        self.assertTrue(open_requests)
        unresolved = [r for r in open_requests
                      if not comparison.receipts.states[r]['completed']]
        self.assertTrue(unresolved)
        for request in unresolved:
            self.assertEqual(comparison.receipts.closed(request), 0)
            self.assertEqual(comparison.receipts.open_upper(request), 1)
        # 已办结的未结窗口请求仍按办结计 1。
        for request in set(open_requests) - set(unresolved):
            self.assertEqual(comparison.receipts.closed(request), 1)
        self.assertEqual(self.plans['v2']['open_windows'], 20)
        self.assertGreaterEqual(self.plans['v2']['open_unresolved_exposed'], 1)
        # 未结回执在两侧都抬高：目标与基线都随上界上升，未结不当作失败。
        self.assertGreater(self.plans['v2']['upper']['target'],
                           self.plans['v2']['lower']['target'])
        self.assertGreater(self.plans['v2']['upper']['baseline'],
                           self.plans['v2']['lower']['baseline'])

    def test_observed_after_asof_is_excluded(self):
        # b0119 的 known=29 晚于 v2 的 asof=28，不能计入该次评审。
        view = ReceiptView.at(self.snap, asof=28, horizon=7)
        self.assertFalse(view.states['r0119']['completed'])

    def test_interval_uses_user_resampling(self):
        # 每个用户在同一入口有多次访问，区间按用户整体重抽样而不是按请求独立。
        proposal = next(p for p in self.snap.proposals if p['plan'] == 'v2')
        comparison = build(self.snap, proposal)
        users = {self.snap.requests[r]['user'] for r in comparison.usable}
        self.assertLess(len(users), len(comparison.usable))
        counts = {}
        for request in comparison.usable:
            user = self.snap.requests[request]['user']
            counts[user] = counts.get(user, 0) + 1
        self.assertGreater(max(counts.values()), 1)
        interval = self.plans['v2']['interval_lower']
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], interval[1])


class Transfer(Fixture):
    def test_card_moved_between_services(self):
        # home 的 b 卡在目录 v1 属于 q1，v2 起属于 q2；旧曝光不按新目录回写。
        proposal = next(p for p in self.snap.proposals if p['plan'] == 'v2')
        comparison = build(self.snap, proposal)
        changed = self.plans['v2']['transfer']['changed_cards']
        self.assertEqual(changed['b'], {'exposure': 'q1', 'current': 'q2'})
        # 旧曝光里 b 仍是 q1：以当次 candidates 快照为准。
        old_request = next(r for r in comparison.exposed
                           if self.snap.requests[r]['logger'] == 'p1')
        self.assertEqual(self.snap.candidates[old_request]['b']['provider'], 'q1')
        self.assertEqual(latest_mapping(self.snap, 'home')['b']['provider'], 'q2')

    def test_era_gains_have_opposite_signs(self):
        by_catalog = self.plans['v2']['by_catalog']
        self.assertGreater(by_catalog['v1']['gain'], 0)
        self.assertLess(by_catalog['v2']['gain'], 0)
        self.assertEqual(self.plans['v2']['verdict'], 'not_transferable')

    def test_two_cards_on_one_provider_are_not_two_services(self):
        # plan v2 的 b|a 有 100 个请求两张卡都在 q1：两个点击只代表一个服务点。
        self.assertEqual(self.plans['v2']['same_provider_orders'], 100)
        request = next(r for r in self.snap.requests
                       if self.snap.requests[r]['entry'] == 'home'
                       and self.snap.card_catalog(r, 'b') == 'v1')
        self.assertEqual(self.snap.unique_providers(request, 'b|a'), ['q1'])
        self.assertEqual(len(self.snap.order_providers(request, 'b|a')), 2)


class Quota(Fixture):
    def test_later_capacity_revision_wins(self):
        # q1/evening 有 known=24/limit=75 的旧版，评审后 known=31/limit=30 生效。
        capacity = effective_capacity(self.snap)
        row = capacity[('q1', 'evening')]
        self.assertEqual(row['limit'], 30)
        self.assertEqual(row['known'], 31)
        self.assertTrue(row['revised'])
        # 已批准 committed=27，余量 3。
        self.assertEqual(row['committed'], 27)
        self.assertEqual(row['headroom'], 3)

    def test_committed_is_the_reservation(self):
        # running.active 是生效日，不是流量；committed 才是已批准占用。
        capacity = effective_capacity(self.snap)
        q2 = capacity[('q2', 'afternoon')]
        self.assertEqual(q2['active_from'], [27])
        self.assertEqual(q2['committed'], 10)
        self.assertEqual(q2['headroom'], 40)

    def test_all_plans_respect_holdout(self):
        for plan in self.plans.values():
            traffic = plan['traffic']
            holdout = plan['holdout']
            capacity = plan['capacity']
            self.assertEqual(capacity['control'], round(traffic * holdout + 0.4999))
            self.assertEqual(capacity['control'] + capacity['treatment'], traffic)
            self.assertGreaterEqual(capacity['control'] / traffic, holdout)

    def test_one_entry_cannot_draw_on_another(self):
        # q1/evening 额度只剩 3，多个入口不能各自按满额批。
        for plan, holdout in (('v2', 0.2), ('v3', 0.15), ('v4', 0.15)):
            comparison = build(
                self.snap,
                next(p for p in self.snap.proposals if p['plan'] == plan),
            )
            evidence = comparison.estimate(False)['target']
            rows = capacity_check(self.snap, comparison, 85, holdout, evidence)['providers']
            q1 = next(row for row in rows if row['provider'] == 'q1')
            self.assertLessEqual(q1['headroom'], 3)
            self.assertTrue(q1['thin'])
            if q1['expected_redemptions'] is not None:
                self.assertGreater(q1['expected_redemptions'], q1['headroom'])
                self.assertEqual(q1['status'], 'insufficient')
            else:
                # 没有办结率时只能标「需求未知」，不能凭缺证判成够。
                self.assertEqual(q1['status'], 'demand_unknown')

    def test_running_rule_is_kept_and_handover_is_explicit(self):
        handover = {row['approval']: row for row in running_handover(self.snap)}
        self.assertEqual(handover['a0']['rule'], 'p1')
        self.assertEqual(handover['a0']['committed'], 27)
        self.assertEqual(handover['a0']['active_from'], 27)
        # q1/evening 评审后减额：保留线上规则，但标出改版并提示核对共享余量。
        self.assertTrue(handover['a0']['revised'])
        self.assertIn('保留现有规则', handover['a0']['note'])
        self.assertEqual(handover['a0']['headroom'], 3)
        # a1 的额度是另一版且未被减额，不需要交接变更。
        self.assertFalse(handover['a1']['revised'])
        self.assertEqual(handover['a1']['committed'], 10)

    def test_overlap_reports_shared_quota_and_users(self):
        pairs = {(p['left'], p['right']): p for p in overlap(self.plans_list())}
        self.assertEqual(pairs[('v2', 'v3')]['shared_users'], 0)
        self.assertIn(('q1', 'evening'), pairs[('v2', 'v3')]['shared_quota'])

    def plans_list(self):
        return list(self.plans.values())


class Verdicts(Fixture):
    def test_each_plan_is_classified_by_evidence(self):
        observed = {name: plan['verdict'] for name, plan in self.plans.items()}
        self.assertEqual(observed['v1'], 'no_counterfactual')
        self.assertEqual(observed['v2'], 'not_transferable')
        self.assertEqual(observed['v3'], 'inconclusive')
        self.assertEqual(observed['v4'], 'no_counterfactual')

    def test_v3_interval_covers_zero(self):
        interval = self.plans['v3']['interval_lower']
        self.assertLessEqual(interval[0], 0)
        self.assertGreaterEqual(interval[1], 0)
        # 点估计为负，但极端权重和未成熟回执足以改变建议。
        self.assertLess(self.plans['v3']['lower']['gain'], 0)
        self.assertFalse(self.plans['v3']['cap_binds'])
        self.assertLess(self.plans['v3']['max_uncapped_weight'], 20)


class Synthetic(unittest.TestCase):
    """用最小合成资料走通正/负/权重截断三条路径。"""

    def snapshot(self, completed_first: bool, probability: float = 0.5,
                 max_weight: float = 10.0):
        snap = Snapshot()
        support = {'x|y': probability, 'y|x': 1 - probability}
        receipts = []
        for index in range(100):
            request = f'q{index:03d}'
            slate = 'x|y' if index < 50 else 'y|x'
            snap.requests[request] = {
                'id': request, 'user': f'u{index:03d}', 'day': '1',
                'entry': 'e', 'slot': 'afternoon', 'region': 'all',
                'slate': slate, 'probability': str(support[slate]), 'logger': 'p',
            }
            snap.candidates[request] = {
                card: {'request': request, 'card': card, 'provider': f'v{card}',
                       'catalog': 'c1', 'region': 'all'}
                for card in ('x', 'y')
            }
            snap.support[request] = dict(support)
            completed = completed_first == (slate == 'x|y')
            receipts.append({
                'receipt': f'r{index:03d}', 'request': request, 'card': 'x',
                'click_day': '1', 'known': '2',
                'state': 'completed' if completed else 'clicked',
                'booking': f'k{index:03d}',
            })
        snap.outcomes = receipts
        snap.capacity = [
            {'provider': 'vx', 'slot': 'afternoon', 'known': '1', 'limit': '100'},
            {'provider': 'vy', 'slot': 'afternoon', 'known': '1', 'limit': '100'},
        ]
        snap.proposals = [{
            'plan': 'p', 'entry': 'e', 'region': 'all', 'from_day': '1',
            'to_day': '1', 'slate': 'x|y', 'horizon': '1', 'min_support': '0.4',
            'max_weight': str(max_weight), 'holdout': '0.2', 'traffic': '50',
            'asof': '10',
        }]
        return snap

    def test_positive_is_supported(self):
        snap = self.snapshot(completed_first=True)
        comparison = build(snap, snap.proposals[0])
        self.assertEqual(verdict(comparison)[0], 'supported')
        estimate = comparison.estimate(False)
        self.assertAlmostEqual(estimate['gain'], 0.5)
        interval = comparison.interval(False, reps=500)
        self.assertGreater(interval[0], 0)

    def test_negative_is_rejected(self):
        snap = self.snapshot(completed_first=False)
        comparison = build(snap, snap.proposals[0])
        self.assertEqual(verdict(comparison)[0], 'negative')
        self.assertAlmostEqual(comparison.estimate(False)['gain'], -0.5)

    def test_weight_cap_is_flagged(self):
        # 联合概率 0.05 时逆概率权重 20 超过 max_weight=10，必须标出截断。
        snap = self.snapshot(completed_first=True, probability=0.05, max_weight=10)
        snap.proposals[0]['min_support'] = '0.01'
        comparison = build(snap, snap.proposals[0])
        self.assertIn('support_weight_capped', comparison.blockers)
        estimate = comparison.estimate(False)
        self.assertEqual(estimate['capped'], 50)
        # 截断后每份曝光权重 10，50 份 / 100 人 = 5.0，已越过概率上界 1；
        # 这说明截断本身会让目标偏高，只能作为警示，不能当发布依据。
        self.assertAlmostEqual(estimate['target'], 5.0)
        self.assertAlmostEqual(estimate['sum_weight'], 500.0)
        self.assertEqual(verdict(comparison)[0], 'inexact')

    def test_marginal_product_differs(self):
        snap = self.snapshot(completed_first=True)
        # 两个位置边际都是 0.5，相乘只有 0.25；联合概率是 0.5。
        self.assertAlmostEqual(snap.marginal_product('q000', 'x|y'), 0.25)
        self.assertAlmostEqual(snap.support['q000']['x|y'], 0.5)


if __name__ == '__main__':
    unittest.main()

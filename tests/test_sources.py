"""核对业务资料的格式与可关联性。"""
import csv
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parents[1] / 'data'
TABLES = {'requests': ('id', 'user', 'day', 'entry', 'slot', 'region', 'slate', 'probability', 'logger'), 'candidates': ('request', 'card', 'provider', 'catalog', 'region'), 'support': ('request', 'slate', 'probability'), 'outcomes': ('receipt', 'request', 'card', 'click_day', 'known', 'state', 'booking'), 'proposals': ('plan', 'entry', 'region', 'from_day', 'to_day', 'slate', 'horizon', 'min_support', 'max_weight', 'holdout', 'traffic', 'asof'), 'capacity': ('provider', 'slot', 'known', 'limit'), 'running': ('approval', 'entry', 'provider', 'slot', 'active', 'committed', 'rule')}
RELATION = ('outcomes', 'request', 'requests', 'id')


class SourceContract(unittest.TestCase):
    def read(self, table):
        with (DATA / f'{table}.csv').open(encoding='utf-8', newline='') as file:
            return list(csv.DictReader(file))

    def test_shape(self):
        for name, columns in TABLES.items():
            with self.subTest(table=name):
                rows = self.read(name)
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(tuple(row), columns)
                    self.assertNotIn(None, row.values())

    def test_references(self):
        left, left_key, right, right_key = RELATION
        self.assertLessEqual({row[left_key] for row in self.read(left)}, {row[right_key] for row in self.read(right)})

    def test_source_variation(self):
        support = {(row['request'], row['slate']): row['probability'] for row in self.read('support')}
        for row in self.read('requests'):
            self.assertEqual(support[row['id'], row['slate']], row['probability'])

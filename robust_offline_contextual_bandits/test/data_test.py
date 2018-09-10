import tensorflow as tf
from robust_offline_contextual_bandits.data import TaggedData, TaggedDatum


class DataTest(tf.test.TestCase):
    def test_tagged_data(self):
        patient = TaggedData()
        patient.append('d1', a=1, b=2)
        patient.append('d2', a=3, b=2)
        patient.append('d3', a=3, b=4)
        b2 = TaggedData(datum for datum in patient if datum['b'] == 2)

        assert len(b2) == 2
        assert b2[0] == TaggedDatum('d1', a=1, b=2)
        assert b2[1] == TaggedDatum('d2', a=3, b=2)

        assert b2[0]['a'] == 1
        assert b2[0]['b'] == 2
        assert b2[0]['c'] is None
        assert b2[0]() == 'd1'

        patient = b2[0].with_tags('d5')
        assert b2[0]['a'] == 1
        assert b2[0]['b'] == 2
        assert b2[0]['c'] is None
        assert patient() == 'd5'


if __name__ == '__main__':
    tf.test.main()

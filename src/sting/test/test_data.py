import random
from unittest import TestCase

from sting.data import *


def _nominal_enum():
    # Helper method, generates a simple 3-value enum for nominal/id features
    return nominal_feature_values('value', ['a', 'b', 'c'])


# noinspection DuplicatedCode
class TestFeature(TestCase):
    def test_string_ftype(self):
        # using the string name for each feature type should work and be casted
        f = Feature('f', 'CLASS')
        self.assertTrue(f.ftype == Feature.Type.CLASS)
        f = Feature('f', 'INDEX')
        self.assertTrue(f.ftype == Feature.Type.INDEX)
        f = Feature('f', 'BINARY')
        self.assertTrue(f.ftype == Feature.Type.BINARY)
        f = Feature('f', 'NOMINAL', _nominal_enum())
        self.assertTrue(f.ftype == Feature.Type.NOMINAL)
        f = Feature('f', 'CONTINUOUS')
        self.assertTrue(f.ftype == Feature.Type.CONTINUOUS)

    def test_values_attribute(self):
        # Failing to define the values attribute for a NOMINAL feature should cause a ValueError
        self.assertRaises(ValueError, lambda: Feature('f', Feature.Type.NOMINAL))
        # Defining the values attribute for any other feature type should cause a ValueError
        self.assertRaises(ValueError, lambda: Feature('f', Feature.Type.INDEX, _nominal_enum()))
        self.assertRaises(ValueError, lambda: Feature('f', Feature.Type.CLASS, _nominal_enum()))
        self.assertRaises(ValueError, lambda: Feature('f', Feature.Type.BINARY, _nominal_enum()))
        self.assertRaises(ValueError, lambda: Feature('f', Feature.Type.CONTINUOUS, _nominal_enum()))

    def test_order(self):
        # Assert that Feature instances are orderable
        # noinspection PyTypeChecker
        sorted([
            Feature('f_class', Feature.Type.CLASS),
            Feature('f_binary', Feature.Type.BINARY),
            Feature('f_nominal', Feature.Type.NOMINAL, _nominal_enum()),
            Feature('f_cont', Feature.Type.CONTINUOUS)
        ])

    def test_to_float(self):
        e = _nominal_enum()
        # Class and binary features should convert to their equivalent float value
        f = Feature('f', 'CLASS')
        self.assertEqual(0., f.to_float(0))
        self.assertEqual(1., f.to_float(1))

        f = Feature('f', 'BINARY')
        self.assertEqual(0., f.to_float(0))
        self.assertEqual(1., f.to_float(1))
        self.assertEqual(0., f.to_float(False))
        self.assertEqual(1., f.to_float(True))

        # Nominal features should convert to their associated enum value
        f = Feature('f', 'NOMINAL', e)
        self.assertEqual(1., f.to_float(e.a))
        self.assertEqual(1., f.to_float('a'))
        self.assertEqual(1., f.to_float(1))
        self.assertEqual(2., f.to_float(e.b))
        self.assertEqual(2., f.to_float('b'))
        self.assertEqual(2., f.to_float(2))
        self.assertEqual(3., f.to_float(e.c))
        self.assertEqual(3., f.to_float('c'))
        self.assertEqual(3., f.to_float(3))

        # Continuous features should be the same because they are already floats
        f = Feature('f', 'CONTINUOUS')
        for _ in range(100):
            n = random.random()
            self.assertEqual(n, f.to_float(n))

    def test_from_float(self):
        e = _nominal_enum()

        # Class should convert to 0 and 1
        f = Feature('f', 'CLASS')
        self.assertEqual(0, f.from_float(0.))
        self.assertEqual(1, f.from_float(1.))

        # Binary should convert to True and False
        f = Feature('f', 'BINARY')
        self.assertEqual(False, f.from_float(0.))
        self.assertEqual(True, f.to_float(1.))

        # Nominal features should convert to their original enum values
        f = Feature('f', 'NOMINAL', e)
        self.assertEqual(e.a, f.from_float(1.))
        self.assertEqual(e.b, f.from_float(2.))
        self.assertEqual(e.c, f.from_float(3.))

        # Continuous should stay the same because they're just floats
        f = Feature('f', 'CONTINUOUS')
        for _ in range(100):
            n = random.random()
            self.assertEqual(n, f.from_float(n))


class TestExample(TestCase):
    def test_weight_value(self):
        # Make sure error is raised when invalid weight value is passed
        self.assertRaises(ValueError, lambda: Example({}, weight=-1.))
        self.assertRaises(ValueError, lambda: Example({}, weight=2.))

    def test_move_label(self):
        # Make sure it throws an error when label is defined both in args and dict
        self.assertRaises(ValueError, lambda: Example({'label': 0}, label=0))
        # Make sure passing 'label' key into dict sets sends it to the .label attr
        self.assertEqual(0, Example({'label': 0}).label)

    def test_move_index(self):
        # Make sure it throws an error when index is defined both in args and dict
        self.assertRaises(ValueError, lambda: Example({'index': 0}, index=0))
        # Make sure passing 'label' key into dict sets sends it to the .label attr
        self.assertEqual(0, Example({'index': 0}).index)


class TestDataSet(TestCase):
    def test_input(self):
        # Test empty DataSet
        d = DataSet()
        self.assertTrue(np.array_equal(d._data.to_numpy(), np.empty((0, 0))))

        # DataFrame for test DataSet
        result_df = DataFrame({
            'index': [1, 2, 3, 4],
            'name': ['Red', 'Blue', 'Green', 'Yellow'],
            'label': [0, 1, 1, 0],
            'weight': [1., 1., 1., 1.]
        })
        result_df.set_index('index', inplace=True)
        result_df['name'] = result_df['name'].astype('category')

        # Test DataFrame input for DataSet
        # Also test removing id and label
        data = {'index': [1, 2, 3, 4],
                'name': ['Red', 'Blue', 'Green', 'Yellow'],
                'label': [0, 1, 1, 0]}
        df = DataFrame(data)
        d = DataSet(data=df)
        pd.testing.assert_frame_equal(d._data, result_df)
        self.assertTrue(d._data.equals(result_df))
        self.assertEqual([1, 2, 3, 4], list(d._data.index))

        # Test Dictionary input for DataSet
        it_dict_data = [{'index': 1, 'name': 'Red', 'label': 0},
                        {'index': 2, 'name': 'Blue', 'label': 1},
                        {'index': 3, 'name': 'Green', 'label': 1},
                        {'index': 4, 'name': 'Yellow', 'label': 0}]
        d = DataSet(data=it_dict_data)
        pd.testing.assert_frame_equal(d._data, result_df)

        # Test Example input for DataSet
        ex1 = Example({'index': 1, 'name': 'Red', 'label': 0})
        ex2 = Example({'index': 2, 'name': 'Blue', 'label': 1})
        ex3 = Example({'index': 3, 'name': 'Green', 'label': 1})
        ex4 = Example({'index': 4, 'name': 'Yellow', 'label': 0})
        d = DataSet(data=[ex1, ex2, ex3, ex4])
        pd.testing.assert_frame_equal(d._data, result_df)

        # Test Unzip
        data, label = d.unzip()
        pd.testing.assert_frame_equal(data, pd.DataFrame(result_df.drop('label', axis=1)))
        self.assertEqual([0, 1, 1, 0], list(label))

        # Test len
        self.assertEqual(d.__len__(), 4)

        # Test Continuous Schema
        self.assertEqual(d.schema[1].ftype, Feature.Type.CONTINUOUS)
        # Test Nominal Schema
        self.assertEqual(d.schema[0].ftype, Feature.Type.NOMINAL)
        # Test Binary Schema
        exb1 = Example({'index': 1, 'name': 1, 'label': 0})
        exb2 = Example({'index': 2, 'name': 0, 'label': 1})
        d = DataSet(data=[exb1, exb2])
        self.assertEqual(d.schema[0].ftype, Feature.Type.BINARY)

    def test_set_weight(self):
        # Test set_weight
        data = {'index': [1, 2, 3, 4],
                'name': ['Red', 'Blue', 'Green', 'Yellow'],
                'label': [0, 1, 1, 0]}
        df = DataFrame(data)
        d = DataSet(data=df)
        self.assertTrue(np.array_equal(d._data['weight'].to_numpy(), np.array([1., 1., 1., 1.])))
        d.set_weight(1, 0.5)
        d.set_weight(2, 0.2)
        d.set_weight(3, 0.7)
        self.assertEqual(list(d._data['weight']), [0.5, 0.2, 0.7, 1.])
        self.assertRaises(ValueError, lambda: d.set_weight(3, 2.5))

    def test_iter(self):
        # Test iter
        ex1 = Example({'name': 'Red', 'label': 0})
        ex2 = Example({'name': 'Blue', 'label': 1})
        ex3 = Example({'name': 'Green', 'label': 0})
        ex4 = Example({'name': 'Yellow', 'label': 1})
        d = DataSet(data=[ex1, ex2, ex3, ex4])
        self.assertEqual(list(d.__iter__()), [ex1, ex2, ex3, ex4])

        # Test getitem
        self.assertEqual(d.__getitem__(0), ex1)
        self.assertEqual(list(d[0:2].__iter__()), [ex1, ex2])

    def test_append(self):
        # Test append
        ex1 = Example({'name': 'Red', 'label': 0})
        ex2 = Example({'name': 'Blue', 'label': 0})

        ex3 = Example({'name': 'Red', 'label': 1})
        ex4 = Example({'name': 'Blue', 'label': 1})

        d1 = DataSet(data=[ex1, ex2])
        d2 = DataSet(data=[ex3, ex4])
        d = d1.append(d2)
        self.assertEqual(list(d.__iter__()), [ex1, ex2, ex3, ex4])

    def test_concat(self):
        # Test concat
        ex1 = Example({'name': 'Red', 'label': 0})
        ex2 = Example({'name': 'Blue', 'label': 1})
        ex3 = Example({'name': 'Red', 'label': 0})
        ex4 = Example({'name': 'Blue', 'label': 1})

        d1 = DataSet(data=[ex1])
        d2 = DataSet(data=[ex2])
        d3 = DataSet(data=[ex3])
        d4 = DataSet(data=[ex4])
        d = DataSet.concat(d1, d2, d3, d4)
        self.assertEqual(list(d.__iter__()), [ex1, ex2, ex3, ex4])

    def test_getitem(self):
        data = {'size': [3.5, 2.2, 1.3, 4.7],
                'name': ['Red', 'Blue', 'Green', 'Yellow']}
        Name = nominal_feature_values('Name', ['Red', 'Blue', 'Green', 'Yellow'])
        schema = [Feature('size', Feature.Type.CONTINUOUS),
                  Feature('name', Feature.Type.NOMINAL, Name)]
        df = DataFrame(data)
        d = DataSet(data=df, schema=schema)
        # Test that single item is an example
        self.assertEqual(Example({'size': 2.2, 'name': 'Blue'}), d[1])


class TestMLData(TestCase):
    def setUp(self) -> None:
        self.test_dataset = parse_c45('example', rootdir='test/')

    def test_parsed(self):
        self.assertTrue(self.test_dataset is not None)

    def test_schema(self):
        self.assertEqual(len(self.test_dataset.schema), 4)
        correct_types = (Feature.Type.BINARY,
                         Feature.Type.NOMINAL,
                         Feature.Type.CONTINUOUS,
                         Feature.Type.NOMINAL)
        for feature, correct_type in zip(self.test_dataset.schema, correct_types):
            self.assertEqual(feature.ftype, correct_type)

    def test_data(self):
        self.assertEqual(len(self.test_dataset), 10)

    def test_missing(self):
        X, _ = self.test_dataset.unzip()
        self.assertTrue(X.iloc[7][3] is np.nan)

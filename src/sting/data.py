import dataclasses
import re
import sys
import typing
import warnings
from abc import abstractmethod
from dataclasses import dataclass, astuple
from enum import Enum, unique, IntEnum
from typing import Optional, Union, Dict, List, Generator, Any, Tuple, Sequence, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, api

__author__ = "Robbie Dozier"
__email__ = "robbied@case.edu"


def nominal_feature_values(feature_name: str, possible_values: Sequence[str]) -> typing.Type[IntEnum]:
    """
    Function for generating an enumeration for nominal feature types. The other option is to manually define an IntEnum
    class. See the documentation for the enum module in the python standard library for more info.

    Example:
        ``Color = nominal_feature_values('Color', ['RED', 'GREEN', 'BLUE'])``

    Args:
        feature_name (str): Name of the feature. It is recommended to use CamelCase, as you are technically defining a
        class here.
        possible_values (Sequence[str]): Possible values the nominal feature can take.

    Returns:
        Type[IntEnum]: An IntEnum class which describes the nominal feature. Each value has an integer assigned to it,
        starting from 1 (*not* 0). For example, if you were to run the above example, ``Color.RED == 1`` would evaluate
        to ``True``.
    """
    return IntEnum(feature_name, possible_values)


@dataclass(repr=True, frozen=True, order=True)
class Feature(object):
    """
    Immutable dataclass representing a feature. Features can be one of the following types: CLASS, ID, BINARY, NOMINAL,
    CONTINUOUS.
    This type is immutable and therefore is hashable. It also supports ordering.

    Fields:
        name (str): Name of the feature.
        ftype (Feature.Type): Type of feature.
        values: For nominal features, the Enum class dictating the possible values the feature can have.

    Examples:
        ``height = Feature(name='height', ftype=Feature.Type.CONTINUOUS)``
        ``color = Feature('color', Feature.Type.NOMINAL, nominal_feature_values('Color', ['RED', 'GREEN', 'BLUE']))``
    """

    @unique
    class Type(Enum):
        """
        Enumerate types of features
        """
        CLASS = 'CLASS'
        INDEX = 'INDEX'
        BINARY = 'BINARY'
        NOMINAL = 'NOMINAL'
        CONTINUOUS = 'CONTINUOUS'

    name: str
    ftype: Type
    values: Optional[typing.Type[Enum]] = None

    def __post_init__(self):
        # Cast to Type instance
        if type(self.ftype) is not Feature.Type:
            super().__setattr__('ftype', Feature.Type[self.ftype])

        # Ensure values is appropriately defined
        if self.values is not None and self.ftype != Feature.Type.NOMINAL:
            raise ValueError('values field should only be defined for NOMINAL features')
        elif self.values is None and self.ftype == Feature.Type.NOMINAL:
            raise ValueError(f'missing values field for NOMINAL feature {self.name}')

    def __hash__(self):
        hash(astuple(self))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Feature):
            return False
        if self.ftype != other.ftype:
            return False

        if self.ftype == Feature.Type.NOMINAL:
            return set(self.values) == set(other.values) and self.name == other.name
        else:
            return self.name == other.name

    def to_float(self, value: Any) -> float:
        """
        Converts a value in some other data type to a float.

        Examples:
            ``binary_feature.to_float(True)  # Would return a 1``

            ``color_nominal_feature.to_float(Color.GREEN)  # Would return a 2, assuming Color is defined "RED = 1,
            GREEN = 2, BLUE = 3``

        Args:
            value: Value to be converted.

        Returns:
            float: The value, represented as a float.
        """
        if self.ftype == Feature.Type.BINARY or self.ftype == Feature.Type.CLASS:
            if value:
                return 1.
            else:
                return 0.
        elif self.values is None:
            return float(value)
        elif isinstance(value, str):
            return float(self.values[value].value)
        elif isinstance(value, int):
            return float(self.values(value).value)
        elif isinstance(value, Enum):
            return float(value.value)
        else:
            raise ValueError(f'Could not convert {self.name} feature value {value} to float')

    def from_float(self, value: float) -> Union[float, Enum, bool, int]:
        """
        Essentially the reverse of ``to_float()``. Takes a float value and attempts to coerce it into a more useful
         type. This method is provided for convenience but it is not recommended that you rely on it.

        Examples:
            ``binary_feature.from_float(1.)  # Would return True``

            ``color_nominal_feature.to_float(2.)  # Would return Color.GREEN, assuming Color is defined "RED = 1,
            GREEN = 2, BLUE = 3``
            enum``

        Args:
            value (float): Float value to be converted.

        Returns:
            The original value for a CONTINUOUS feature; A boolean for a BINARY feature; and int for a CLASS or INDEX
            feature; an Enum (or IntEnum) for a NOMINAL feature.
        """

        if self.ftype == Feature.Type.CONTINUOUS:
            return value
        elif self.ftype == Feature.Type.BINARY:
            return bool(value)
        elif self.ftype == Feature.Type.CLASS or self.ftype == Feature.Type.INDEX:
            return int(value)
        else:
            return self.values(int(value))


class ClassLabelFeature(Feature):
    """Special wrapper of the ``Feature`` class for class labels."""

    def __init__(self, name: str = 'label'):
        super().__init__(name=name, ftype=Feature.Type.CLASS)


class IndexFeature(Feature):
    """Special wrapper of the ``Feature`` class for example indices."""

    def __init__(self, name: str = 'index'):
        super().__init__(name=name, ftype=Feature.Type.INDEX)


class WeightFeature(Feature):
    """Special wrapper of the ``Feature`` class for example weight."""

    def __init__(self, name: str = 'weight'):
        super().__init__(name=name, ftype=Feature.Type.CONTINUOUS)


@dataclass(repr=True, eq=True, frozen=True)
class Example(object):
    """
    Immutable dataclass providing an easy way to work with labeled or unlabeled dataset examples.

    Examples:
        ``example1 = Example({'height': 3.05, 'color': Color.RED, 'is_friendly': False}, 1)``
        ``example2 = Example({'height': 2.1, 'color': Color.BLUE, 'is_friendly': True}, None)``

    Fields:
        features (dict): Dictionary mapping feature names to values.
        label: (Optional[int]): Class label. If None, this is an unlabelled example.
        weight (float): Example weight for boosting, etc.
    """
    features: Dict[str, Union[int, float, str, Enum]]
    index: Optional[int] = dataclasses.field(compare=False, default=None)
    label: Optional[int] = None
    weight: float = 1.

    def __post_init__(self):
        # Assert weight is positive
        if self.weight < 0 or self.weight > 1.:
            raise ValueError('Example weight must be in [0, 1]')

        # If there is an index in features, move to index attribute. If index is already defined, raise a ValueError
        if 'index' in self.features.keys() and self.index is None:
            super().__setattr__('index', self.features['index'])
            del self.features['index']
        elif 'index' in self.features.keys() and self.index is not None:
            raise ValueError('index defined in features and passed as attribute')

        # If there is a label in features, move to label attribute. If label is already defined, raise a ValueError
        if 'label' in self.features.keys() and self.label is None:
            super().__setattr__('label', self.features['label'])
            del self.features['label']
        elif 'label' in self.features.keys() and self.label is not None:
            raise ValueError('label defined in features and passed as attribute')

    def unzip(self, weight=False) -> Union[Tuple[Dict[str, Union[int, float, str, Enum]], Optional[int]],
                                           Tuple[Dict[str, Union[int, float, str, Enum]], Optional[int], float]]:
        """
        "Unpacks" the Example into a tuple.

        Example:
            ``feature_dict, label = ex.unzip()``
            ``feature_dict, label, weight = ex.unzip(True)``

        Args:
            weight: If True, also returns the weight. Leave this False unless you need the weight.

        Returns:
            tuple: Tuple containing the feature dict, the label, and possibly the weight.
        """
        if weight:
            return self.features, self.label, self.weight
        else:
            return self.features, self.label


class AbstractDataSet(Iterable):
    """
    Interface for working with datasets. The ``DataSet`` class has been implemented for you and it is recommended that
    you use it.
    """

    @property
    @abstractmethod
    def schema(self) -> Sequence[Feature]:
        """
        Returns:
            A sequence of ``Feature`` instances describing the schema of the dataset.
        """
        pass

    @abstractmethod
    def unzip(self) -> Tuple[Iterable[Any], Optional[Iterable[int]]]:
        """
        "Unpacks" the dataset into the features and the labels. Use this if you want to bypass this interface and work
        on examples in bulk.

        Example:
            ``X, y = dset.unzip()``

        Returns:
            A tuple containing the features and the labels.
        """
        pass

    @abstractmethod
    def set_weight(self, index: int, weight: float):
        """
        Sets the weight for a certain example in the dataset.

        Args:
            index: Index at which to set the weight
            weight: New weight value
        """
        pass

    @abstractmethod
    def append(self, other: 'AbstractDataSet') -> 'AbstractDataSet':
        """
        Append another dataset to this dataset.

        Returns:
            A new dataset concatenated with this dataset.

        Raises:
            ValueError if schemas do not match.
        """
        pass

    @classmethod
    @abstractmethod
    def concat(cls, *args) -> 'AbstractDataSet':
        """
        Concatenate two or more datasets.
        Args:
            *args: Datasets to concatenate.

        Returns:
            Concatenated datasets.

        Raises:
            ValueError if schemas do not match.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Generator[Example, None, None]:
        """
        Iterate through dataset. Provides compatibility for ``for`` loops.

        Example:
            ``for example in dset:``

        Returns:
            A Generator of Examples.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, slice]) -> Union['AbstractDataSet', Example]:
        """
        Get one or more elements of the dataset, can be accessed with the indexing (`[:]`) operator. If a single value
        is selected, return an Example instance.
        """
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Check dataset equality with another dataset.
        """
        pass


def _infer_schema(df: DataFrame) -> List[Feature]:
    """
    Inspects the values of a DataFrame and infers and returns a schema (list of Feature instances)
    """
    schema = []
    for feature_name in df:
        values = df[feature_name]
        if feature_name == "weight":
            ftype = Feature.Type.CONTINUOUS
        elif feature_name == 'label':
            ftype = Feature.Type.CLASS
        elif api.types.is_string_dtype(values.dtype) or api.types.is_categorical_dtype(values.dtype):
            # Interpret strings as nominal
            ftype = Feature.Type.NOMINAL
        elif api.types.is_bool_dtype(values.dtype):
            # Interpret bools as binary
            ftype = Feature.Type.BINARY
        else:
            # Attempt to use to_numeric()
            try:
                numeric = pd.to_numeric(values)
                if set(numeric.unique()).issubset({0, 1, None, False, True, np.nan, 0., 1.}):
                    # Integers which are only 0 or 1 are binary
                    ftype = Feature.Type.BINARY
                elif set(values.unique().astype(int)) == set(range(1, int(values.max()) + 1)):
                    # Integers which are all values from 1 to some integer are nominal
                    ftype = Feature.Type.NOMINAL
                else:
                    # Default to continuous
                    ftype = Feature.Type.CONTINUOUS
            except ValueError:
                raise TypeError('Unable to resolve schema for feature', feature_name)

        if ftype == Feature.Type.NOMINAL:
            nominal_values = [str(s) for s in sorted(values.unique())]
            feature = Feature(feature_name, ftype, nominal_feature_values(feature_name, nominal_values))
        elif feature_name == 'weight':
            feature = WeightFeature()
        elif ftype == Feature.Type.CLASS:
            feature = ClassLabelFeature()
        else:
            feature = Feature(feature_name, ftype)

        schema.append(feature)

    return schema


def _check_schema(df: DataFrame, schema: Sequence[Feature]):
    """
    Takes a DataFrame and schema (list of Feature instances) and verifies that all values are legal, making corrections
    if necessary.
    """
    for feature in schema:
        if feature.ftype == Feature.Type.CONTINUOUS:
            # Ensure continuous values are numeric
            df[feature.name] = pd.to_numeric(df[feature.name])
        elif feature.ftype == Feature.Type.BINARY:
            # Ensure binary values are boolean
            if not set(df[feature.name].unique()).issubset({0, 1, None, False, True, np.nan, pd.NA, 0., 1.}):
                raise ValueError('Invalid values for boolean encountered in feature', feature.name)
            df[feature.name] = df[feature.name].astype(float).astype('Int64')
        elif feature.ftype == Feature.Type.NOMINAL:
            # Ensure nominal values are categorical
            df[feature.name] = df[feature.name].astype('category')
        elif feature.ftype == Feature.Type.CLASS:
            # Remove class label from schema
            schema = list(filter(lambda x: x.ftype != Feature.Type.CLASS, schema))
        else:
            # This shouldn't happen
            raise ValueError(f'Unexpected feature type {feature.ftype} encountered for feature {feature.name}')

    return schema


class DataSet(AbstractDataSet):
    def __init__(self,
                 data: Optional[Union[DataFrame, Iterable[Dict[str, Any]], Iterable[Example]]] = None,
                 labels: Optional[Iterable[int]] = None,
                 schema: Optional[Sequence[Feature]] = None):
        """
        data can either be:
            - None, in which case an empty dataset is created
            - A pandas DataFrame
            - An Iterable of dicts of the form [{'feature_name': value, ...}, ...]
            - An Iterable of Example instances

        If the schema is not provided, it will be inferred from the values in the table based on the following rules:
            1. Strings and Categorical data will automatically be interpreted as NOMINAL.
            2. Booleans or values that are only either 0 or 1, will be interpreted as BINARY.
            3. Integers that admit all values from 1 to some value will be interpreted as NOMINAL.
            4. Numerical data which does not satisfy 1-3 will be interpreted as CONTINUOUS.

        It is recommended that you always provide a schema to ensure the correct interpretation of values. If data and
        schema are both None, a ValueError will be raised.

        Note that "weight" and "label" are special keys which will be interpreted as the example weight and class label,
        respectively. Values with the feature name "index" will be moved to the index of the DataFrame itself.

        If the labels argument is filled but labels exist in data a ValueError will be raised.
        """

        self._schema = schema
        if data is None:
            self._data = DataFrame()
            if self._schema is None:
                self._schema = []
            return
        elif isinstance(data, DataFrame):
            df = data
        elif isinstance(next(iter(data)), dict):
            # Create dataframe from dict values
            df = DataFrame(data=data)
        elif isinstance(next(iter(data)), Example):
            # Create dataframe from Example values
            df = DataFrame(data=[dict(**ex.features, label=ex.label, weight=ex.weight, index=ex.index) for ex in data])
        else:
            raise ValueError('Invalid data type')

        # Eliminate redundant index
        try:
            df.set_index('index', inplace=True)
        except KeyError:
            pass

        if labels is not None and 'label' in df.columns:
            raise ValueError('Must either provide labels in data or pass them in labels argument, not both')
        elif labels is not None and 'label' not in df.columns:
            df['label'] = labels

        # Initialize weights if they are not present
        if 'weight' not in df.columns:
            df['weight'] = 1.

        if self._schema is None:
            # Infer schema
            self._schema = _infer_schema(df)
        # Check schema and conform data
        self._schema = _check_schema(df, self._schema)

        self._data = df

    def unzip(self) -> Tuple[DataFrame, Optional[Iterable[int]]]:
        """
        "Unpacks" the dataset into the features and the labels. Use this if you want to bypass this interface and work
        on examples in bulk.

        Example:
            ``X, y = dset.unzip()``

        Returns:
            A tuple containing the features (as a Pandas DataFrame) and the labels.
        """

        if 'label' in self._data.columns:
            return self._data.drop('label', axis=1), self._data['label']
        else:
            return self._data, None

    @property
    def schema(self) -> Sequence[Feature]:
        """
        Returns:
            A sequence of ``Feature`` instances describing the schema of the dataset.
        """
        return self._schema

    def set_weight(self, index: int, weight: float):
        """
        Sets the weight for a certain example in the dataset.

        Args:
            index: Index at which to set the weight
            weight: New weight value
        """
        if weight < 0 or weight > 1:
            raise ValueError('weight must be in [0, 1]')
        self._data.at[index, 'weight'] = weight

    def __iter__(self) -> Generator[Example, None, None]:
        """
        Iterate through dataset. Provides compatibility for ``for`` loops.

        Example:
            ``for example in dset:``

        Returns:
            A Generator of Examples.
        """
        for i in range(len(self._data)):
            row = self._data.iloc[i]
            index = row.name
            weight = row.pop('weight')
            try:
                label = row.pop('label')
            except KeyError:
                label = None
            feature_dict = dict(row)
            yield Example(features=feature_dict, index=index, weight=weight, label=label)

    def __len__(self):
        """
        Returns:
            Size of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item: Union[int, slice, list]) -> Union['DataSet', Example]:
        """
        Get one or more elements of the dataset, can be accessed with the indexing (`[:]`) operator. If a single value
        is selected, return an Example instance.
        """
        if isinstance(item, int):
            row = self._data.iloc[item]
            index = row.name
            weight = row.pop('weight')
            try:
                label = row.pop('label')
            except KeyError:
                label = None
            feature_dict = dict(row)
            return Example(features=feature_dict, index=index, weight=weight, label=label)
        elif isinstance(item, slice):
            return DataSet(
                data=self._data.iloc[item],
                schema=self.schema
            )
        elif isinstance(item, list):
            return DataSet(
                data=self._data.iloc[item],
                labels=self._labels[item],
                schema=self.schema
            )
        else:
            raise ValueError('Unexpected index type')

    def drop(self, columns: Optional[Iterable[str]]):
        """
        Drop some column features
        :param columns: columns of features of the dataset to drop
        """
        self._data = self._data.drop(columns=columns)

    def __eq__(self, other):
        if not isinstance(other, DataSet):
            return False
        return self._schema == other._schema and self._data == other._data

    def append(self, other: 'DataSet') -> 'DataSet':
        """
        Append another dataset to this dataset.

        Returns:
            A new dataset concatenated with this dataset.

        Raises:
            ValueError if schemas do not match.
        """
        if not np.array_equal(other._schema, self._schema):
            raise ValueError('Dataset schemas do not match, cannot append')

        return DataSet(
            data=self._data.append(other._data),
            schema=self.schema
        )

    @classmethod
    def concat(cls, *args: 'DataSet') -> 'DataSet':
        """
        Concatenate two or more datasets.
        Args:
            *args: Datasets to concatenate.

        Returns:
            Concatenated datasets.

        Raises:
            ValueError if schemas do not match.
        """
        # TODO: This can be optimized
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0].append(args[1])
        else:
            return DataSet.concat(args[0].append(args[1]), *args[2:])


# Beyond here is code for parsing C45 data

_NAMES_EXT = '.names'
_DATA_EXT = '.data'

_COMMENT_RE = '#.*'
_BINARY_RE = '\\s*0\\s*,\\s*1\\s*'


def parse_c45(file_base, rootdir='.') -> DataSet:
    """
    Returns an ExampleSet from the parsed C4.5-formatted data file

    Arguments:
    file_base -- basename of the file, as in 'file_base.names'
    rootdir   -- root of directory tree to search for files

    """
    schema_name = file_base + _NAMES_EXT
    schema_filename = _find_file(schema_name, rootdir)
    if schema_filename is None:
        raise ValueError('Schema file not found')

    data_name = file_base + _DATA_EXT
    data_filename = _find_file(data_name, rootdir)
    if data_filename is None:
        raise ValueError('Data file not found')

    return _parse_c45(schema_filename, data_filename)


def _parse_c45(schema_filename, data_filename) -> DataSet:
    """Parses C4.5 given file names"""
    try:
        schema = _parse_schema(schema_filename)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise Exception('Error parsing schema: %s' % e)

    try:
        df = _parse_csv(schema, data_filename)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise Exception('Error parsing examples: %s' % e)

    # Remove ID from schema
    schema = list(filter(lambda f: f.ftype != Feature.Type.INDEX, schema))

    return DataSet(data=df, schema=schema)


def _parse_schema(schema_filename) -> List[Feature]:
    """Parses C4.5 '.names' schema file"""
    features = []
    needs_id = True
    with open(schema_filename) as schema_file:
        for line in schema_file:
            feature = _parse_feature(line, needs_id)
            if feature is not None:
                if needs_id and feature.ftype == Feature.Type.INDEX:
                    needs_id = False
                features.append(feature)

    # Fix the problem that the class feature is listed first in the
    # '.names' file, but is the last feature in the '.data' file
    try:
        features.remove(ClassLabelFeature())
    except:
        raise Exception('File does not contain "Class" line')
    features.append(ClassLabelFeature())

    return features


def _parse_feature(line, needs_id) -> Optional[Feature]:
    """
    Parse a feature from the given line. The second argument
    indicates whether we need an ID for our schema, in which
    case the first non-CLASS feature is selected.

    """
    line = _trim_line(line)
    if len(line) == 0:
        # Blank line
        return None
    if re.match(_BINARY_RE, line) is not None:
        # Class feature
        return ClassLabelFeature()
    colon = line.find(':')
    if colon < 0:
        raise Exception('No feature name found.')
    name = line[:colon].strip()
    remainder = line[colon + 1:]
    values = _parse_values(remainder)
    if needs_id:
        return IndexFeature()
    elif len(values) == 1 and values[0].startswith('continuous'):
        return Feature(name, Feature.Type.CONTINUOUS)
    elif len(values) == 2 and '0' in values and '1' in values:
        return Feature(name, Feature.Type.BINARY)
    else:
        return Feature(name, Feature.Type.NOMINAL, nominal_feature_values(name, values))


def _parse_values(value_string):
    """Parse comma-delimited values from a string"""
    values = list()
    for raw in value_string.split(','):
        raw = raw.strip()
        if len(raw) > 1 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1].strip()
        values.append(raw)
    return values


def _parse_csv(schema, data_filename) -> pd.DataFrame:
    """Parse examples from a '.data' file given a schema using pandas"""

    df = pd.read_csv(data_filename, header=None, comment='#').replace(r'"|\s+', '', regex=True)
    df.columns = map(lambda f: f.name, schema)
    df.replace('?', np.nan, inplace=True)

    # Conform to schema
    for col, feature in zip(df, schema):
        if feature.ftype == Feature.Type.CLASS or feature.ftype == Feature.Type.INDEX:
            # Cast to int
            df[col] = df[col].astype(int)
        elif feature.ftype == Feature.Type.NOMINAL:
            # Cast to categorical
            df[col] = df[col].astype('category')
        elif feature.ftype == Feature.Type.BINARY:
            # Cast to bool
            df[col] = df[col].astype(float).astype('Int64')
        else:
            # Cast to float
            df[col] = df[col].astype(float)

    df.set_index('index', inplace=True, drop=True)

    return df


def _parse_examples(schema, data_filename) -> List[Dict[str, Any]]:
    """Parse examples from a '.data' file given a schema"""
    examples = []
    with open(data_filename) as data_file:
        for line in data_file:
            line = _trim_line(line)
            if len(line) == 0:
                # Skip blank line
                continue
            try:
                ex = _parse_example(schema, line)
                examples.append(ex)
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                warnings.warn('C45 parser warning: skipping line: "%s"' % line)

    return examples


def _parse_example(schema, line) -> Dict[str, Any]:
    """Parse a single example from the line of a data file"""
    values = _parse_values(line)
    if len(values) != len(schema):
        raise Exception('Feature-data size mismatch: %s' % line)

    ex = {}
    for i, value in enumerate(values):
        if value == '?':
            # Unknown value remains 'None'
            continue

        # Cast to proper type
        feature = schema[i]
        if feature.ftype == Feature.Type.INDEX or feature.ftype == Feature.Type.NOMINAL:
            ex[feature.name] = value
        elif feature.ftype == Feature.Type.BINARY or feature.ftype == Feature.Type.CLASS:
            ex[feature.name] = bool(int(value))
        elif feature.ftype == Feature.Type.CONTINUOUS:
            ex[feature.name] = float(value)
        else:
            raise ValueError('Unknown schema type "%s"' % feature.ftype)

    return ex


def _trim_line(line):
    """Removes comments and periods from the given line"""
    line = re.sub(_COMMENT_RE, '', line)
    line = line.strip()
    if len(line) > 0 and line[-1] == '.':
        line = line[:-1].strip()
    return line


def _find_file(filename, rootdir):
    """
    Finds a file with filename located in some
    subdirectory of the root directory
    """
    import os
    for dirpath, _, filenames in os.walk(os.path.expanduser(rootdir)):
        if filename in filenames:
            return os.path.join(dirpath, filename)

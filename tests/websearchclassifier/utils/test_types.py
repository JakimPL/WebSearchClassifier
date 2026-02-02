from typing import Union

import numpy as np

from websearchclassifier.utils.types import does_belong_to_union, is_bool, is_float, is_integer, is_string


class TestDoesBelongToUnion:
    """Tests for does_belong_to_union function."""

    def test_bool_with_bool_type(self) -> None:
        BoolUnion = Union[bool, np.bool_]
        assert does_belong_to_union(True, BoolUnion)
        assert does_belong_to_union(False, BoolUnion)
        assert does_belong_to_union(np.bool_(True), BoolUnion)
        assert does_belong_to_union(np.bool_(False), BoolUnion)

    def test_integer_with_integer_type(self) -> None:
        IntegerUnion = Union[int, np.integer]
        assert does_belong_to_union(42, IntegerUnion)
        assert does_belong_to_union(0, IntegerUnion)
        assert does_belong_to_union(-10, IntegerUnion)
        assert does_belong_to_union(np.int8(5), IntegerUnion)
        assert does_belong_to_union(np.int16(5), IntegerUnion)
        assert does_belong_to_union(np.int32(5), IntegerUnion)
        assert does_belong_to_union(np.int64(5), IntegerUnion)
        assert does_belong_to_union(np.uint8(5), IntegerUnion)

    def test_float_with_float_type(self) -> None:
        FloatUnion = Union[float, np.floating]
        assert does_belong_to_union(3.14, FloatUnion)
        assert does_belong_to_union(0.0, FloatUnion)
        assert does_belong_to_union(-1.5, FloatUnion)
        assert does_belong_to_union(np.float32(3.14), FloatUnion)
        assert does_belong_to_union(np.float64(3.14), FloatUnion)

    def test_string_with_string_type(self) -> None:
        StringUnion = Union[str, np.str_]
        assert does_belong_to_union("hello", StringUnion)
        assert does_belong_to_union("", StringUnion)
        assert does_belong_to_union(np.str_("hello"), StringUnion)

    def test_wrong_type_returns_false(self) -> None:
        IntegerUnion = Union[int, np.integer]
        StringUnion = Union[str, np.str_]
        BoolUnion = Union[bool, np.bool_]
        assert not does_belong_to_union("string", IntegerUnion)
        assert not does_belong_to_union(42, StringUnion)
        assert not does_belong_to_union(3.14, BoolUnion)
        assert not does_belong_to_union([1, 2, 3], IntegerUnion)
        assert not does_belong_to_union(None, StringUnion)


class TestIsBool:
    """Tests for is_bool function."""

    def test_native_bool_returns_true(self) -> None:
        assert is_bool(True)
        assert is_bool(False)

    def test_numpy_bool_returns_true(self) -> None:
        assert is_bool(np.bool_(True))
        assert is_bool(np.bool_(False))
        assert is_bool(np.bool_(1))
        assert is_bool(np.bool_(0))

    def test_non_bool_returns_false(self) -> None:
        assert not is_bool(1)
        assert not is_bool(0)
        assert not is_bool(1.0)
        assert not is_bool("True")
        assert not is_bool(None)
        assert not is_bool([])
        assert not is_bool({})
        assert not is_bool(np.int32(1))
        assert not is_bool(np.float64(1.0))


class TestIsInteger:
    """Tests for is_integer function."""

    def test_native_int_returns_true(self) -> None:
        assert is_integer(0)
        assert is_integer(42)
        assert is_integer(-10)
        assert is_integer(1000000)
        assert is_integer(True)
        assert is_integer(False)

    def test_numpy_integer_returns_true(self) -> None:
        assert is_integer(np.int8(5))
        assert is_integer(np.int16(100))
        assert is_integer(np.int32(1000))
        assert is_integer(np.int64(10000))
        assert is_integer(np.uint8(5))
        assert is_integer(np.uint16(100))
        assert is_integer(np.uint32(1000))
        assert is_integer(np.uint64(10000))

    def test_non_integer_returns_false(self) -> None:
        assert not is_integer(3.14)
        assert not is_integer("42")
        assert not is_integer(None)
        assert not is_integer([1, 2, 3])
        assert not is_integer(np.float32(5.0))
        assert not is_integer(np.bool_(True))


class TestIsFloat:
    """Tests for is_float function."""

    def test_native_float_returns_true(self) -> None:
        assert is_float(3.14)
        assert is_float(0.0)
        assert is_float(-1.5)
        assert is_float(1e10)
        assert is_float(float("inf"))
        assert is_float(float("-inf"))

    def test_numpy_float_returns_true(self) -> None:
        assert is_float(np.float16(3.14))
        assert is_float(np.float32(3.14))
        assert is_float(np.float64(3.14))

    def test_nan_returns_true(self) -> None:
        assert is_float(float("nan"))
        assert is_float(np.float32("nan"))

    def test_non_float_returns_false(self) -> None:
        assert not is_float(42)
        assert not is_float(True)
        assert not is_float("3.14")
        assert not is_float(None)
        assert not is_float([3.14])
        assert not is_float(np.int32(5))
        assert not is_float(np.bool_(True))


class TestIsString:
    """Tests for is_string function."""

    def test_native_string_returns_true(self) -> None:
        assert is_string("hello")
        assert is_string("")
        assert is_string("hello world")
        assert is_string("123")
        assert is_string("True")

    def test_numpy_string_returns_true(self) -> None:
        assert is_string(np.str_("hello"))
        assert is_string(np.str_(""))
        assert is_string(np.str_("test string"))

    def test_non_string_returns_false(self) -> None:
        assert not is_string(42)
        assert not is_string(3.14)
        assert not is_string(True)
        assert not is_string(None)
        assert not is_string([])
        assert not is_string({})
        assert not is_string(b"bytes")
        assert not is_string(np.int32(5))
        assert not is_string(np.float64(3.14))

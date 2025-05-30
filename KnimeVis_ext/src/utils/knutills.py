import knime.extension as knext
import logging
import os

LOGGER = logging.getLogger(__name__)

# pre-define values factory strings
# link to all supported ValueFactoryStrings
# https://github.com/knime/knime-python/blob/49aeaba3819edd635519641c81cc7f9541cf090e/org.knime.python3.arrow.types/plugin.xml

ZONED_DATE_TIME_ZONE_VALUE = "org.knime.core.data.v2.time.ZonedDateTimeValueFactory2"
LOCAL_TIME_VALUE = "org.knime.core.data.v2.time.LocalTimeValueFactory"
LOCAL_DATE_VALUE = "org.knime.core.data.v2.time.LocalDateValueFactory"
LOCAL_DATE_TIME_VALUE = "org.knime.core.data.v2.time.LocalDateTimeValueFactory"

PNG_IMAGE_VALUE = "org.knime.core.data.image.png.PNGImageValueFactory"


PATH_VALUE = "org.knime.filehandling.core.data.location.FSLocationValue"
# SVG_IMAGE_VALUE = ""


def is_zoned_datetime(column: knext.Column) -> bool:
    """
    Checks if date&time column has the timezone or not.
    @return: True if selected date&time column has time zone
    """
    return __is_type_x(column, ZONED_DATE_TIME_ZONE_VALUE)


def is_datetime(column: knext.Column) -> bool:
    """
    Checks if a column is of type Date&Time.
    @return: True if selected column is of type date&time
    """
    return __is_type_x(column, LOCAL_DATE_TIME_VALUE)


def is_time(column: knext.Column) -> bool:
    """
    Checks if a column is of type Time only.
    @return: True if selected column has only time.
    """
    return __is_type_x(column, LOCAL_TIME_VALUE)


def is_date(column: knext.Column) -> bool:
    """
    Checks if a column is of type date only.
    @return: True if selected column has date only.
    """
    return __is_type_x(column, LOCAL_DATE_VALUE)


def boolean_or(*functions):
    """
    Return True if any of the given functions returns True
    @return: True if any of the functions returns True
    """

    def new_function(*args, **kwargs):
        return any(f(*args, **kwargs) for f in functions)

    return new_function


def is_type_timestamp(column: knext.Column):
    """
    This function checks on all the supported timestamp columns in KNIME.
    Note that legacy date&time types are not supported.
    @return: True if timestamp column is compatible with the respective logical types supported in KNIME.
    """

    return boolean_or(is_time, is_date, is_datetime, is_zoned_datetime)(column)


def __is_type_x(column: knext.Column, type: str) -> bool:
    """
    Checks if column contains the given type
    @return: True if Column Type is of that type
    """
    # if isinstance(column.ktype, knext.LogicalType):
    #     LOGGER.warn(column.ktype.logical_type)
    return (
        isinstance(column.ktype, knext.LogicalType)
        and type in column.ktype.logical_type
    )


def is_string(column: knext.Column) -> bool:

    return column.ktype == knext.string()

def is_path(column: knext.Column) -> bool:
    """
    Checks if the column is of type Path (knext.Path()).
    """
    return __is_type_x(column, PATH_VALUE)

def is_numeric(column: knext.Column) -> bool:
    """
    Checks if column is numeric e.g. int, long or double.
    @return: True if Column is numeric
    """
    return (
        column.ktype == knext.double()
        or column.ktype == knext.int32()
        or column.ktype == knext.int64()
    )


def is_boolean(column: knext.Column) -> bool:
    """
    Checks if column is boolean
    @return: True if Column is boolean
    """
    return column.ktype == knext.boolean()


def is_numeric_or_string(column: knext.Column) -> bool:
    """
    Checks if column is numeric or string
    @return: True if Column is numeric or string
    """
    return boolean_or(is_numeric, is_string)(column)


def is_int_or_string(column: knext.Column) -> bool:
    """
    Checks if column is int or string
    @return: True if Column is numeric or string
    """
    return column.ktype in [
        knext.int32(),
        knext.int64(),
        knext.string(),
    ]


def is_binary(column: knext.Column) -> bool:
    """
    Checks if column is of binary object
    @return: True if Column is binary object
    """
    return column.ktype == knext.blob()


def is_png(column: knext.Column) -> bool:
    """
    Checks if column contains PNG image
    @return: True if Column is image
    """
    return __is_type_x(column, PNG_IMAGE_VALUE)



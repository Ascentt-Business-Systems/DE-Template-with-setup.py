"""
    See https://stackoverflow.com/a/42151923.
"""

import base64
import hashlib
import json
from datetime import date, datetime
from typing import Any, Tuple


def get_hash_str(
    obj2hash: Any,
    skip: Tuple[str] = (),
    encoding: str = "utf-8",
    sha512: bool = False,
    use_repr: bool = False,
) -> str:
    """Get hash string from the given object.

    Args:
        obj2hash (Any): Object to get a hash string.
        skip (Tuple[str], optional): Keys to skip when hashing the given object only if it's a dictionary.  # noqa: E501
        encoding (str, optional): Encoding type. Defaults to "utf-8".
        sha512 (bool, optional): Flag to use sha512 instead of sha256. Defaults to False.
        use_repr (bool, optional): Flag to use repr() instead of json.dumps(). Defaults to False.

    Returns:
        str: Hash string from the given object value.
    """
    hashable = make_hashable(obj2hash=obj2hash, skip=skip)
    encoded = hash_stringifier(hashable=hashable, encoding=encoding, use_repr=use_repr)
    hasher = hashlib.sha256 if sha512 is False else hashlib.sha512
    hex_str = hasher(encoded).hexdigest()
    b64_hash = base64.urlsafe_b64encode(bytes.fromhex(hex_str)).decode()
    return b64_hash


def hash_stringifier(hashable: Tuple, encoding: str = "utf-8", use_repr: bool = False) -> str:
    """Stringify the hashed object

    Args:
        hashable (Tuple): Hashable object in tuple.
        encoding (str, optional): Encoding type. Defaults to "utf-8".
        use_repr (bool, optional): Flag to use repr() instead of json.dumps(). Defaults to False.

    Returns:
        str: _description_
    """
    if use_repr is True:
        stringified = repr(hashable)
    else:
        stringified = json.dumps(
            hashable,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            indent=None,
            skipkeys=True,  # Only safe for native python types: str, int, list, etc.
        )
    return stringified.encode(encoding)


def make_hashable(obj2hash: Any, skip: Tuple[str] = ()) -> Tuple:
    """Make the given object hashable.

    Args:
        obj2hash (Any): Object to hash.

    Returns:
        Tuple: Hashed object in a tuple.
    """
    if isinstance(obj2hash, (tuple, list)):
        return tuple(sorted(make_hashable(obj2hash=e, skip=skip) for e in obj2hash))
    elif isinstance(obj2hash, dict):
        return tuple(
            sorted(
                (k, make_hashable(obj2hash=v, skip=skip))
                for k, v in obj2hash.items()
                if k not in skip
            )
        )
    elif isinstance(obj2hash, (set, frozenset)):
        return tuple(sorted(make_hashable(obj2hash=e, skip=skip) for e in obj2hash))
    elif isinstance(obj2hash, datetime):
        return obj2hash.isoformat()
    elif isinstance(obj2hash, date):
        return obj2hash.isoformat()
    return obj2hash

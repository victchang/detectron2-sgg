# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import copy
from typing import Any, Dict, List, Tuple, Union
from detectron2.structures import Instances
import torch


class SGInstances(Instances):
    """
    Override detectron2.structures.Instances to store relations.
    """
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        self._rel_fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set_rel_fields(self, name: str, value: Any) -> None:
        """
        Set the rel field named `name` to `value`.
        """
        self._rel_fields[name] = value

    def get_rel_fields(self, name: str) -> Any:
        """
        Returns the rel field called `name`.
        """
        return self._rel_fields[name]

    def set_rel_fields_from_dict(self, dict: Dict[str, Any]) -> None:
        """
        Set the rel fields from dict.
        """
        self._rel_fields = copy.deepcopy(dict)

    def get_rel_fields_dict(self) -> Any:
        """
        Returns the rel field dict
        """
        return self._rel_fields

    def to_instances(self) -> "Instances":
        """
        Returns:
            instances:
        """
        ret = Instances(self._image_size, **self._fields)
        return ret

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "SGInstances":
        """
        Returns:
            SGInstances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = SGInstances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        for k, v in self._rel_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set_rel_fields(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "SGInstances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `SGInstances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("SGInstances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = SGInstances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        ret.set_rel_fields_from_dict(self._rel_fields)
        return ret
    
    @staticmethod
    def cat(instance_lists: List["SGInstances"]) -> "SGInstances":
        """
        Args:
            instance_lists (list[SGInstances])

        Returns:
            SGInstances
        """
        assert all(isinstance(i, SGInstances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = SGInstances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)

        ret.set_rel_fields_from_dict(instance_lists[0]._rel_fields)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}], ".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        s += "rel_fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._rel_fields.items())))
        return s

    __repr__ = __str__
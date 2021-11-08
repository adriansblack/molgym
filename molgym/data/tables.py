from typing import Sequence, Iterable, Tuple, Union


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'{self.__class__.__name__}: {tuple(s for s in self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


DiscreteBag = Tuple[int, ...]
ContinuousBag = Tuple[float, ...]
Bag = Union[DiscreteBag, ContinuousBag]


def discrete_bag_from_atomic_numbers(zs: Iterable[int], z_table: AtomicNumberTable) -> DiscreteBag:
    bag = [0] * len(z_table)
    for z in zs:
        bag[z_table.z_to_index(z)] += 1

    return tuple(bag)


def remove_element_from_bag(e: int, bag: DiscreteBag) -> DiscreteBag:
    if bag[e] < 1:
        raise ValueError(f"Cannot remove element with index '{e}' from '{bag}'")

    copy = list(bag)
    copy[e] -= 1
    return tuple(copy)


def bag_is_empty(bag: DiscreteBag) -> bool:
    return sum(bag) == 0

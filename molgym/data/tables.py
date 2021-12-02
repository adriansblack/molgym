from typing import Sequence, Iterable, Tuple


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        assert zs[0] == 0
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'{self.__class__.__name__}: {tuple(self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


Bag = Tuple[int, ...]


def bag_from_atomic_numbers(zs: Iterable[int], z_table: AtomicNumberTable) -> Bag:
    bag = [0] * len(z_table)
    for z in zs:
        bag[z_table.z_to_index(z)] += 1

    return tuple(bag)


def remove_element_from_bag(e: int, bag: Bag) -> Bag:
    if bag[e] < 1:
        raise ValueError(f"Cannot remove element with index '{e}' from '{bag}'")

    copy = list(bag)
    copy[e] -= 1
    return tuple(copy)


def add_element_to_bag(e: int, bag: Bag) -> Bag:
    copy = list(bag)
    copy[e] += 1
    return tuple(copy)


def bag_is_empty(bag: Bag) -> bool:
    return all(item < 1 for item in bag)


def no_real_atoms_in_bag(bag: Bag) -> bool:
    return bag_is_empty(bag[1:])

from typing import List

from easycv.manager import ManagerMixin


AVAILABLE_KEY = [
    'a', 'g', 'm', 'd', 'e', 'o', 'z']


class PressKeyManager(ManagerMixin):
    def __init__(self,
                 name: str,
                 num_states: int,
                 init_state: int,
                 custom_state_name: List[str],
                 use_auto_state: bool = False) -> None:
        assert name in AVAILABLE_KEY, f"{name} is not available!"
        super().__init__(name)

        if isinstance(init_state, int):
            self._state = init_state % num_states
        else:
            raise TypeError('init_state must be a int, but got'
                            f' {type(init_state)}')
        self._num_states = num_states
        self._custom_state_name = custom_state_name
        # auto state
        self.use_auto_state = use_auto_state
        self.auto_state = None

    @property
    def is_auto_state(self) -> bool:
        return self.use_auto_state and self._state == self._num_states - 1

    @property
    def state(self) -> int:
        if self.use_auto_state and self._state == self._num_states - 1:
            return self.auto_state
        return self._state

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def custom_state_name(self) -> List[str]:
        return self._custom_state_name

    @classmethod
    def created_key(self) -> List[str]:
        created_key = []
        for key in AVAILABLE_KEY:
            if PressKeyManager.check_instance_created(key):
                created_key.append(key)
        return created_key

    def press_key(self) -> None:
        self._state = (self._state + 1) % self._num_states

    def get_state_name(self) -> str:
        return self._custom_state_name[self.state]

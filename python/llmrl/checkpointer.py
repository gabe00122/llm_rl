from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from flax.nnx.filterlib import Filter
from jax.sharding import Mesh


class Checkpointer:
    def __init__(self, directory: str):
        if not directory.startswith("gs://"):
            directory = Path(directory).absolute().as_posix()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model: object, global_step: int, param_filter: Filter = nnx.Param):
        state = nnx.state(model, param_filter)
        self.mngr.save(global_step, args=ocp.args.StandardSave(state))

    def restore(self, model: object, step: int, param_filter: Filter = nnx.Param):
        device = jax.devices()[0]
        mesh = Mesh((device,), ("batch",))

        target_state = nnx.state(model, param_filter)
        abstract_state = jax.tree.map(
            lambda x, s: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=s),
            target_state,
            nnx.get_named_sharding(target_state, mesh),
        )

        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )

        nnx.update(model, restored_state)

    def restore_latest(self, model: object, param_filter: Filter = nnx.Param) -> int:
        step = self.mngr.latest_step()
        if step is None:
            return 0
        self.restore(model, step, param_filter)
        return step

    def close(self):
        self.mngr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

"""
REST module for interacting with RPL Tag Camera Engine.
"""

from pathlib import Path

from fastapi.datastructures import State
from rpl_tag_engine_driver.rpl_tag_engine_driver import RPLTagEngine
from wei.modules.rest_module import RESTModule
from wei.types.module_types import ModuleState, ModuleStatus
from wei.types.step_types import ActionRequest, StepResponse
from wei.utils import extract_version

rest_module = RESTModule(
    name="rpl_tag_engine_node",
    version=extract_version(Path(__file__).parent.parent / "pyproject.toml"),
    description="A node to control RPL's camera tag engine.",
)

rest_module.arg_parser.add_argument(
    "--host",
    type=str,
    default="146.137.240.84",
    help="IP address to connect to camera.",
)

rest_module.arg_parser.add_argument(
    "--database",
    type=str,
    default="RPLtag.db",
    help="Database filename",
)

rest_module.arg_parser.add_argument(
    "--camera",
    type=str,
    default="CALIB_Logi_RPL_20240508/",
    help="Camera calibration filename or directory",
)

rest_module.arg_parser.add_argument(
    "--verbose",
    type=bool,
    default=True,
    help="Increase output verbosity",
)


@rest_module.startup()
def camera_startup(state: State):
    """Camera startup handler."""
    state.camera = None
    state.camera = RPLTagEngine(database=state.database, camera=state.camera, verbose=state.verbose)
    print("MIR Base online")


@rest_module.state_handler()  # ** TBD, Need to check in person.
def state(state: State):
    """Returns the current state of the camera module"""
    if state.status not in [
        ModuleStatus.ERROR,
        None,
    ]:
        stat = state.camera.status
        if stat == "BUSY":
            state.status = ModuleStatus.BUSY
        elif stat == "IDLE":
            state.status = ModuleStatus.IDLE
        state.status = ModuleStatus.IDLE
    return ModuleState(status=state.status, error="")


@rest_module.action(
    name="run_camera",
    description="Run the RPLTagEngine and return serialized detection",
)
def run_camera(
    state: State,
    action: ActionRequest,
) -> StepResponse:
    """Run the RPLTagEngine and return serialized detection"""
    state.camera.run_camera()
    return StepResponse.step_succeeded()


if __name__ == "__main__":
    rest_module.start()

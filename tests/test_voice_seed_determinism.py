import json
import os
import subprocess
import sys

from tau2.user_simulation_voice_presets import stable_task_seed


def test_stable_task_seed_is_repeatable_in_process():
    assert stable_task_seed(300, "retail-task-7") == stable_task_seed(
        300, "retail-task-7"
    )
    assert stable_task_seed(300, "retail-task-7") != stable_task_seed(
        300, "retail-task-8"
    )


def test_stable_task_seed_is_repeatable_across_processes():
    code = (
        "import json;"
        "from tau2.user_simulation_voice_presets import stable_task_seed;"
        "print(json.dumps([stable_task_seed(300, str(i)) for i in [0,1,7,8,9]]))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    first = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    second = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)

    assert json.loads(first) == json.loads(second)

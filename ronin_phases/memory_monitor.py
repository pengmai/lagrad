import subprocess
import pathlib
from dataclasses import dataclass

MONITOR_BIN = (
    pathlib.Path(__file__).parents[1] / "memory_monitor" / "memory_monitor.out"
)


def run_safe(args, stdin: bytes = None, suppress_stderr=False, env=None):
    try:
        p = subprocess.run(args, input=stdin, check=True, capture_output=True, env=env)
        if p.stderr and not suppress_stderr:
            print(p.stderr.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode("utf-8"))
        raise Exception(e.stderr.decode("utf-8"))
    return p.stdout


@dataclass
class MemUsageResult:
    max_rss: int
    max_vsize: int


def run_with_memory(binary: str, passloc: str, timeout=None):
    p = subprocess.Popen(
        [binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )
    with open(passloc, "rb") as f:
        passwd = f.read()
    usage = run_safe(["sudo", "-S", MONITOR_BIN, str(p.pid)], stdin=passwd).decode(
        "utf-8"
    )
    usage_dict = {
        kv.split(":")[0].strip(): int(kv.split(":")[1].strip())
        for kv in usage.split(",")
    }
    p.wait()
    assert (
        p.returncode == 0
    ), f"Process exited with nonzero exit. stdout: {p.stdout.read()} stderr: {p.stderr.read()}"
    stdout = p.stdout.read()

    usage = MemUsageResult(
        max_rss=usage_dict["max rss"], max_vsize=usage_dict["max vsize"]
    )
    return stdout.decode("utf-8"), usage

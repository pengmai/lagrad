import asyncio
import sys
import psutil

# import pathlib
import subprocess


async def main():
    p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    psprocess = psutil.Process(p.pid)
    counter = 0
    max_rss = 0
    while p.poll() == None:
        try:
            rss = psprocess.memory_info().rss
        except Exception:
            break
        if rss > max_rss:
            max_rss = rss
        # counter += 1
        # if counter == 100:
        #     print("rss:", psprocess.memory_info().rss)
        #     counter = 0
        await asyncio.sleep(0.05)
    print(p.stdout.read().decode("utf-8"))
    print(f"Max RSS: {max_rss}")

    assert (
        p.returncode == 0
    ), f"Process exited with nonzero exit. stdout: {p.stdout.read()} stderr: {p.stderr.read()}"


if __name__ == "__main__":
    asyncio.run(main())

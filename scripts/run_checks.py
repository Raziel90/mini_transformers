import subprocess


def check_code() -> None:
    print("Running Black...")
    subprocess.run(["black", "."], check=True)

    print("Running mypy...")
    subprocess.run(["mypy", "."], check=True)

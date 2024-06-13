from importlib import resources
import mini_transformers
import os


def clear_package_folder(folder_name: str) -> None:
    with resources.path(mini_transformers, folder_name) as folder_path:
        for dirpath, dirnames, files in os.walk(folder_path, topdown=False):
            for file in files:
                print(f"Removing file {file} ...")
                os.remove(os.path.join(dirpath, file))
            for dirname in dirnames:
                print(f"Removing folder {dirname} ...")
                os.rmdir(os.path.join(dirpath, dirname))
        print(f"{dirpath} cleared!")


def clear_logs() -> None:
    clear_package_folder("logs")
    # with resources.path(mini_transformers, "logs") as logs_path:
    #     for dirpath, dirnames, files in os.walk(logs_path, topdown=False):
    #         for file in files:
    #             print(f"Removing file {file} ...")
    #             os.remove(os.path.join(dirpath, file))
    #         for dirname in dirnames:
    #             print(f"Removing folder {dirname} ...")
    #             os.rmdir(os.path.join(dirpath, dirname))
    #     print(f"{dirpath} cleared!")


def clear_checkpoints() -> None:
    clear_package_folder("checkpoints")
    # with resources.path(mini_transformers, "checkpoints") as checkpoint_path:
    #     for dirpath, dirnames, files in os.walk(checkpoint_path, topdown=False):
    #         for file in files:
    #             print(f"Removing file {file} ...")
    #             # os.remove(os.path.join(dirpath, file))
    #         for dirname in dirnames:
    #             print(f"Removing folder {dirname} ...")
    #             print(os.path.exists(os.path.join(dirpath, dirname)))
    #             os.rmdir(os.path.join(dirpath, dirname))
    #     print(f"{dirpath} cleared!")

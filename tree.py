import os

EXCLUDE_DIRS = {"venv", ".venv", "env", "__pycache__", ".git"}

def print_tree(root, prefix=""):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return

    items = [item for item in items if item not in EXCLUDE_DIRS]

    for i, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = i == len(items) - 1

        print(prefix + ("└── " if is_last else "├── ") + item)

        if os.path.isdir(path):
            print_tree(path, prefix + ("    " if is_last else "│   "))

if __name__ == "__main__":
    root = os.getcwd()
    print(os.path.basename(root) or root)
    print_tree(root)





import argparse
from huggingface_hub import create_repo, upload_folder

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--hf-namespace", type=str)
parser.add_argument("--skip-create", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    folder_path = f"./{args.model}"
    repo_id = f"{args.hf_namespace}/{args.model}"

    if not args.skip_create:
        print(f"Making Repo: {repo_id}")

        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=False,
        )

    print(f"Uploading Repo: {repo_id}")
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
    )
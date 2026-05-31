from literature_wiki_graphrag.config import get_settings
from literature_wiki_graphrag.storage import ensure_project_dirs


def main() -> None:
    settings = get_settings()
    ensure_project_dirs(settings.data_dir, settings.raw_responses_dir, settings.output_dir)
    print("LiteratureWikiGraphRAG setup looks ready.")
    print(f"Data directory: {settings.data_dir}")
    print(f"Raw responses: {settings.raw_responses_dir}")
    print(f"Outputs: {settings.output_dir}")


if __name__ == "__main__":
    main()

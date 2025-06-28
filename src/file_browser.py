import os
from pathlib import Path

def list_documents(directory: str, extensions = {'.pdf','.txt'}):
    # List all supported documents in a directory
    path = Path(directory)
    if not path:
        print(f'Directory {directory} does not exist')
        return []
    
    files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    if not files:
        print("No supported documents found")
    else:
        for i, file in enumerate(files, 1):
            print(f'{i}. {file.name}')

    return files

def select_documents(files: list):
    # Prompt user to select documents by index.
    if not files:
        print("No files to select")
        return []
    
    while True:
        indices = input("Enter file numbers to load (comma-separated), or press Enter to select all or 'q' to quit: ").strip()

        if indices in {"q", "quit", "exit"}:
            print("Exiting document selection.")
            return None  # Explicit None to indicate cancellation

        if not indices:
            return files # Return all files if enter is hit
        
        selected = []
        seen = set()

        for token in indices.split(','):
            token = token.strip()

            # Check if input is a number and ignore if non-numeric
            if not token.isdigit():
                print(f'Index {token} is not a valid number')
                continue

            idx = int(token)

            # Check if input within valid range of numbers
            if not (1<= idx <= len(files)):
                print(f'Index {idx} is not within valid range 1 to {len(files)}')
                continue

            # Check duplicates and ignore
            if idx in seen:
                print(f'Index {idx} already selected')
                continue

            seen.add(idx)
            selected.append(files[idx-1])

        if selected:
            return selected
        else:
            print("No valid documents selected. Please try again.\n")


if __name__ == "__main__":
    doc_dir = "data/documents"
    files = list_documents(doc_dir)
    selected = select_documents(files)

    if selected is None:
        print(" Selection canceled. Exiting.")
        exit(0)

    if selected:
        print("\n Selected files:")
        for f in selected:
            print("•", f)
    else:
        print(" No documents selected.")
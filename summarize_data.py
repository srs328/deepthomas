import os
import nibabel as nib


def generate_nifti_manifests(root_dir):
    """Scans and maps MRI data folders."""
    nifti_extensions = (".nii", ".nii.gz")
    for dirpath, _, filenames in os.walk(root_dir):
        # Ignore files inside MONAI bundle directories to keep them separate
        if any(b_dir in dirpath for b_dir in ["model_fold", "algorithm_"]):
            continue

        nifti_files = [f for f in filenames if f.endswith(nifti_extensions)]
        if nifti_files:
            manifest_path = os.path.join(dirpath, ".mri_dataset_manifest.txt")
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(f"MRI Folder Manifest: {os.path.abspath(dirpath)}\n")
                f.write(f"Total NIfTI Files: {len(nifti_files)}\n")
                f.write("=" * 60 + "\n\n")
                for filename in sorted(nifti_files):
                    full_path = os.path.join(dirpath, filename)
                    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    try:
                        img = nib.load(full_path)
                        f.write(f"File: {filename}\n")
                        f.write(f"  - Size: {file_size_mb:.2f} MB\n")
                        f.write(f"  - Dimensions (Shape): {img.shape}\n")
                        f.write(
                            f"  - Voxel Spacing/TR (Zooms): {img.header.get_zooms()}\n"
                        )
                        f.write(f"  - Data Type: {img.header.get_data_dtype()}\n")
                        f.write("-" * 40 + "\n")
                    except Exception as e:
                        f.write(f"File: {filename}\n")
                        f.write(f"  - Size: {file_size_mb:.2f} MB\n")
                        f.write(f"  - [ERROR] Header parse failed: {str(e)}\n")
                        f.write("-" * 40 + "\n")
            print(f"Created MRI manifest in: {dirpath}")


def summarize_monai_bundles(root_dir):
    """Maps MONAI Auto3Dseg bundle structures and lists key configuration/code files."""
    # We target config, scripts, and markdown documentation
    target_extensions = (".yaml", ".json", ".py", ".md")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Identify Auto3Dseg bundle directories (typically contain 'configs' or match MONAI naming)
        if (
            "configs" in dirnames
            or "scripts" in dirnames
            or "algorithm_templates" in dirnames
            or "model_fold" in dirpath
        ):
            manifest_path = os.path.join(dirpath, ".bundle_structure_manifest.txt")

            # Avoid rewriting manifests inside subdirectories recursively
            if os.path.exists(manifest_path):
                continue

            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(
                    f"MONAI Auto3Dseg Bundle Summary: {os.path.basename(dirpath)}\n"
                )
                f.write(f"Path: {os.path.abspath(dirpath)}\n")
                f.write("=" * 60 + "\n\n")

                f.write("--- INTERNAL DIRECTORY TREE ---\n")
                # Generate a clean text tree of the subdirectories
                for sub_root, sub_dirs, sub_files in os.walk(dirpath):
                    level = sub_root.replace(dirpath, "").count(os.sep)
                    indent = " " * 4 * (level)
                    f.write(f"{indent}[D] {os.path.basename(sub_root)}/\n")

                    # Log the key files visually in the tree structure
                    for file in sorted(sub_files):
                        if file.endswith(target_extensions):
                            f.write(f"{indent}    ├── [File] {file}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("--- KEY TRAINING CONFIGURATIONS & SCRIPTS ---\n")

                # List files with their direct relative paths for LLM reference
                for sub_root, _, sub_files in os.walk(dirpath):
                    for file in sorted(sub_files):
                        if file.endswith(target_extensions):
                            rel_path = os.path.relpath(
                                os.path.join(sub_root, file), dirpath
                            )
                            f.write(f"- {rel_path}\n")

            print(f"Created MONAI Bundle summary in: {dirpath}")


if __name__ == "__main__":
    project_root = os.getcwd()

    print("1. Scanning for MRI NIfTI data...")
    generate_nifti_manifests(os.path.join(project_root, "data"))

    print("\n2. Scanning for MONAI Auto3Dseg Bundle directories...")
    summarize_monai_bundles(os.path.join(project_root, "training_work_dirs"))

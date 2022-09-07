import shutil

from quantimage2_backend_common.models import FeaturePreset


def populate_presets():
    preset_name = "PET/CT"

    # Check if preset exists already
    existing_preset = FeaturePreset.find_by_name(preset_name)

    if existing_preset is None:
        print("Populating presets")

        # Copy the PET/CT preset to the right folder
        preset_file_name = "PETCT_all.yaml"
        preset_src_path = f"presets_default/{preset_file_name}"
        preset_dst_path = f"/quantimage2-data/feature-presets/{preset_file_name}"
        shutil.copy(preset_src_path, preset_dst_path)

        preset = FeaturePreset(preset_name, preset_dst_path)
        preset.save_to_db()

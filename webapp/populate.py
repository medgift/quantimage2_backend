import shutil

from quantimage2_backend_common.models import FeaturePreset


def populate_presets():
    preset_name_pet_ct = "PET/CT"
    preset_name_mri_ct = "MRI/CT"
    preset_name_mri_pet = "MRI/PET"
    
    # Check if preset exists already
    existing_preset_pet_ct = FeaturePreset.find_by_name(preset_name_pet_ct)
    existing_preset_mri_ct = FeaturePreset.find_by_name(preset_name_mri_ct)
    existing_preset_mri_pet = FeaturePreset.find_by_name(preset_name_mri_pet)

    if existing_preset_pet_ct is None:
        print("Populating presets")

        # Copy the PET/CT preset to the right folder
        preset_file_name = "PETCT_all.yaml"
        preset_src_path = f"presets_default/{preset_file_name}"
        preset_dst_path = f"/quantimage2-data/feature-presets/{preset_file_name}"
        shutil.copy(preset_src_path, preset_dst_path)

        preset = FeaturePreset(preset_name_pet_ct, preset_dst_path)
        preset.save_to_db()
    
    if existing_preset_mri_ct is None:
        print("Populating presets")

        # Copy the PET/CT preset to the right folder
        preset_file_name = "MRI_CT.yaml"
        preset_src_path = f"presets_default/{preset_file_name}"
        preset_dst_path = f"/quantimage2-data/feature-presets/{preset_file_name}"
        shutil.copy(preset_src_path, preset_dst_path)

        preset = FeaturePreset(preset_name_mri_ct, preset_dst_path)
        preset.save_to_db()
        
    if existing_preset_mri_pet is None:
        print("Populating presets")

        # Copy the PET/CT preset to the right folder
        preset_file_name = "MRI_PET.yaml"
        preset_src_path = f"presets_default/{preset_file_name}"
        preset_dst_path = f"/quantimage2-data/feature-presets/{preset_file_name}"
        shutil.copy(preset_src_path, preset_dst_path)

        preset = FeaturePreset(preset_name_mri_pet, preset_dst_path)
        preset.save_to_db()
        

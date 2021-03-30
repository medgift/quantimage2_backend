from imaginebackend_common.models import (
    Modality,
    ROI,
    FeatureDefinition,
    FeatureValue,
    FeatureExtraction,
)
from more_itertools import first_true

OKAPY_PATIENT_ID_FIELD = "patient"
OKAPY_MODALITY_FIELD = "modality"
OKAPY_ROI_FIELD = "VOI"
OKAPY_FEATURE_NAME_FIELD = "feature_name"
OKAPY_FEATURE_VALUE_FIELD = "feature_value"


def store_features(feature_extraction_task_id, feature_extraction_id, features):

    feature_extraction = FeatureExtraction.find_by_id(feature_extraction_id)

    # Store or retrieve metadata (modalities, ROIs & feature definitions) first
    modalities = list(features[OKAPY_MODALITY_FIELD].unique())
    modality_instances = store_extraction_associations(
        Modality, modalities, feature_extraction.modalities
    )
    feature_extraction.modalities = modality_instances

    rois = list(features[OKAPY_ROI_FIELD].unique())
    roi_instances = store_extraction_associations(ROI, rois, feature_extraction.rois)
    feature_extraction.rois = roi_instances

    feature_names = list(features[OKAPY_FEATURE_NAME_FIELD].unique())
    feature_definition_instances = store_extraction_associations(
        FeatureDefinition, feature_names, feature_extraction.feature_definitions
    )
    feature_extraction.feature_definitions = feature_definition_instances
    feature_extraction.save_to_db()

    modalities_map = {modality.name: modality.id for modality in modality_instances}
    rois_map = {roi.name: roi.id for roi in roi_instances}
    definitions_map = {
        definition.name: definition.id for definition in feature_definition_instances
    }

    # Store feature values
    feature_value_instances = []

    # Build instances to save in bulk
    for idx, row in features.iterrows():
        feature_value_instance = {
            "value": row[OKAPY_FEATURE_VALUE_FIELD],
            "feature_definition_id": definitions_map[row[OKAPY_FEATURE_NAME_FIELD]],
            "feature_extraction_task_id": feature_extraction_task_id,
            "modality_id": modalities_map[row[OKAPY_MODALITY_FIELD]],
            "roi_id": rois_map[row[OKAPY_ROI_FIELD]],
        }

        feature_value_instances.append(feature_value_instance)

    # Batch create the instances
    FeatureValue.save_features_batch(feature_value_instances)
    return feature_value_instances


def store_modalities(modalities):
    modality_instances = []

    for modality in modalities:
        instance, created = Modality.get_or_create(
            criteria={"name": modality,}, defaults={"name": modality}
        )
        modality_instances.append(instance)

    return modality_instances


def store_rois(rois, feature_extraction_id):
    roi_instances = []

    for roi in rois:
        roi_instance, created = ROI.get_or_create(
            criteria={"name": roi,}, defaults={"name": roi}
        )

        # Link ROI to feature extraction
        extraction_definition, created = Fea.get_or_create(
            criteria={
                "feature_extraction_id": feature_extraction_id,
                "roi_id": roi_instance.id,
            },
            defaults={
                "feature_extraction_id": feature_extraction_id,
                "roi_id": roi_instance.id,
            },
        )

        roi_instances.append(instance)

    return roi_instances


def store_extraction_associations(model, names, existing):
    instances = []

    existing_map = {x.name: x for x in existing}

    for name in names:
        if name not in existing_map:
            instance, created = model.get_or_create(
                criteria={"name": name}, defaults={"name": name},
            )
        else:
            instance = existing_map[name]

        instances.append(instance)

    return instances

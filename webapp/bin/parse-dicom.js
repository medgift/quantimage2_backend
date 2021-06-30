const fs = require("fs");
const dicomParser = require("dicom-parser");

// Constants (file separator & DICOM tags)
const FILE_SEPARATOR = ",";
const OUTPUT_SEPARATOR = "\t";

const MODALITIES = {RTSTRUCT: "RTSTRUCT", SEG: "SEG"}

const MODALITY_TAG = "x00080060";

// RTSTRUCT Tags
const STRUCTURE_SET_ROI_SEQUENCE_TAG = "x30060020";
const ROI_NAME_TAG = "x30060026";

// SEG Tags
const SEGMENT_SEQUENCE_TAG = "x00620002";
const SEGMENT_DESCRIPTION_TAG = "x00620006";

// Parse arguments (list of files to parse)
const dicomFiles = process.argv.slice(2)[0].split(",");

let roiList = [];

for (let dicomFile of dicomFiles) {
    // Read the DICOM file into a buffer
    const dicomFileAsBuffer = fs.readFileSync(dicomFile);

    // Parse the byte array to get a DataSet object that has the parsed contents
    const dataSet = dicomParser.parseDicom(dicomFileAsBuffer);

    // Check which fields to check based on the modality
    let rois;
    if (dataSet.string(MODALITY_TAG) == MODALITIES.RTSTRUCT) {
        // Add the ROIs to the global list
        rois = dataSet.elements[STRUCTURE_SET_ROI_SEQUENCE_TAG].items.map((roi) =>
            roi.dataSet.string(ROI_NAME_TAG)
        );
    } else if (dataSet.string(MODALITY_TAG) == MODALITIES.SEG) {
        rois = dataSet.elements[SEGMENT_SEQUENCE_TAG].items.map((roi) =>
            roi.dataSet.string(SEGMENT_DESCRIPTION_TAG)
        )
    } else {
        throw new Error("Unsupported ROI modality")
    }

    roiList = roiList.concat(...rois);
}

// Output the global list to stdout
console.log(roiList.join(OUTPUT_SEPARATOR));
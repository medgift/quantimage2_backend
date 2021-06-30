const fs = require("fs");
const dicomParser = require("dicom-parser");

// Constants (file separator & DICOM tags)
const FILE_SEPARATOR = ",";
const OUTPUT_SEPARATOR = "\t";

const STRUCTURE_SET_ROI_SEQUENCE_TAG = "x30060020";
const ROI_NAME_TAG = "x30060026";
const ROI_CONTOUR_SEQUENCE_TAG = "x30060039";

// Parse arguments (list of files to parse)
const dicomFiles = process.argv.slice(2)[0].split(",");

let roiList = [];

for (let dicomFile of dicomFiles) {
  // Read the DICOM file into a buffer
  const dicomFileAsBuffer = fs.readFileSync(dicomFile);

  // Parse the byte array to get a DataSet object that has the parsed contents
  const dataSet = dicomParser.parseDicom(dicomFileAsBuffer, {
    untilTag: ROI_CONTOUR_SEQUENCE_TAG,
  });

  // Add the ROIs to the global list
  let rois = dataSet.elements.x30060020.items.map((roi) =>
    roi.dataSet.string("x30060026")
  );

  roiList = roiList.concat(...rois);
}

// Output the global list to stdout
console.log(roiList.join(OUTPUT_SEPARATOR));
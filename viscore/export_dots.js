// Version: 02/24/2025

// This script allows you to export dots from Viscore in CoralNet format (CSV).
// Each row in the CSV file contains a "view", which is a dot projected to each image.
// Output CSV file will be saved in the aux_dir/files folder.
// The only required input is the layer name. The script will find the user with the most
// annotations for that layer and export the dots for that user.

// ###################################
// Option 1: 1 Hardcoded layer
// let layers = ['CFN_CD-3_2024-08-21'];
// ###################################

// ###################################
// Option 2: If you want all active layers
let layers = cn.list_layers();
// ###################################

let svc = C.init_svc("svc", "fg");

// Used to access the API
let cn = yield C.get_api("local/c/coralnet:api");

// List all API functions available
C.p("Functions: ", cn.api_list());

// List all layers
C.p("Layers: ", layers);

// Loop over the active layers
for (let j = 0; j < layers.length; j++) {
  // Get the current layer name
  let layerName = layers[j];

  // Output annotations info
  let output = yield C.vpc.info(layerName);
  C.md(`---### Layer \`${layerName}\``);

  // List the users that have created annotations for the specified layer
  let users = yield C.vpc.list_users(layerName);
  C.p("Number of users: " + users.length, users);

  const csv_filename = sys.aux_dir("files/" + layerName + ".csv");
  const have = yield sys.file_exists(csv_filename);
  if (have[0]) {
    C.md(`#### \`SKIPPING\` existing CSV file \`${csv_filename}\``);
    continue;
  }

  // Find the user with the most annotations
  let maxAnnotations = 0;
  let maxUser = "";

  for (let i = 0; i < users.length; i++) {
    try {
      // Get the number of annotations for this user
      let annotations = output[i]["users"][users[i]]["c"];
      C.p("Annotations: ", annotations);

      // Update max user if necessary
      if (annotations > maxAnnotations) {
        maxAnnotations = annotations;
        maxUser = users[i];
      }
    } catch (error) {
      continue;
    }
  }

  // Using Org, provide layer(s), user, and number of dots
  C.p(
    yield cn.export_dots({
      layer: layerName,
      isect_setup: {
        // Find smallest rod radius r (Rmin <= r <= Rmax) that hits a point
        min_d: metric.unscale(0.002), // minimum rod radius Rmin
        d: metric.unscale(0.016), // maximum rod radius Rmax
        err2_thresh: 0.005,
        treefract: 1,
        steps: 3
      },
      max_cams_per_dot: 100,
      user: maxUser
    }),
    "\n"
  );
  C.save();

  // Threshold smaller == tighter, more conservative (see e value in cams)
  C.p(
    cn.export_csv({
      layer: layerName,
      user: maxUser,
      thresh: 1.0,
      fn: csv_filename
    })
  );
  C.save();
}
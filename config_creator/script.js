let jsonOut = {'Video': {}, 'Proc': {}, 'Preproc': {}, 'Postproc': {}};

$(document).ready(function() {
    $('.js-example-basic-multiple').select2();


    displayType();
    displayDetector();
    displayTracker();
    displayPreproc();
    displayPostProc();
    
    generateJSONVideo();
    generateJSONDetector();
    generateJSONTracker();
    generateJSONEvaluator();
    generateJSONPreproc();
    generateJSONPostproc();

    $("input, select").change(function(){
        $("#text_area_json")[0].innerHTML = JSON.stringify(jsonOut, null, 4);
    })

});

function displayType() {
    $("input[name='proc_type']").change(function(){
        jsonOut['Proc'] = {}
        if (this.value === "detector") {
            $(".detector").attr("hidden", false);
            $(".tracker").attr("hidden", true);
            $(".evaluator").attr("hidden", true);
            jsonOut['Proc']['Detector'] = {}
        }
        if (this.value === "tracker") { 
            $(".detector").attr("hidden", true);
            $(".tracker").attr("hidden", false);
            $(".evaluator").attr("hidden", true);
            jsonOut['Proc']['Tracker'] = {}
        }
        if (this.value === "evaluator") { 
            $(".detector").attr("hidden", true);
            $(".tracker").attr("hidden", true);
            $(".evaluator").attr("hidden", false);
            jsonOut['Proc']['Evaluator'] = {}
        }
    })
}

function displayDetector() {
    $("select[name='select_detector_type']").change(function(){
        if ($(this).children(':selected')[0].value !== "") {
            $(".detector_type").attr("hidden", false)
        } else{
            $(".detector_type").attr("hidden", true)
        }
    })

    $("select[name='select_detector_model_type']").change(function(){
        if ($(this).children(':selected')[0].value === "YOLO") {
            $(".yolo").attr("hidden", false)
            $(".detectron2").attr("hidden", true)
            $(".bs").attr("hidden", true)
            $(".needs_model").attr("hidden", false)
            $(".needs_config").attr("hidden", false)
        } else if ($(this).children(':selected')[0].value === "Detectron2") {
            $(".yolo").attr("hidden", true)
            $(".detectron2").attr("hidden", false)
            $(".bs").attr("hidden", true)
            $(".needs_model").attr("hidden", false)
            // $(".needs_config").attr("hidden", false)
        } else if ($(this).children(':selected')[0].value === "BackgroundSubtractor") {
            $(".yolo").attr("hidden", true)
            $(".detectron2").attr("hidden", true)
            $(".bs").attr("hidden", false)
            $(".needs_model").attr("hidden", true)
            $(".needs_config").attr("hidden", true)
        } else{
            $(".yolo").attr("hidden", true)
            $(".detectron2").attr("hidden", true)
            $(".bs").attr("hidden", true)
            $(".needs_model").attr("hidden", true)
            $(".needs_config").attr("hidden", true)
        }
    })

    $("select[name='select_detector_implem']").change(function(){
        val = $(this).children(':selected')[0].value 
        if (val === "cv2-DM" || val === "torch-Ultralytics") {
            $(".yolo_nms").attr("hidden", false)
        }else{
            $(".yolo_nms").attr("hidden", true)
        }
        if (val === "cv2-DM" || val === "cv2-RN" || val === "Default"){
            $(".needs_config").attr("hidden", false)
        } else if (val === "torch-Ultralytics"){
            $(".needs_config").attr("hidden", true)
        }
        if (val === "mean" || val === "median"){
            $(".mean_median").attr("hidden", false)
        }else{
            $(".mean_median").attr("hidden", true)
        }
    })

    $("#yolo_gpu_cb").change(function(){
        det_implem = $("select[name='select_detector_implem']").children(':selected')[0].value
        if (this.checked && det_implem.startsWith("cv2")) {
            $(".yolo_hp").attr("hidden", false)
        } else {
            $(".yolo_hp").attr("hidden", true)
        }
    })
}

function displayTracker() {
    $("select[name='select_tracker_type']").change(function(){
        if ($(this).children(':selected')[0].value !== "") {
            $(".tracker_type").attr("hidden", false)
        } else{
            $(".tracker_type").attr("hidden", true)
        }
    })

    $("select[name='select_tracker_model_type']").change(function(){
        val = $(this).children(':selected')[0].value
        if (val === "Centroid"){
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", true)
            $(".iou").attr("hidden", true)
            $(".centroid").attr("hidden", false)
        } else if (val  === "SORT") {
            $(".sort").attr("hidden", false)
            $(".deepsort").attr("hidden", true)
            $(".iou").attr("hidden", true)
            $(".centroid").attr("hidden", true)
        } else if (val === "DeepSORT") {
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", false)
            $(".iou").attr("hidden", true)
            $(".centroid").attr("hidden", true)
        } else if (val === "IOU") {
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", true)
            $(".iou").attr("hidden", false)
            $(".centroid").attr("hidden", true)
        } else {
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", true)
            $(".centroid").attr("hidden", true)
            $(".iou").attr("hidden", true)
        }
    })

    $("select[name='select_tracker_implem']").change(function(){
        val = $(this).children(':selected')[0].value
        if (val === "KIOU"){
            $(".kiou").attr("hidden", false)
        }else{
            $(".kiou").attr("hidden", true)
        }
    })

    $("#deepsort_avg_conf_cb").change(function(){
        if (this.checked) {
            $(".deepsort_avg_conf").attr("hidden", false)
        } else {
            $(".deepsort_avg_conf").attr("hidden", true)
        }
    })
}

function displayPreproc () {
    $("input[name='ar_type']").change(function(){
        if (this.value === "letterbox") {
            $(".letterbox").attr("hidden", false);
        }
        else {
            $(".letterbox").attr("hidden", true);
        }
    })

    $("input[name='roi_pre_cb']").change(function(){
        if (this.checked) {
            $(".roi_pre").attr("hidden", false);
        }
        else {
            $(".roi_pre").attr("hidden", true);
        }
    })

    $("input[name='roi_pre_type']").change(function(){
        if (this.value === "file") {
            $(".roi_pre_file").attr("hidden", false);
            $(".roi_pre_coords").attr("hidden", true);
        }
        else {
            $(".roi_pre_file").attr("hidden", true);
            $(".roi_pre_coords").attr("hidden", false);
        }
    })
}

function displayPostProc() {
    $("select[name='select_NMS_implem']").change(function(){
        val = $(this).children(':selected')[0].value 
        if (val !== "") {
            $(".post_nms").attr("hidden", false)
        } else{
            $(".post_nms").attr("hidden", true)
        }
    })

    $("input[name='roi_post_cb']").change(function(){
        if (this.checked) {
            $(".roi_post").attr("hidden", false);
        }
        else {
            $(".roi_post").attr("hidden", true);
        }
    })

    $("input[name='roi_post_type']").change(function(){
        if (this.value === "file") {
            $(".roi_post_file").attr("hidden", false);
            $(".roi_post_coords").attr("hidden", true);
        }
        else {
            $(".roi_post_file").attr("hidden", true);
            $(".roi_post_coords").attr("hidden", false);
        }
    })
}

function generateJSONVideo() {
    $("input[name='path']").change(function(){
        jsonOut['Video']['path'] = this.value
    })
    
    $("input[name='async_cb']").change(function(){
        jsonOut['Video']['async'] = this.checked
    })
    $("input[name='frame_interval']").change(function(){
        jsonOut['Video']['frame_interval'] = Number(this.value)
    })
}

function generateJSONDetector() {
    $("select[name='select_detector_type']").change(function(){
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc']['Detector']['type'] = val
            jsonOut['Proc'][val] = {}
        } else {
            delete jsonOut['Proc']['Detector']['type']
            delete jsonOut['Proc'][val]
        }
    })

    $("select[name='select_detector_model_type']").change(function(){
        var type = $("select[name='select_detector_type']").children(':selected')[0].value
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc'][type]['model_type'] = val
            jsonOut['Proc'][val] = {}
        } else {
            delete jsonOut['Proc'][type]['model_type']
            delete jsonOut['Proc'][val]
        }
    })

    $("select[name='select_detector_implem']").change(function(){
        var type = $("select[name='select_detector_type']").children(':selected')[0].value
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc'][type]['pref_implem'] = val
        } else {
            delete jsonOut['Proc'][type]['pref_implem']
        }
    })

    $("input[name='path_detector_model']").change(function(){
        var type = $("select[name='select_detector_type']").children(':selected')[0].value
        if ($(this)[0].value){
            jsonOut['Proc'][type]['model_path'] = $(this)[0].value
        }else{
            delete jsonOut['Proc'][type]['model_path']
        }
    })

    $("input[name='path_config_model']").change(function(){
        var type = $("select[name='select_detector_type']").children(':selected')[0].value
        if ($(this)[0].value){
            jsonOut['Proc'][type]['config_path'] = $(this)[0].value
        }else{
            delete jsonOut['Proc'][type]['config_path']
        }
    })

    // YOLO

    $("input[name='model_input_width'], input[name='model_input_height']").change(function(){
        var type = $("select[name='select_detector_type']").children(':selected')[0].value
        var width = Number($("input[name='model_input_width']")[0].value) || 416
        var height = Number($("input[name='model_input_height']")[0].value) || 416
        jsonOut['Proc'][type]['input_width'] = width
        jsonOut['Proc'][type]['input_height'] = height
    })

    $("input[name='yolo_conf_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['YOLO']['conf_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['YOLO']['conf_thresh']
        }
    })

    $("input[name='yolo_nms_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['YOLO']['nms_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['nms']['nms_thresh']
        }
    })

    $("input[name='yolo_nms_across_classes_cb']").change(function() {
        jsonOut['Proc']['YOLO']['nms_across_classes'] = this.checked
    })

    $("input[name='yolo_gpu_cb']").change(function(){
        jsonOut['Proc']['YOLO']['GPU'] = this.checked
        if (!this.checked){
            $("input[name='yolo_hp_cb']").prop("checked", false);
            jsonOut['Proc']['YOLO']['half_precision'] = false
        }
    })

    $("input[name='yolo_hp_cb']").change(function(){
        jsonOut['Proc']['YOLO']['half_precision'] = this.checked
    })

    // Detectron2

    $("input[name='det2_conf_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Detectron2']['conf_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['YOLO']['conf_thresh']
        }
    })

    $("input[name='det2_nms_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Detectron2']['nms_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['nms']['nms_thresh']
        }
    })

    $("input[name='det2_gpu_cb']").change(function(){
        jsonOut['Proc']['Detectron2']['GPU'] = this.checked
    })

    // Background subtraction

    $("input[name='bs_contour_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Detectron2']['bs_contour_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['Detectron2']['bs_contour_thresh']
        }
    })

    $("input[name='bs_intensity']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Detectron2']['bs_intensity'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['Detectron2']['bs_intensity']
        }
    })

    $("input[name='bs_max_last_images']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Detectron2']['bs_max_last_images'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['Detectron2']['bs_max_last_images']
        }
    })

}

function generateJSONTracker() {
    $("select[name='select_tracker_type']").change(function(){
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc']['Tracker']['type'] = val
            jsonOut['Proc'][val] = {}
        } else {
            delete jsonOut['Proc']['Tracker']['type']
            delete jsonOut['Proc'][val]
        }
    })

    $("select[name='select_tracker_model_type']").change(function(){
        var type = $("select[name='select_tracker_type']").children(':selected')[0].value
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc'][type]['model_type'] = val
            jsonOut['Proc'][val] = {}
        } else {
            delete jsonOut['Proc'][type]['model_type']
            delete jsonOut['Proc'][val]
        }
    })

    $("select[name='select_tracker_implem']").change(function(){
        var type = $("select[name='select_tracker_type']").children(':selected')[0].value
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc'][type]['pref_implem'] = val
        } else {
            delete jsonOut['Proc'][type]['pref_implem']
        }
    })

    // Centroid

    $("input[name='centroid_max_age']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['Centroid']['max_age'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['Centroid']['max_age']
        }
    })

    // IOU

    $("input[name='iou_max_age']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['IOU']['max_age'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['IOU']['max_age']
        }
    })

    $("input[name='iou_min_hits']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['IOU']['min_hits'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['IOU']['min_hits']
        }
    })

    $("input[name='iou_iou_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['IOU']['iou_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['IOU']['iou_thresh']
        }
    })

    // SORT

    $("input[name='sort_max_age']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['SORT']['max_age'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['SORT']['max_age']
        }
    })

    $("input[name='sort_min_hits']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['SORT']['min_hits'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['SORT']['min_hits']
        }
    })

    $("input[name='sort_iou_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['SORT']['iou_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['SORT']['iou_thresh']
        }
    })

    $("input[name='sort_memory_fade']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['SORT']['memory_fade'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['SORT']['memory_fade']
        }
    })

    // DeepSORT

    $("input[name='deepsort_tracker_model']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['model_path'] = $(this)[0].value
        }else{
            delete jsonOut['Proc']['DeepSORT']['model_path']
        }
    })

    $("input[name='deepsort_max_age']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['max_age'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['DeepSORT']['max_age']
        }
    })

    $("input[name='deepsort_min_hits']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['min_hits'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['DeepSORT']['min_hits']
        }
    })

    $("input[name='deepsort_iou_thresh']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['iou_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['DeepSORT']['iou_thresh']
        }
    })

    $("input[name='deepsort_max_cosine_dist']").change(function(){
        if ($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['max_cosine_dist'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['DeepSORT']['max_cosine_dist']
        }
    })

    $("input[name='deepsort_avg_conf_cb']").change(function(){
        jsonOut['Proc']['DeepSORT']['avg_det_conf'] = this.checked
        if (!this.checked){
            delete jsonOut['Proc']['DeepSORT']['avg_det_conf_thresh']
        }
    })

    $("input[name='deepsort_avg_conf_thresh']").change(function(){
        if($(this)[0].value){
            jsonOut['Proc']['DeepSORT']['avg_det_conf_thresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['DeepSORT']['avg_det_conf_thresh']
        }
    })

    $("input[name='deepsort_most_common_class_cb']").change(function(){
        jsonOut['Proc']['DeepSORT']['most_common_class'] = this.checked
    })
}

function generateJSONEvaluator() {
    $("select[name='select_evaluator_type']").change(function(){
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Proc']['Evaluator']['type'] = val
        } else {
            delete jsonOut['Proc']['Evaluator']['type']
        }
    })

    $("select[name='multiple_select_metrics']").change(function(){
        metrics = []
        $(this).children(':selected').each(function(){
            metrics.push($(this)[0].value)
        })
        jsonOut['Proc']['Evaluator']['metrics'] = metrics
    })
}

function generateJSONPreproc() {
    $("input[name='ar_type']").change(function(){
        if (this.value === "letterbox") {
            jsonOut['Preproc']['border'] = {'centered': false}
        }else{
            delete jsonOut['Preproc']['border']
        }
    })
    
    $("input[name='center_cb']").change(function(){
        if (jsonOut['Preproc']['border']){
            jsonOut['Preproc']['border'] = {'centered': this.checked}
        }
    })

    $("input[name='resize_width'], input[name='resize_height']").change(function(){
        var width = Number($("input[name='resize_width']")[0].value) || 416
        var height = Number($("input[name='resize_height']")[0].value) || 416
        jsonOut['Preproc']['resize'] = {
            'width': width,
            'height': height
        }
    })

    $("input[name='roi_pre_cb']").change(function(){
        if (!this.checked) {
            delete jsonOut['Preproc']['roi']
        }
    })

    $("input[name='roi_pre_file_path']").change(function(){
        jsonOut['Preproc']['roi'] = {}
        jsonOut['Preproc']['roi']['path'] = $("input[name='roi_pre_file_path']")[0].value
    })

    $("input[name='roi_pre_coords']").change(function(){
        jsonOut['Preproc']['roi'] = {}
        jsonOut['Preproc']['roi']['coords'] = $("input[name='roi_pre_coords']")[0].value
    })
}

function generateJSONPostproc() {
    $("select[name='select_NMS_implem']").change(function(){
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Postproc']['nms'] = {}
            jsonOut['Postproc']['nms']['pref_implem'] = $(this).children(':selected')[0].value
            jsonOut['Postproc']['nms']['nms_thresh'] = 0.45
        }else {
            delete jsonOut['Postproc']['nms']
        }
    })

    $("input[name='post_nms_thresh']").change(function(){
        if (this.value){
            jsonOut['Postproc']['nms']['nms_thresh'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['nms']['nms_thresh']
        }
    })

    $("input[name='post_conf_thresh']").change(function(){
        if (this.value){
            jsonOut['Postproc']['min_conf'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['min_conf']
        }
    })

    $("input[name='post_max_height']").change(function(){
        if (this.value){
            jsonOut['Postproc']['max_height'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['max_height']
        }
    })

    $("input[name='post_min_height']").change(function(){
        if (this.value){
            jsonOut['Postproc']['min_height'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['min_height']
        }
    })

    $("input[name='post_max_width']").change(function(){
        if (this.value){
            jsonOut['Postproc']['max_width'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['max_width']
        }
    })
    
    $("input[name='post_min_width']").change(function(){
        if (this.value){
            jsonOut['Postproc']['min_width'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['min_width']
        }
    })

    $("input[name='post_min_area']").change(function(){
        if (this.value){
            jsonOut['Postproc']['min_area'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['min_area']
        }
    })

    $("input[name='post_resize_width'], input[name='post_resize_height']").change(function(){
        var width = Number($("input[name='post_resize_width']")[0].value)
        var height = Number($("input[name='post_resize_height']")[0].value)
        jsonOut['Preproc']['resize_results'] = {
            'width': width,
            'height': height
        }
    })

    $("input[name='post_top_k']").change(function(){
        if (this.value){
            jsonOut['Postproc']['top_k']  = Number(this.value)
        }else{
            delete jsonOut['Postproc']['top_k'] 
        }
    })

    
    $("input[name='post_coi']").change(function(){
        if (this.value){
            jsonOut['Postproc']['coi']  = this.value
        }else{
            delete jsonOut['Postproc']['top_k'] 
        }
    })

    $("input[name='roi_post_cb']").change(function(){
        if (!this.checked) {
            delete jsonOut['Postproc']['roi']
        }
    })

    $("input[name='roi_post_file_path']").change(function(){
        jsonOut['Postproc']['roi'] = {}
        jsonOut['Postproc']['roi']['path'] = $("input[name='roi_post_file_path']")[0].value
    })

    $("input[name='roi_post_coords']").change(function(){
        jsonOut['Postproc']['roi'] = {}
        jsonOut['Postproc']['roi']['coords'] = $("input[name='roi_post_coords']")[0].value
    })

    $("input[name='max_outside_roi_thresh']").change(function(){
        if (this.value){
            jsonOut['Postproc']['roi']['max_outside_roi_thresh'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['roi']['max_outside_roi_thresh']
        }
    })

}

function copyToClipboard() {
    var copyText = $("#text_area_json")[0]
    copyText.focus();
    copyText.select();
    document.execCommand("copy");
    $("#text_area_json_btn")[0].innerHTML = "Copied!"

    setTimeout(
        function(){ $("#text_area_json_btn")[0].innerHTML = "Copy to clipboard!" },
        2000
      );
}

function saveTextAsFile(textToWrite, fileNameToSaveAs)
{
    var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'}); 
    var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    if (window.webkitURL != null){
        // Chrome allows the link to be clicked
        // without actually adding it to the DOM.
        downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
    }
    else{
        // Firefox requires the link to be added to the DOM
        // before it can be clicked.
        downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
        downloadLink.onclick = destroyClickedElement;
        downloadLink.style.display = "none";
        document.body.appendChild(downloadLink);
    }

    downloadLink.click();
}

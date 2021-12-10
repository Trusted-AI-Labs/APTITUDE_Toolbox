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
        } else{
            $(".yolo").attr("hidden", true)
        }
    })

    $("select[name='select_detector_implem']").change(function(){
        val = $(this).children(':selected')[0].value 
        if (val === "cv2-DM" || val === "cv2-RM") {
            $(".opencv").attr("hidden", false)
            if (val === "cv2-DM") {
                $(".cv2-DM").attr("hidden", false)
            }else{
                $(".cv2-DM").attr("hidden", true)
            }
        } else{
            $(".opencv").attr("hidden", true)
        }
    })

    $("#ocv_gpu_cb").change(function(){
        if (this.checked) {
            $(".ocv_gpu").attr("hidden", false)
        } else {
            $(".ocv_gpu").attr("hidden", true)
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
        if (val == "Centroid"){
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", true)
            $(".centroid").attr("hidden", false)
        } else if (val  === "SORT") {
            $(".sort").attr("hidden", false)
            $(".deepsort").attr("hidden", true)
            $(".centroid").attr("hidden", true)
        } else if (val === "DeepSORT") {
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", false)
            $(".centroid").attr("hidden", true)
        } else {
            $(".sort").attr("hidden", true)
            $(".deepsort").attr("hidden", true)
            $(".centroid").attr("hidden", true)
        }
    })

    $("select[name='select_detector_implem']").change(function(){
        val = $(this).children(':selected')[0].value 
        if (val === "cv2-DM" || val === "cv2-RM") {
            $(".opencv").attr("hidden", false)
            if (val === "cv2-DM") {
                $(".cv2-DM").attr("hidden", false)
            }else{
                $(".cv2-DM").attr("hidden", true)
            }
        } else{
            $(".opencv").attr("hidden", true)
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

    $("input[name='roi_cb']").change(function(){
        if (this.checked) {
            $(".roi").attr("hidden", false);
        }
        else {
            $(".roi").attr("hidden", true);
        }
    })

    $("input[name='roi_type']").change(function(){
        if (this.value === "file") {
            $(".roi_file").attr("hidden", false);
            $(".roi_coords").attr("hidden", true);
        }
        else {
            $(".roi_file").attr("hidden", true);
            $(".roi_coords").attr("hidden", false);
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
}

function generateJSONVideo() {
    $("input[name='path']").change(function(){
        jsonOut['Video']['path'] = this.value
        console.log(jsonOut)
    })
    
    $("input[name='async_cb']").change(function(){
        jsonOut['Video']['async'] = this.checked
        console.log(jsonOut)
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
            if (val.startsWith("cv2")){
                jsonOut['Proc']['cv2'] = {}
            }
        } else {
            delete jsonOut['Proc'][type]['pref_implem']
            delete jsonOut['Proc']['cv2']
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
            jsonOut['Proc']['YOLO']['nms_tresh'] = Number($(this)[0].value)
        }else{
            delete jsonOut['Proc']['nms']['nms_tresh'] 
        }
    })

    $("input[name='ocv_gpu_cb']").change(function(){
        jsonOut['Proc']['cv2']['GPU'] = this.checked
        if (!this.checked){
            $("input[name='ocv_hp_cb']").prop("checked", false);
            jsonOut['Proc']['cv2']['half_precision'] = false
        }
    })

    $("input[name='ocv_hp_cb']").change(function(){
        jsonOut['Proc']['cv2']['half_precision'] = this.checked
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

    $("input[name='roi_cb']").change(function(){
        if (!this.checked) {
            delete jsonOut['Preproc']['roi']
        }
    })

    $("input[name='roi_file_path']").change(function(){
        jsonOut['Preproc']['roi'] = {}
        jsonOut['Preproc']['roi']['path'] = $("input[name='roi_file_path']")[0].value
    })

    $("input[name='roi_coords']").change(function(){
        jsonOut['Preproc']['roi'] = {}
        jsonOut['Preproc']['roi']['coords'] = $("input[name='roi_coords']")[0].value
    })
}

function generateJSONPostproc() {
    $("select[name='select_NMS_implem']").change(function(){
        var val = $(this).children(':selected')[0].value
        if (val !== ""){
            jsonOut['Postproc']['nms'] = {}
            jsonOut['Postproc']['nms']['pref_implem'] = $(this).children(':selected')[0].value
            jsonOut['Postproc']['nms']['nms_tresh'] = 0.45
        }else {
            delete jsonOut['Postproc']['nms']
        }
    })

    $("input[name='post_nms_thresh']").change(function(){
        if (this.value){
            jsonOut['Postproc']['nms']['nms_tresh'] = Number(this.value)
        }else{
            delete jsonOut['Postproc']['nms']['nms_tresh'] 
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

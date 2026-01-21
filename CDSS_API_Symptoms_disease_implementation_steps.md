## Following Changes to make in existing Code

### Step 1: Open application\views\admin\patient\appointment.php and add following lines:

####  Javascript Section:Add the following lines at the end after closing script (</script) tag OR can be added under exisiting Script Section:
```
<!-- Added By Raushan For Symptoms Selection Using DropDown -->

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.17.9/tagify.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/tagify/4.17.9/tagify.min.js"></script>

<script>
    var input = document.querySelector("#chiefcomplaint");

    var tagify = new Tagify(input, {
        whitelist: [],
        dropdown: {
            maxItems: 10,
            enabled: 1,
            closeOnSelect: false
        },
        addTagOnBlur: true,

// converts Tagify JSON tocomma-separated string

        originalInputValueFormat: valuesArr =>
            valuesArr.map(item => item.value).join(',')
    });

    tagify.on('input', function(e) {
        var value = e.detail.value;
        if (value.length < 1) return;

        fetch('<?php echo site_url("admin/symptoms/fetch_symptoms"); ?>?term=' + encodeURIComponent(value))
            .then(res => res.json())
            .then(function(data) {
                tagify.settings.whitelist = data;
                tagify.dropdown.show(value);
            });
    });
</script>
```

####  CSS/Style Sction:Add the following lines after existing closing style (/style) tag OR can be added under existing style section:
```
<style>
.tagify__tag {
        background: #3b82f6;
        color: #fff;
        border-radius: 4px;
        font-size: 13px;
    }

.tagify {
        min-height: 38px;
        height: auto !important;
        overflow: visible;
        border: 1px solid #d1d5db;
        padding: 4px;
    }
.chief-complaint-input {
        min-height: 38px;
        padding: 6px 10px;
        font-size: 14px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .chief-complaint-input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
    }

</style>
```
### Step 2: In application\views\admin\patient\appointment.php  add ID to symptoms textarea element:

#### Search for "lang->line('chief_complaint')" below this line we will find following textarea declaration:
```
<textarea style="height: 28px;" name="symptoms" class="form-control" ><?php echo set_value('address'); ?></textarea>
```
#### Change this to:
```
<textarea id="chiefcomplaint"  style="height: 28px;" name="symptoms" class="form-control" ><?php echo set_value('address'); ?></textarea>
```
### Step 3: Create new Folder named 'Cdss_Api' under application/controllers

### Step 4: Create new Controller file named 'Symptoms_Disease_API.php' under 'Cdss_Api' folder

### Step 5: Add the following lines in 'Symptoms_Disease_API.php' Controller file:
```
<?php
defined('BASEPATH') OR exit('No direct Script access allowed');
class Symptoms extends Admin_Controller{

public function __construct(){
    parent::__construct();
    $this->load->model('symptoms_model');

}
public function fetch_symptoms(){

    $term = $this->input->get('term', true);

    if(empty($term)){
        echo json_encode([]);
        return;
    }
    
    $data = $this->db->select('symptoms_name')->like('symptoms_name',$term)->order_by('symptoms_name', 'ASC')->limit(10)->get('symptoms_data');

    $result = [];
    foreach($data->result() as $row){
        $result[] = $row->symptoms_name;
    }



     $this->output
            ->set_content_type('application/json')
            ->set_output(json_encode($result));
}


 
    public function predictDisease()
    {
        $symptoms = $this->input->post('symptoms');

//         $symptoms = $this->input->post('symptoms');
// if(empty($symptoms)){
//     $symptoms = $this->input->get('symptoms', true);
// }

        if (empty($symptoms)) {
            echo "<h1>No symptoms selected!</h1>";
            return;
        }

        $symptoms_array = array_map('trim', explode(',', $symptoms));
        $data = json_encode(['symptoms' => $symptoms_array]);

        $ch = curl_init('http://155.248.254.195:6000/predict');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $data);

        $response = curl_exec($ch);

        if ($response === false) {
            echo "<h1>Error contacting Flask API: " . curl_error($ch) . "</h1>";
            curl_close($ch);
            return;
        }

        curl_close($ch);

        $result = json_decode($response, true);

        if (!isset($result['predictions']) || !is_array($result['predictions'])) {
            echo "<h1>Error in prediction! Raw response:</h1><pre>" . htmlspecialchars($response) . "</pre>";
            return;
        }

        arsort($result['predictions']);

        $labels = ["Strong Match", "Moderate Match", "Fair Match", "Low Match"];
        $diseases = array_keys($result['predictions']);
        $probs    = array_values($result['predictions']);

        echo "<h1>Predicted Diseases:</h1><ul>";

        foreach ($diseases as $i => $disease) {
            $prob = $probs[$i];
            $matchLabel = $labels[$i] ?? $labels[3];

            echo "<li><strong>" . htmlspecialchars($disease) . "</strong>: {$matchLabel} ({$prob}%)</li>";
        }

        echo "</ul>";
    }

public function sendProvisional()
{
    $provisional = $this->input->post('provisional');

    if (empty($provisional)) {
        echo json_encode(['error' => 'No provisional diagnosis provided']);
        return;
    }

    $data = [
        'disease_name' => $provisional
    ];

    $jsonData = json_encode($data);

    $flask_url = 'http://155.248.254.195:5000/extract';
    $ch = curl_init($flask_url);

    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonData);

    $response = curl_exec($ch);

    if ($response === false) {
        echo json_encode([
            'error'   => 'Error contacting Flask API',
            'details' => curl_error($ch)
        ]);
        curl_close($ch);
        return;
    }

    curl_close($ch);

    $decodedResponse = json_decode($response, true);

    // Save in session 
    $this->session->set_userdata('treatment_data', $decodedResponse);

    echo $response;
}

public function sendSymptomsAndDiagnosis()
{
    
    $chiefComplaint = $this->input->post('chiefcomplaint'); 
    $provisionalDiagnosis = $this->input->post('initial_diagnosis'); 

    // Convert comma-separated strings to arrays
    $symptoms_array = !empty($chiefComplaint)
        ? array_map('trim', explode(',', $chiefComplaint))
        : [];

    $diagnosis_array = !empty($provisionalDiagnosis)
        ? array_map('trim', explode(',', $provisionalDiagnosis))
        : [];

    // Prepare final data array
    $final_data = [
        'symptoms' => $symptoms_array,
        'final_diagnosis_by_doctor' => $diagnosis_array
    ];

    // Convert data to JSON
    $jsonData = json_encode($final_data);

    // cURL request to  API
    $ch = curl_init('http://155.248.254.195:6000/receive');
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $jsonData);

    $response = curl_exec($ch);

    if ($response === false) {
        echo json_encode([
            'error' => 'Error contacting Flask API',
            'details' => curl_error($ch)
        ]);
        curl_close($ch);
        return;
    }

    curl_close($ch);

    
    echo $response;
}


}
```
